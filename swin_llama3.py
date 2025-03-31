import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
from swin_transformer import swin_t
from llama3_transformer_block import *

# Giả sử các hằng số này được định nghĩa ở đâu đó
BATCH_SIZE = 32  # Ví dụ
MAX_EPOCH = 100
LEARNING_RATE = 0.001
EMBED_DIM = 768
MLP_SCALE = 4
NUM_HEAD = 12
NUM_KV_HEAD = 2
HEAD_DIM = EMBED_DIM // NUM_HEAD
DROPOUT = 0.1
MAX_SEQUENCE = 128
ROPE_BASE = 10000
EPS_NORM = 1e-5
TEMPERATURE = 0.7
TOP_P = 0.9
MILESTONES = [0.5, 0.75]  # Tỷ lệ epoch để giảm LR
REDUCE_LR_FACTOR = 0.1
MODEL_NAME = "ImageCaptioning"

class ImageCaptioning(nn.Module):
    def __init__(self):
        super(ImageCaptioning, self).__init__()

        self.batch_size = BATCH_SIZE
        self.max_epoch = MAX_EPOCH
        self.lr = LEARNING_RATE

        # MLP
        MLP = FeedForward(
            gate_proj=nn.Linear(EMBED_DIM, int(EMBED_DIM * MLP_SCALE), bias=False),
            down_proj=nn.Linear(int(EMBED_DIM * MLP_SCALE), EMBED_DIM, bias=False),
            up_proj=nn.Linear(EMBED_DIM, int(EMBED_DIM * MLP_SCALE), bias=False),
        )

        # Feature Extractor (Swin Transformer)
        FEATURE_EXTRACTOR = swin_t(num_classes=EMBED_DIM)

        # Self Attention cho Encoder
        SELF_ATTENTION = CausalSelfAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEAD,
            num_kv_heads=NUM_KV_HEAD,
            head_dim=HEAD_DIM,
            q_proj=nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
            k_proj=nn.Linear(EMBED_DIM, NUM_KV_HEAD * HEAD_DIM, bias=False),
            v_proj=nn.Linear(EMBED_DIM, NUM_KV_HEAD * HEAD_DIM, bias=False),
            output_proj=nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
            pos_embeddings=RotaryPositionalEmbedding(
                dim=HEAD_DIM, max_seq_len=1, base=ROPE_BASE),
            max_seq_len=1,
            attn_dropout=DROPOUT,
        )

        ENCODER_LAYER = TransformerEncoderLayer(
            attn=SELF_ATTENTION,
            mlp=copy.deepcopy(MLP),
            sa_norm=RMSNorm(dim=EMBED_DIM, eps=EPS_NORM),
            mlp_norm=RMSNorm(dim=EMBED_DIM, eps=EPS_NORM),
        )

        self.encoder = TransformerEncoder(
            feature_extractor=FEATURE_EXTRACTOR,
            layer=ENCODER_LAYER,
            num_layers=NUM_LAYER,
            max_seq_len=MAX_SEQUENCE,
            num_heads=NUM_HEAD,
            head_dim=HEAD_DIM,
            norm=RMSNorm(EMBED_DIM, eps=EPS_NORM),
        )

        # Decoder (LLaMA 3)
        TOKEN_EMBEDDING = nn.Embedding(len(Tokenizer.decoder), EMBED_DIM)
        ROPE = RotaryPositionalEmbedding(
            dim=HEAD_DIM, max_seq_len=MAX_SEQUENCE, base=ROPE_BASE)
        
        SELF_ATTENTION_1 = CausalSelfAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEAD,
            num_kv_heads=NUM_KV_HEAD,
            head_dim=HEAD_DIM,
            q_proj=nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
            k_proj=nn.Linear(EMBED_DIM, NUM_KV_HEAD * HEAD_DIM, bias=False),
            v_proj=nn.Linear(EMBED_DIM, NUM_KV_HEAD * HEAD_DIM, bias=False),
            output_proj=nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
            pos_embeddings=ROPE,
            max_seq_len=MAX_SEQUENCE,
            attn_dropout=DROPOUT,
        )
        
        SELF_ATTENTION_2 = CausalSelfAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEAD,
            num_kv_heads=NUM_KV_HEAD,
            head_dim=HEAD_DIM,
            q_proj=nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
            k_proj=nn.Linear(EMBED_DIM, NUM_KV_HEAD * HEAD_DIM, bias=False),
            v_proj=nn.Linear(EMBED_DIM, NUM_KV_HEAD * HEAD_DIM, bias=False),
            output_proj=nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
            pos_embeddings=ROPE,
            max_seq_len=MAX_SEQUENCE,
            attn_dropout=DROPOUT,
        )
        
        DECODER_LAYER = TransformerDecoderLayer(
            attn1=SELF_ATTENTION_1,
            attn2=SELF_ATTENTION_2,
            mlp=copy.deepcopy(MLP),
            sa_norm_x1=RMSNorm(dim=EMBED_DIM, eps=EPS_NORM),
            sa_norm_x2=RMSNorm(dim=EMBED_DIM, eps=EPS_NORM),
            mlp_norm=RMSNorm(dim=EMBED_DIM, eps=EPS_NORM),
        )
        
        OUT_PROJECTION = nn.Linear(EMBED_DIM, len(Tokenizer.decoder), bias=False)
        self.decoder = TransformerDecoder(
            tok_embedding=TOKEN_EMBEDDING,
            layer=DECODER_LAYER,
            num_layers=NUM_LAYER,
            max_seq_len=MAX_SEQUENCE,
            num_heads=NUM_HEAD,
            head_dim=HEAD_DIM,
            norm=RMSNorm(EMBED_DIM, eps=EPS_NORM),
            output=OUT_PROJECTION,
        )

    def forward(self, image, caption):
        image_feature = self.encoder(image)
        return self.decoder(caption, image_feature)

    def captionize(self, image, temperature=TEMPERATURE, top_p=TOP_P):
        assert image.shape[0] == 1

        self.encoder.setup_caches(max_batch_size=1)
        encoder_feat = self.encoder(
            image,
            input_pos=torch.tensor([0], device=image.device),
        )
        self.encoder.clear_caches()

        self.decoder.setup_caches(max_batch_size=1)

        pred_token = Tokenizer.encoder[START_TOKEN]
        token = [pred_token] + [Tokenizer.encoder[PAD_TOKEN]] * (MAX_SEQUENCE - 1)
        for index in range(MAX_SEQUENCE - 1):
            caption = torch.LongTensor([token[:index + 1]]).to(image.device)
            pred_token = self.decoder(
                caption,
                encoder_feat,
                input_pos=torch.arange(index + 1, device=image.device),
            )

            if temperature > 0:
                pred_token = (pred_token / temperature).softmax(-1)[0, -1]  # Chỉ lấy token cuối
                psort, pidx = torch.sort(pred_token, dim=-1, descending=True)
                psum = torch.cumsum(psort, dim=-1)
                psort[psum - psort > top_p] = 0.
                psort.div_(psort.sum(dim=-1, keepdim=True))
                pred_token = torch.multinomial(psort, num_samples=1)
                pred_token = torch.gather(pidx, -1, pred_token).item()
            else:
                pred_token = pred_token.softmax(-1).argmax(-1)[-1].item()

            token[index + 1] = pred_token
            if pred_token == Tokenizer.encoder[END_TOKEN]:
                break

        self.decoder.clear_caches()
        return self.postprocess_text(Tokenizer.decode(token))

    def postprocess_text(self, text):
        text = text.replace(START_TOKEN, "")
        text = text.replace(END_TOKEN, "")
        text = text.replace(PAD_TOKEN, "")
        text = re.sub(r'\s([,.!?])', r'\1', text)
        text = '. '.join(map(lambda s: s.strip().capitalize(), text.split('.')))
        return text

# Hàm huấn luyện
def train_model(model, train_loader, val_loader, num_epochs, device, save_path, log_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=Tokenizer.encoder[PAD_TOKEN])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * ms) for ms in MILESTONES],
        gamma=REDUCE_LR_FACTOR,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    with open(log_path, "w") as log_file:
        log_file.write(f"🔹 Bắt đầu training: {save_path}\n")

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for image, caption in train_loader:
                image, caption = image.to(device), caption.to(device)
                optimizer.zero_grad()
                pred = model(image, caption[:, :-1])
                pred = pred.view(-1, pred.shape[-1])
                caption = caption[:, 1:].reshape(-1)
                loss = criterion(pred, caption)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), math.log2(math.sqrt(math.e * math.tau) * math.pi))
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for image, caption in val_loader:
                    image, caption = image.to(device), caption.to(device)
                    pred = model(image, caption[:, :-1])
                    pred = pred.view(-1, pred.shape[-1])
                    caption = caption[:, 1:].reshape(-1)
                    loss = criterion(pred, caption)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Lưu mô hình tốt nhất
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, save_path)

            # Ghi log
            log = f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            print(log, end="")
            log_file.write(log)

            scheduler.step()

        # Vẽ biểu đồ loss
        plt.plot(train_losses, color="r", label="train")
        plt.plot(val_losses, color="b", label="validation")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"experiment/training/{MODEL_NAME}_loss_plot.png")
        plt.clf()

        final_log = f"""
==================================================
📌 KẾT QUẢ TỐT NHẤT - Epoch {val_losses.index(best_val_loss) + 1}:
- Best Val Loss: {best_val_loss:.6f}
==================================================
"""
        print(final_log)
        log_file.write(final_log)

# Hàm đánh giá (test)
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    test_rogue = ROUGEScore()
    rogue1_fmeasure = []

    with torch.no_grad():
        for image, caption in test_loader:
            image, caption = image.to(device), caption.to(device)
            for i in range(image.shape[0]):
                pred = model.captionize(image[i].unsqueeze(0))
                target = model.postprocess_text(Tokenizer.decode(caption[i].cpu().numpy().tolist()))
                rogue1 = test_rogue(pred, target)['rouge1_fmeasure'].item()
                rogue1_fmeasure.append(rogue1)

    avg_rogue1 = np.mean(rogue1_fmeasure)
    print(f"ROGUE-1 F-measure: {avg_rogue1:.4f}")
    return avg_rogue1

# Khởi tạo và chạy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptioning()
train_loader = torch.utils.data.DataLoader(
    TrainDataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = torch.utils.data.DataLoader(
    ValDataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    TestDataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

train_model(model, train_loader, val_loader, MAX_EPOCH, device, "best_model.pth", "training_log.txt")
evaluate_model(model, test_loader, device)