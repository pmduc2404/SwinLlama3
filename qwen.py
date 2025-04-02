from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import torch

MODEL_ID = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=False
        )

else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        use_cache=False
        )

processor = AutoProcessor.from_pretrained(MODEL_ID)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

image = Image.open(r"D:\NCKHSV.2024-2025\SwinLlama3\output_1111111111.jpg")

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids  = model.generate(**inputs, max_new_tokens=384, do_sample=True, temperature=0.7, use_cache=True, top_k=50)

generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]
print(output_text)
