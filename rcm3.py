import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os
import time
from collections import Counter
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import re

class EmotionBasedMusicRecommender:
    def __init__(self, data_path=r'D:\NCKHSV.2024-2025\SwinLlama3\Music-recommendation-system\Spotify Dataset Analysis\data.csv.zip', n_clusters=4):
        self.n_clusters = n_clusters
        self.data_path = data_path
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
        
        self.load_data()
        
        self.EMOTION_CLUSTER_MAP = {
            "happy": [0, 2],     # Energetic happy and calm happy
            "sad": [1, 3],       # Melancholic sad and depressive sad
            "angry": [0, 3],     # High energy clusters
            "neutral": [2],      # Mid-energy cluster
            "surprised": [0, 2], # Energetic clusters
            "fearful": [1, 3],   # Low valence clusters
            "disgusted": [3]     # Low valence high energy
        }
        
        self.EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
        
        self.user_history = []
        self.user_preferences = Counter()
        self.feedback_weights = {'loved': 2, 'liked': 1, 'disliked': -1, 'hated': -2}
        
        self.load_image_analyzer()
        
    def load_data(self):
        print("Loading music dataset...")
        try:
            if (self.model_dir / 'processed_data.pkl').exists():
                with open(self.model_dir / 'processed_data.pkl', 'rb') as f:
                    data_dict = pickle.load(f)
                    self.data = data_dict['data']
                    self.kmeans = data_dict['kmeans']
                    self.audio_features = data_dict['audio_features']
                    self.model = data_dict['model']
                    print("Loaded preprocessed data and models successfully.")
                    return
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            print("Processing data from scratch...")
        
        self.data = pd.read_csv(self.data_path, compression='zip')
        self.data.drop_duplicates(inplace=True, subset=['name'])
        
        self.data['song_name'] = self.data['name']
        
        self.audio_features = [
            'danceability', 'energy', 'valence', 'loudness', 
            'acousticness', 'instrumentalness', 'tempo', 'speechiness', 'liveness'
        ]
        
        self.create_clusters()
        self.train_model()
        
        data_dict = {
            'data': self.data,
            'kmeans': self.kmeans,
            'audio_features': self.audio_features,
            'model': self.model
        }
        with open(self.model_dir / 'processed_data.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        
    def create_clusters(self):
        print(f"Creating {self.n_clusters} emotion clusters...")
        
        X = MinMaxScaler().fit_transform(self.data[self.audio_features])
        
        pca = PCA(n_components=min(5, len(self.audio_features)))
        X_pca = pca.fit_transform(X)
        
        self.kmeans = KMeans(
            init="k-means++",
            n_clusters=self.n_clusters,
            random_state=42
        ).fit(X_pca)
        
        self.data['emotion_cluster'] = self.kmeans.labels_
        self.analyze_clusters()
        
    def analyze_clusters(self):
        print("Analyzing emotion clusters...")
        
        cluster_means = self.data.groupby('emotion_cluster')[self.audio_features].mean()
        
        # Interpret emotional characteristics based on audio features
        # High energy + high valence = Happy/Excited
        # Low energy + low valence = Sad/Depressed
        # High energy + low valence = Angry/Tense
        # Low energy + high valence = Calm/Peaceful
        
        print("Cluster emotional characteristics:")
        for cluster in range(self.n_clusters):
            means = cluster_means.loc[cluster]
            emotion_desc = ""
            
            if means['energy'] > 0.5 and means['valence'] > 0.5:
                emotion_desc = "Happy/Excited"
            elif means['energy'] < 0.5 and means['valence'] < 0.5:
                emotion_desc = "Sad/Melancholic"
            elif means['energy'] > 0.5 and means['valence'] < 0.5:
                emotion_desc = "Angry/Tense"
            else:
                emotion_desc = "Calm/Peaceful"
                
            print(f"Cluster {cluster}: {emotion_desc}")
            print(f"  Energy: {means['energy']:.2f}, Valence: {means['valence']:.2f}")
            print(f"  Danceability: {means['danceability']:.2f}, Acousticness: {means['acousticness']:.2f}")
            print()
        
    def train_model(self):
        """Train a model to predict emotion clusters from audio features"""
        print("Training emotion prediction model...")
        
        # Prepare data
        X = self.data[self.audio_features]
        y = self.data['emotion_cluster']
        
        # Train a LightGBM model (better accuracy than previous implementation)
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=self.n_clusters,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=42
        )
        self.model.fit(X, y)
        
        # Print feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking for emotion prediction:")
        for f in range(len(self.audio_features)):
            print(f"{f+1}. {self.audio_features[indices[f]]} ({importances[indices[f]]})")
    
    def load_image_analyzer(self):
        """Load the Qwen2-VL model for image analysis"""
        print("Loading Qwen2-VL model for image analysis...")
        try:
            model_id = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Initialize the model
            self.vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=dtype, 
                device_map="auto" if device == "cuda" else None
            )
            self.vl_processor = AutoProcessor.from_pretrained(model_id)
            
            self.device = device
            print(f"Qwen2-VL model loaded successfully (using {device})")
        except Exception as e:
            print(f"Error loading Qwen2-VL model: {e}")
            print("Image analysis functionality will be limited")
            self.vl_model = None
            self.vl_processor = None
    
    def analyze_image_with_llm(self, image_path):
        """
        Analyze image using Qwen2-VL model to detect emotions
        
        Parameters:
        - image_path: Path to the image file
        
        Returns:
        - detected_emotion: The detected emotion
        """
        if self.vl_model is None or self.vl_processor is None:
            print("Qwen2-VL model not available. Using manual emotion selection.")
            return self.manual_emotion_selection()
        
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Prepare the conversation prompt asking for emotional analysis
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": """Analyze this image to determine the dominant emotional expression.

Step 1: Carefully examine the facial expressions (eyes, mouth, eyebrows, overall facial tension).
Step 2: Note any body language cues (posture, hand gestures, head position).
Step 3: Consider the image context and environment.
Step 4: If multiple people are present, focus on the most prominent or centered person.

Based on your analysis, classify the primary emotion as EXACTLY ONE of these categories:
- angry (signs: furrowed brows, tight jaw, intense stare)
- disgusted (signs: wrinkled nose, raised upper lip, squinted eyes)
- fearful (signs: widened eyes, raised eyebrows, tense features)
- happy (signs: genuine smile, crinkled eyes, relaxed face)
- neutral (signs: relaxed face, minimal expression, even tone)
- sad (signs: downturned mouth, drooping eyes, furrowed brow)
- surprised (signs: raised eyebrows, widened eyes, open mouth)

Format your response as:
"Dominant emotion: [EMOTION NAME]
Brief explanation: [2-3 sentences explaining the visual cues that indicate this emotion]"

It's critical to select ONLY ONE emotion from the provided list."""},
                    ],
                }
            ]
            
            text_prompt = self.vl_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.vl_processor(
                text=[text_prompt], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            
            inputs = inputs.to(self.device)
            
            print("Analyzing image with Qwen2-VL model...")
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32):
                    output_ids = self.vl_model.generate(
                        **inputs, 
                        max_new_tokens=384, 
                        do_sample=True, 
                        temperature=0.7, 
                        use_cache=True, 
                        top_k=50
                    )
            
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]):] 
                for i in range(len(inputs.input_ids))
            ]
            
            output_text = self.vl_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            
            print("VL Model Analysis:")
            print(output_text)
            
            dominant_pattern = r'dominant emotion:\s*(angry|disgusted|fearful|happy|neutral|sad|surprised)'
            match = re.search(dominant_pattern, output_text.lower(), re.IGNORECASE)
            if match:
                detected_emotion = match.group(1).lower()
                print(f"Detected emotion (formatted response): {detected_emotion}")
                return detected_emotion
                
            emotion_indicators = {
                'angry': [r'anger\b', r'angry\b', r'furious\b', r'irritated\b', r'rage\b'],
                'disgusted': [r'disgust\b', r'disgusted\b', r'revulsion\b', r'repulsed\b'],
                'fearful': [r'fear\b', r'fearful\b', r'afraid\b', r'scared\b', r'terrified\b', r'anxious\b'],
                'happy': [r'happy\b', r'happiness\b', r'joy\b', r'joyful\b', r'delighted\b', r'cheerful\b', r'pleased\b'],
                'neutral': [r'neutral\b', r'expressionless\b', r'impassive\b', r'stoic\b', r'emotionless\b'],
                'sad': [r'sad\b', r'sadness\b', r'unhappy\b', r'sorrowful\b', r'depressed\b', r'melancholy\b'],
                'surprised': [r'surprise\b', r'surprised\b', r'shocked\b', r'astonished\b', r'startled\b']
            }
            
            emotion_counts = {emotion: 0 for emotion in self.EMOTIONS}
            
            for emotion, patterns in emotion_indicators.items():
                for pattern in patterns:
                    matches = re.findall(pattern, output_text.lower())
                    emotion_counts[emotion] += len(matches)
            
            # Get emotion with highest count
            max_count = max(emotion_counts.values())
            if max_count > 0:
                # If there's a tie, prioritize emotions in this order
                priority_order = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
                top_emotions = [e for e, c in emotion_counts.items() if c == max_count]
                for emotion in priority_order:
                    if emotion in top_emotions:
                        print(f"Detected emotion (keyword frequency): {emotion}")
                        return emotion
            
            # Finally, fall back to simple mentions
            for emotion in self.EMOTIONS:
                if emotion.lower() in output_text.lower():
                    print(f"Detected emotion (simple mention): {emotion}")
                    return emotion
                
            # Default to neutral if no emotion is found
            print("No specific emotion detected, defaulting to neutral")
            return "neutral"
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return self.manual_emotion_selection()
    
    def detect_emotion_from_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return self.manual_emotion_selection()
            
        try:
            img = Image.open(image_path)
            img.verify()
            
            print(f"Analyzing image: {image_path}")
            
            detected_emotion = self.analyze_image_with_llm(image_path)
            return detected_emotion
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return self.manual_emotion_selection()
    
    def manual_emotion_selection(self):
        print("\nPlease select your current emotion:")
        for i, emotion in enumerate(self.EMOTIONS):
            print(f"{i+1}. {emotion.capitalize()}")
            
        while True:
            try:
                choice = int(input("Enter number (1-7): "))
                if 1 <= choice <= 7:
                    selected_emotion = self.EMOTIONS[choice-1]
                    print(f"Selected emotion: {selected_emotion}")
                    return selected_emotion
                else:
                    print("Invalid selection. Please enter a number between 1 and 7.")
            except ValueError:
                print("Please enter a valid number.")
    
    def get_recommendations(self, emotion, num_recommendations=10, personalize=True):
        suitable_clusters = self.EMOTION_CLUSTER_MAP.get(emotion, [0])
        
        filtered_data = self.data[self.data['emotion_cluster'].isin(suitable_clusters)]
        
        if personalize and self.user_history:
            artist_weights = {}
            for artist, weight in self.user_preferences.items():
                artist_weights[artist] = weight
            
            filtered_data['personalization_score'] = filtered_data.apply(
                lambda row: self._calculate_personalization_score(row, artist_weights), 
                axis=1
            )
            
            recommendations = filtered_data.sort_values(
                by=['personalization_score', 'popularity'], 
                ascending=[False, False]
            ).head(num_recommendations)
        else:
            recommendations = filtered_data.sort_values(
                by='popularity', ascending=False
            ).head(num_recommendations)
        
        result = recommendations[['song_name', 'artists', 'popularity', 'emotion_cluster']]
        
        result['artists'] = result['artists'].apply(self._format_artists)
        
        return result
    
    def _format_artists(self, artists_str):
        try:
            cleaned = artists_str.strip('[]').replace("'", "")
            artists = [a.strip() for a in cleaned.split(',')]
            return " & ".join(artists)
        except:
            return artists_str
    
    def _calculate_personalization_score(self, row, artist_weights):
        score = 0
        try:
            artists_str = row['artists']
            # Extract artists from the string
            artists = eval(artists_str) if isinstance(artists_str, str) else artists_str
            
            # Sum weights for all artists in the song
            for artist in artists:
                score += artist_weights.get(artist, 0)
        except:
            pass
        
        return score
    
    def add_user_feedback(self, song_name, artists, feedback_type):
        if feedback_type not in self.feedback_weights:
            raise ValueError(f"Feedback type must be one of {list(self.feedback_weights.keys())}")
        
        weight = self.feedback_weights[feedback_type]
        
        if isinstance(artists, str):
            try:
                artists = eval(artists) if artists.startswith('[') else [artists]
            except:
                artists = [artists]
        
        for artist in artists:
            self.user_preferences[artist] += weight
        
        # Add to user history
        self.user_history.append({
            'song_name': song_name,
            'artists': artists,
            'feedback': feedback_type,
            'timestamp': time.time()
        })
        
        print(f"Feedback recorded: {feedback_type} for '{song_name}'")
    
    def visualize_emotion_clusters(self):
        X = MinMaxScaler().fit_transform(self.data[self.audio_features])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': self.data['emotion_cluster']
        })
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=plot_df, 
            x='PC1', 
            y='PC2', 
            hue='Cluster',
            palette='viridis',
            alpha=0.7
        )
        plt.title('Emotion Clusters Visualization')
        plt.savefig('emotion_clusters.png')
        plt.close()
        
        print("Emotion clusters visualization saved to 'emotion_clusters.png'")
        
        self._visualize_feature_importance()
    
    def _visualize_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [self.audio_features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Emotion Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("Feature importance visualization saved to 'feature_importance.png'")

    def run(self):
        print("\n===== Emotion-Based Music Recommender =====\n")
        
        while True:
            print("\nOptions:")
            print("1. Get music recommendations from image analysis")
            print("2. Get music recommendations based on manual emotion selection")
            print("3. Add feedback on previous recommendations")
            print("4. Visualize emotion clusters")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                image_path = input("Enter the path to your image file: ")
                emotion = self.detect_emotion_from_image(image_path)
                recommendations = self.get_recommendations(emotion)
                print("\nRecommended songs based on detected emotion:")
                print(recommendations)
            
            elif choice == '2':
                emotion = self.manual_emotion_selection()
                recommendations = self.get_recommendations(emotion)
                print("\nRecommended songs based on your emotion:")
                print(recommendations)
                
            elif choice == '3':
                song_name = input("Enter song name: ")
                artists = input("Enter artist(s): ")
                print("\nFeedback options:")
                print("1. Loved")
                print("2. Liked")
                print("3. Disliked")
                print("4. Hated")
                
                feedback_choice = input("Enter your feedback (1-4): ")
                feedback_map = {
                    '1': 'loved',
                    '2': 'liked',
                    '3': 'disliked',
                    '4': 'hated'
                }
                
                if feedback_choice in feedback_map:
                    self.add_user_feedback(song_name, artists, feedback_map[feedback_choice])
                else:
                    print("Invalid feedback choice")
            
            elif choice == '4':
                self.visualize_emotion_clusters()
            
            elif choice == '5':
                print("Thank you for using the Emotion-Based Music Recommender!")
                break
                
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    recommender = EmotionBasedMusicRecommender()
    recommender.run()