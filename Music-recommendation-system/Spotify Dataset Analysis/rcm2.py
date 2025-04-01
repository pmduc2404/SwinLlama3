import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import lightgbm as lgb
import cv2
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os
import time
from collections import Counter

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
        
        self.load_emotion_detector()
        
        self.EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
        
        self.user_history = []
        self.user_preferences = Counter()
        self.feedback_weights = {'loved': 2, 'liked': 1, 'disliked': -1, 'hated': -2}
        
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
        
        # Create clusters with more emotional nuance
        self.create_clusters()
        
        # Train a classifier model for emotion prediction
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
        
        # Apply PCA for improved clustering
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
    
    def load_emotion_detector(self):
        """Load facial emotion detection models with improved error handling"""
        print("Loading emotion detection models...")
        try:
            self.detection_model_path = 'haarcascade_frontalface_default.xml'
            self.emotion_model_path = 'final_model3.h5'
            
            # Check if model files exist
            if not os.path.exists(self.detection_model_path):
                print(f"Warning: Face detection model not found at {self.detection_model_path}")
                print("You'll need to download the Haar cascade file from OpenCV GitHub repository")
                
            if not os.path.exists(self.emotion_model_path):
                print(f"Warning: Emotion model not found at {self.emotion_model_path}")
                print("You'll need to provide a trained emotion recognition model")
                
            # Load models
            self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
            self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
            
            print("Emotion detection models loaded successfully")
        except Exception as e:
            print(f"Error loading emotion detection models: {e}")
            print("Falling back to manual emotion selection")
            self.face_detection = None
            self.emotion_classifier = None
    
    def detect_emotion(self, timeout=30):
        """
        Detect user's emotion with improved timeout and feedback
        
        Parameters:
        - timeout: Maximum time to wait for emotion detection (seconds)
        
        Returns:
        - detected_emotion: The detected emotion
        """
        if self.face_detection is None or self.emotion_classifier is None:
            print("Emotion detection models not available.")
            return self.manual_emotion_selection()
            
        print("Starting emotion detection. Press 'q' to stop...")
        detected_emotions = []
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera. Using manual emotion selection.")
            cap.release()
            return self.manual_emotion_selection()
            
        start_time = time.time()
        elapsed_time = 0
        
        while elapsed_time < timeout:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detection.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                
                # Normalize and reshape for model
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0
                
                # Make prediction
                predictions = self.emotion_classifier.predict(img_pixels)
                emotion_idx = np.argmax(predictions[0])
                emotion = self.EMOTIONS[emotion_idx]
                
                # Store detected emotion
                detected_emotions.append(emotion)
                
                # Display emotion on frame
                cv2.putText(frame, emotion, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display countdown
            elapsed_time = time.time() - start_time
            remaining = max(0, timeout - int(elapsed_time))
            cv2.putText(frame, f"Time: {remaining}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Determine most frequent emotion
        if detected_emotions:
            emotion_counts = Counter(detected_emotions)
            majority_emotion = emotion_counts.most_common(1)[0][0]
            print(f"Detected emotion: {majority_emotion}")
            return majority_emotion
        else:
            print("No emotions detected. Using manual selection.")
            return self.manual_emotion_selection()
    
    def manual_emotion_selection(self):
        """Allow manual selection of emotion if detection fails"""
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
        """
        Get music recommendations based on detected emotion with personalization
        
        Parameters:
        - emotion: The detected emotion
        - num_recommendations: Number of songs to recommend
        - personalize: Whether to personalize recommendations based on user history
        
        Returns:
        - recommendations: DataFrame of recommended songs
        """
        # Get suitable clusters for the emotion
        suitable_clusters = self.EMOTION_CLUSTER_MAP.get(emotion, [0])
        
        # Filter data by suitable clusters
        filtered_data = self.data[self.data['emotion_cluster'].isin(suitable_clusters)]
        
        # Personalization based on user history and preferences
        if personalize and self.user_history:
            # Calculate personalization scores based on user preferences
            artist_weights = {}
            for artist, weight in self.user_preferences.items():
                artist_weights[artist] = weight
            
            # Apply personalization weights
            filtered_data['personalization_score'] = filtered_data.apply(
                lambda row: self._calculate_personalization_score(row, artist_weights), 
                axis=1
            )
            
            # Sort by popularity and personalization score
            recommendations = filtered_data.sort_values(
                by=['personalization_score', 'popularity'], 
                ascending=[False, False]
            ).head(num_recommendations)
        else:
            # If no personalization, sort by popularity only
            recommendations = filtered_data.sort_values(
                by='popularity', ascending=False
            ).head(num_recommendations)
        
        # Select only necessary columns for output
        result = recommendations[['song_name', 'artists', 'popularity', 'emotion_cluster']]
        
        # Format artist names (remove brackets and quotes)
        result['artists'] = result['artists'].apply(self._format_artists)
        
        return result
    
    def _format_artists(self, artists_str):
        """Format artist names for better display"""
        # Remove brackets, quotes and split by commas
        try:
            # Remove brackets and quotes
            cleaned = artists_str.strip('[]').replace("'", "")
            # Split by comma and join with &
            artists = [a.strip() for a in cleaned.split(',')]
            return " & ".join(artists)
        except:
            return artists_str
    
    def _calculate_personalization_score(self, row, artist_weights):
        """Calculate personalization score based on user preferences"""
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
        """
        Add user feedback for a song to improve future recommendations
        
        Parameters:
        - song_name: Name of the song
        - artists: List or string of artists
        - feedback_type: One of ('loved', 'liked', 'disliked', 'hated')
        """
        # Validate feedback type
        if feedback_type not in self.feedback_weights:
            raise ValueError(f"Feedback type must be one of {list(self.feedback_weights.keys())}")
        
        # Get weight for this feedback
        weight = self.feedback_weights[feedback_type]
        
        # Process artists
        if isinstance(artists, str):
            try:
                artists = eval(artists) if artists.startswith('[') else [artists]
            except:
                artists = [artists]
        
        # Update user preferences
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
        """Visualize the emotion clusters with PCA for dimensionality reduction"""
        # Apply PCA to reduce dimensions for visualization
        X = MinMaxScaler().fit_transform(self.data[self.audio_features])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': self.data['emotion_cluster']
        })
        
        # Plot
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
        
        # Visualize feature importance
        self._visualize_feature_importance()
    
    def _visualize_feature_importance(self):
        """Visualize feature importance for emotion prediction"""
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
        """Run the emotion-based recommendation system"""
        print("\n===== Emotion-Based Music Recommender =====\n")
        
        while True:
            print("\nOptions:")
            print("1. Get music recommendations based on detected emotion")
            print("2. Get music recommendations based on manual emotion selection")
            print("3. Add feedback on previous recommendations")
            print("4. Visualize emotion clusters")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                emotion = self.detect_emotion()
                recommendations = self.get_recommendations(emotion)
                print("\nRecommended songs based on your emotion:")
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


# Example usage
if __name__ == "__main__":
    recommender = EmotionBasedMusicRecommender()
    recommender.run()