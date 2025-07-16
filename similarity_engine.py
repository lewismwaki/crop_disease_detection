import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DiseaseSimilarityEngine:
    def __init__(self, csv_path="disease_descriptions.csv"):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.csv_path = csv_path
        
    def load_and_vectorize(self):
        self.df = pd.read_csv(self.csv_path)
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['description'])
        
        return self
    
    #top K similar diseases using cosine similarity
    def find_similar_diseases(self, predicted_class, top_k=3):
        try:
            pred_idx = self.df[self.df['class_name'] == predicted_class].index[0]
            
            similarities = cosine_similarity(
                self.tfidf_matrix[pred_idx], 
                self.tfidf_matrix
            ).flatten()
            
            similar_indices = similarities.argsort()[::-1][1:top_k+1]
            
            results = []
            for idx in similar_indices:
                results.append({
                    'class_name': self.df.iloc[idx]['class_name'],
                    'description': self.df.iloc[idx]['description'],
                    'similarity_score': similarities[idx]
                })
            
            return results
            
        except IndexError:
            return [{"error": f"Class '{predicted_class}' not found in database"}]
    
    def explain_prediction(self, predicted_class, confidence_score=None):
        similar_diseases = self.find_similar_diseases(predicted_class)
        explanation = f"🔎 Detected: {predicted_class.replace('___', ' - ')}\n"
        
        # Add description for the detected class
        try:
            detected_desc = self.df[self.df['class_name'] == predicted_class]['description'].iloc[0]
            explanation += f"📋 Description: {detected_desc}\n"
        except IndexError:
            pass
        
        if confidence_score:
            explanation += f"📊 Confidence: {confidence_score:.2%}\n"
        
        explanation += "\n🧠 Related diseases to monitor:\n"
        
        for disease in similar_diseases:
            if 'error' not in disease:
                name = disease['class_name'].replace('___', ' - ')
                score = disease['similarity_score']
                desc = disease['description']
                explanation += f"• {name} (similarity: {score:.2f})\n"
                explanation += f"  └─ {desc}\n\n"
        
        return explanation
    
    def save_model(self, filepath="tfidf_vectorizer.pkl"):
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'dataframe': self.df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath="tfidf_vectorizer.pkl"):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        engine = cls.__new__(cls)
        engine.vectorizer = model_data['vectorizer']
        engine.tfidf_matrix = model_data['tfidf_matrix']
        engine.df = model_data['dataframe']
        engine.csv_path = None
        
        return engine
