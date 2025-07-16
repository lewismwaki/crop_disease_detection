import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from similarity_engine import DiseaseSimilarityEngine
import json
import os
import sys

class DiseaseDetector:
    def __init__(self):
        self.model = None
        self.similarity_engine = None
        self.classes = None
        self._load_models()
    
    def _load_models(self):
        try:
            self.model = tf.keras.models.load_model('crop_disease_model.keras')
            self.similarity_engine = DiseaseSimilarityEngine.load_model('tfidf_vectorizer.pkl')
            
            with open('deployment_info.json', 'r') as f:
                self.classes = json.load(f)['classes']
                
        except FileNotFoundError as e:
            if 'tfidf_vectorizer.pkl' in str(e):
                self.similarity_engine = DiseaseSimilarityEngine('disease_descriptions.csv')
                self.similarity_engine.load_and_vectorize()
                self.similarity_engine.save_model('tfidf_vectorizer.pkl')
            else:
                print(f"Missing file: {e}")
                sys.exit(1)
    
    def predict(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        # CNN Prediction
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.classes[predicted_class_idx]
        
        # NLP Explanation
        explanation = self.similarity_engine.explain_prediction(predicted_class, confidence)
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'explanation': explanation
        }

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"')
    
    detector = DiseaseDetector()
    result = detector.predict(image_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(result['explanation'])

if __name__ == "__main__":
    main()
