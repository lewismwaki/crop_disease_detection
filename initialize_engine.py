from similarity_engine import DiseaseSimilarityEngine
import os

def initialize_similarity_engine():
    if not os.path.exists('disease_descriptions.csv'):
        print("Error: disease_descriptions.csv not found!")
        return False
    
    engine = DiseaseSimilarityEngine('disease_descriptions.csv')
    engine.load_and_vectorize()
    engine.save_model('tfidf_vectorizer.pkl')
    
    print(f"Similarity engine initialized with {len(engine.df)} diseases")
    return True

if __name__ == "__main__":
    initialize_similarity_engine()
