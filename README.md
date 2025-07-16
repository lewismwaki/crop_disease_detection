# Crop Disease Detection with NLP Explanation

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Initialize similarity engine** (run once):
```bash
python initialize_engine.py
```

2. **Make prediction with image**:
```bash
python main.py image.png
```
# Drop image into the root directory and change `image.png` to your image file name.

Or run interactively:
```bash
python main.py
# Enter image path when prompted
```

## Output

Get CNN prediction + confidence + similar diseases explanation:

```
🔎 Detected: Tomato - Late blight
📊 Confidence: 94.32%

🧠 Related diseases to monitor:
• Potato - Early blight (similarity: 0.45)
  └─ Fungal blight marked by concentric rings on leaves

• Tomato - Early blight (similarity: 0.41)  
  └─ Fungal disease with brown concentric rings on leaves
```

## Files

- `main.py` - Main entry point (upload image → get results)
- `similarity_engine.py` - Core NLP similarity logic, can be tested on cnn.ipynb
- `initialize_engine.py` - One-time setup for similarity engine
- `crop_disease_model.keras` - Trained CNN model, trained on cnn.ipynb
- `disease_descriptions.csv` - Disease database

## similarity engine explanations

The DiseaseSimilarityEngine class provides the following capabilities:

- **load_and_vectorize()** - Loads disease descriptions from CSV and creates TF-IDF vectors for similarity matching
- **find_similar_diseases()** - Uses cosine similarity to find diseases most similar to a predicted class
- **explain_prediction()** - Generates human-readable explanations combining prediction confidence with similar diseases
- **save_model()** - Serializes the trained vectorizer and data for deployment
- **load_model()** - Loads pre-trained vectorizer model from disk
