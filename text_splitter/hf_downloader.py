from sentence_transformers import SentenceTransformer
import os

# Define your custom model directory (e.g., D drive)
model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"  # Use forward slashes
os.makedirs(model_dir, exist_ok=True)

# Download and save the model locally
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save(model_dir)
print(f"Model saved to {model_dir}")