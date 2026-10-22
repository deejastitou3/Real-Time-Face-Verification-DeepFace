import os
import pickle
import numpy as np
from deepface import DeepFace

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def extract_and_save_embeddings(dataset_path="dataset"):
    """Extracts FaceNet embeddings for all images in the dataset and saves to a pickle file."""
    database = {}
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found. Please create it and add subfolders for each person.")
        return

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            database[person] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    # DeepFace represent returns a list of dicts, we take the first one
                    result = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)
                    emb = np.array(result[0]['embedding'])
                    database[person].append(emb)
                    print(f"Processed: {person}/{img_name}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
    # Save the database
    with open("models/embeddings.pkl", "wb") as f:
        pickle.dump(database, f)
    print("✅ Embeddings successfully saved to models/embeddings.pkl")

if __name__ == "__main__":
    extract_and_save_embeddings()