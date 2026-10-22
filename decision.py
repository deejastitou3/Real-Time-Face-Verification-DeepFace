import numpy as np
from deepface import DeepFace

def extract_embedding(face_bgr):
    """Extracts a 512-dim embedding from a preprocessed face image (BGR format)."""
    try:
        result = DeepFace.represent(img_path=face_bgr, model_name='Facenet', enforce_detection=False)
        return np.array(result[0]['embedding'])
    except Exception as e:
        print(f"Embedding extraction error: {e}")
        return None

def verify_face(face_bgr, database, threshold=9.5):
    """Verifies the face against the database using Euclidean distance."""
    emb = extract_embedding(face_bgr)
    if emb is None:
        return "Inconnu", 999.0, 0.0
    
    best_dist = 999.0
    best_person = "Inconnu"
    
    for person, embeddings in database.items():
        for e in embeddings:
            dist = np.linalg.norm(emb - e)
            if dist < best_dist:
                best_dist = dist
                best_person = person
                
    if best_dist < threshold:
        # Confidence calculation: max(0, (1 - dist/threshold) * 100)
        confidence = max(0.0, (1 - (best_dist / threshold)) * 100)
        return best_person, best_dist, confidence
    else:
        return "Inconnu", best_dist, 0.0