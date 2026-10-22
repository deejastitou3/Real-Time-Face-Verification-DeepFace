import cv2
import numpy as np

# Load eye cascade for geometric alignment
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def align_eyes(face_img):
    """Aligns the face based on eye positions to reduce pose variation."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(eyes) != 2:
        return face_img # Fallback if eyes aren't clearly detected
    
    # Sort eyes by x-coordinate to ensure left and right
    eyes = sorted(eyes, key=lambda eye: eye[0])
    left_eye, right_eye = eyes[0], eyes[1]
    
    # Calculate centers
    left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
    right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
    
    # Calculate rotation angle
    dy = right_center[1] - left_center[1]
    dx = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotate image to align eyes horizontally
    center = (face_img.shape[1]//2, face_img.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return aligned

def apply_clahe(img):
    """Applies Contrast Limited Adaptive Histogram Equalization in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def normalize_face(face_img):
    """Photometric normalization: mean=0, std=1."""
    face_float = face_img.astype(np.float32)
    mean = np.mean(face_float, axis=(0, 1))
    std = np.std(face_float, axis=(0, 1))
    return (face_float - mean) / (std + 1e-6)

def preprocess_face(face_img):
    """Complete preprocessing pipeline."""
    aligned = align_eyes(face_img)
    clahe_img = apply_clahe(aligned)
    normalized = normalize_face(clahe_img)
    return normalized.astype(np.uint8) # Convert back to uint8 for DeepFace