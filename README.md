# ️ Real-Time Face Verification System (FaceGuard)

A complete, real-time biometric verification pipeline built with Python, OpenCV, and DeepFace. This system implements a robust chain of face detection, advanced preprocessing, neural network embedding extraction, and statistical decision-making, all wrapped in a responsive, multithreaded Tkinter GUI.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![DeepFace](https://img.shields.io/badge/DeepFace-FaceNet-orange)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-purple)

##  About the Project
Unlike standard identification ("Who is this?"), this system focuses on **verification** ("Is this the claimed person?"). It allows users to enroll new faces and verify them in real-time via webcam or image upload, without requiring expensive model retraining. The system achieves >95% verification accuracy on custom datasets by utilizing advanced preprocessing and optimal Euclidean distance thresholding.

##  Key Features
- **Real-Time Processing:** Multithreaded architecture ensures the Tkinter GUI remains fluid while running heavy neural network inference.
- **Advanced Preprocessing:** Implements geometric eye alignment, CLAHE (Contrast Limited Adaptive Histogram Equalization), and photometric normalization to maximize model robustness against lighting and pose variations.
- **Deep Learning Embeddings:** Utilizes the pre-trained FaceNet model (via DeepFace) to extract highly discriminative 512-dimensional facial embeddings.
- **Statistical Decision Making:** Verifies identity using Euclidean distance comparison against a stored database, with an optimized threshold of 9.5.
- **Dual Input Modes:** Supports both live webcam verification and static image upload from disk.

##  System Architecture
The pipeline is divided into four modular stages:
1. **Detection:** Localizes the face in the frame using the Viola-Jones algorithm (Haar Cascade).
2. **Preprocessing:** Aligns eyes geometrically, enhances local contrast via CLAHE in LAB color space, and normalizes photometry (mean=0, std=1).
3. **Embedding Extraction:** Passes the preprocessed face through FaceNet to generate a 512-dim vector.
4. **Decision:** Computes the minimum Euclidean distance between the query embedding and the enrolled database. If `distance < threshold (9.5)`, access is granted.

##  Tech Stack
- **Languages:** Python 3.12
- **Computer Vision:** OpenCV (`cv2`)
- **Deep Learning:** DeepFace (`Facenet` model)
- **Data Handling:** NumPy, Pickle
- **GUI & Concurrency:** Tkinter, Threading

##  Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deejastitou3/Real-Time-Face-Verification-DeepFace.git
   cd Real-Time-Face-Verification-DeepFace

## Author
Khadija Stitou
Engineering Student in Intelligent Embedded Systems & Telecommunications
