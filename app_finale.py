import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import pickle
import os
from PIL import Image, ImageTk # Requires 'pillow' library
from detection import detect_face
from decision import verify_face

class FaceGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceGuard - Système de Vérification d'Identité")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.cap = None
        self.webcam_running = False
        self.current_frame = None
        self.database = {}
        
        self.load_database()
        self.setup_ui()
        
    def load_database(self):
        """Loads the pre-computed embeddings database."""
        db_path = "models/embeddings.pkl"
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.database = pickle.load(f)
            print(f"✅ Loaded database with {len(self.database)} persons.")
        else:
            messagebox.showwarning("Attention", "Base de données introuvable. Veuillez d'abord exécuter extract_embeddings.py")
            
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="FaceGuard Verification System", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=10)
        
        # Canvas for webcam
        self.cam_canvas = tk.Canvas(self.root, width=640, height=480, bg="black")
        self.cam_canvas.pack(pady=5)
        
        # Control Frame
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(pady=10)
        
        self.btn_start = tk.Button(control_frame, text="START Webcam", command=self.start_webcam, width=15, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.btn_start.grid(row=0, column=0, padx=10)
        
        self.btn_verify = tk.Button(control_frame, text="VERIFY", command=self.verify_webcam, width=15, bg="#2196F3", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_verify.grid(row=0, column=1, padx=10)
        
        self.btn_stop = tk.Button(control_frame, text="STOP", command=self.stop_webcam, width=15, bg="#f44336", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=2, padx=10)
        
        # Result Frame
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.result_frame.pack(pady=10)
        
        self.lbl_identity = tk.Label(self.result_frame, text="Identity: --", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        self.lbl_identity.pack()
        
        self.lbl_distance = tk.Label(self.result_frame, text="Distance: --", font=("Arial", 12), bg="#f0f0f0", fg="#666")
        self.lbl_distance.pack()
        
        self.lbl_confidence = tk.Label(self.result_frame, text="Confidence: --%", font=("Arial", 12), bg="#f0f0f0", fg="#666")
        self.lbl_confidence.pack()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return
            
        self.webcam_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_verify.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        
        # Start feed loop in a separate thread to keep GUI responsive
        threading.Thread(target=self._feed_loop, daemon=True).start()

    def _feed_loop(self):
        """Continuously captures frames and updates the canvas."""
        while self.webcam_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            
            # Draw bounding box if face is detected
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (160, 100, 220), 2)
                
            self._show_on_canvas(frame)
            
        self.stop_webcam()

    def _show_on_canvas(self, frame):
        """Converts OpenCV frame to Tkinter PhotoImage and updates canvas."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 480))
        
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)
        
        self.cam_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.cam_canvas.image = photo # Keep a reference to prevent garbage collection

    def verify_webcam(self):
        """Triggers verification without blocking the UI."""
        if self.current_frame is None:
            return
            
        face = detect_face(self.current_frame)
        if face is not None:
            # Run heavy verification in a background thread
            threading.Thread(target=self._run_verify, args=(face, "Webcam"), daemon=True).start()
        else:
            self.root.after(0, lambda: self._set_result("No Face Detected", 0.0, 0.0, "Webcam"))

    def _run_verify(self, face, source):
        """Executes the verification and schedules UI update."""
        name, dist, conf = verify_face(face, self.database, threshold=9.5)
        # Schedule UI update in the main thread safely
        self.root.after(0, lambda: self._set_result(name, dist, conf, source))

    def _set_result(self, name, dist, conf, source):
        """Safely updates the UI labels from the main thread."""
        self.lbl_identity.config(text=f"Identity: {name}")
        self.lbl_distance.config(text=f"Distance: {dist:.2f}")
        self.lbl_confidence.config(text=f"Confidence: {conf:.1f}%")
        
        # Color coding based on result
        if name != "Inconnu" and name != "No Face Detected":
            self.lbl_identity.config(fg="green")
        else:
            self.lbl_identity.config(fg="red")

    def stop_webcam(self):
        self.webcam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_verify.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        self.cam_canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceGuardApp(root)
    root.mainloop()