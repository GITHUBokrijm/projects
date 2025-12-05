import cv2
import os
import numpy as np
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Paths and constants
DATASET_DIR = "dataset"
MODEL_PATH = "trainer.yml"
LABELS_PATH = "labels.pickle"
ATTENDANCE_DIR = "attendance_logs"
NUM_SAMPLES_PER_PERSON = 20
CONFIDENCE_THRESHOLD = 80  # lower is stricter, range depends on model

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")

        # Window styling
        self.root.geometry("950x600")
        self.root.configure(bg="#1e1e2f")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", font=("Segoe UI", 11), background="#1e1e2f", foreground="#ffffff")
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#00d4ff")

        # Camera & recognition state
        self.cap = None
        self.running = False
        self.mode = "idle"  # "idle", "register", "attendance"
        self.current_name = None
        self.sample_count = 0
        self.recognizer = None
        self.labels = {}  # name -> id
        self.id_to_name = {}  # id -> name
        self.today_marked = set()
        self.current_frame = None

        # Status variable
        self.status_var = tk.StringVar(value="Status: Idle")

        # Load existing model (if any)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._load_model()

        # Build UI
        self._build_layout()

        # Start camera automatically
        self.start_camera()

    def _build_layout(self):
        # Left control panel
        left_frame = tk.Frame(self.root, bg="#25253a", bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        header = ttk.Label(left_frame, text="Face Attendance", style="Header.TLabel")
        header.pack(pady=(10, 20))

        # Name input
        name_label = ttk.Label(left_frame, text="Person Name:")
        name_label.pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(left_frame, textvariable=self.name_var, width=25)
        name_entry.pack(padx=5, pady=5)

        register_btn = ttk.Button(left_frame, text="Register New Face", command=self.start_registration)
        register_btn.pack(fill=tk.X, padx=5, pady=(10, 5))

        train_btn = ttk.Button(left_frame, text="Retrain Model", command=self.train_model)
        train_btn.pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(left_frame, orient="horizontal").pack(fill=tk.X, padx=5, pady=15)

        attendance_btn = ttk.Button(left_frame, text="Start Attendance", command=self.start_attendance)
        attendance_btn.pack(fill=tk.X, padx=5, pady=5)

        stop_btn = ttk.Button(left_frame, text="Stop Attendance", command=self.stop_attendance)
        stop_btn.pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(left_frame, orient="horizontal").pack(fill=tk.X, padx=5, pady=15)

        quit_btn = ttk.Button(left_frame, text="Quit", command=self.on_close)
        quit_btn.pack(fill=tk.X, padx=5, pady=(5, 10))

        # Status label

        # Right main area
        right_frame = tk.Frame(self.root, bg="#1e1e2f")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera preview
        self.video_label = tk.Label(right_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Attendance list
        list_frame = tk.Frame(right_frame, bg="#1e1e2f")
        list_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        list_label = ttk.Label(list_frame, text="Today's Attendance:")
        list_label.pack(anchor=tk.W)

        self.attendance_list = tk.Listbox(list_frame, height=5, bg="#25253a", fg="#ffffff",
                                          font=("Consolas", 10), bd=0, highlightthickness=0)
        self.attendance_list.pack(fill=tk.X, pady=(3, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------------- Camera handling ---------------

    def start_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
            self.cap = None
            return
        self.running = True
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Status: Failed to read from camera")
            self.root.after(30, self._update_frame)
            return

        self.current_frame = frame.copy()
        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        if self.mode == "register":
            self._handle_registration(gray, faces, display_frame)
        elif self.mode == "attendance":
            self._handle_attendance(gray, faces, display_frame)

        # Convert frame to Tkinter image
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self._update_frame)

    # --------------- Registration ---------------

    def start_registration(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input required", "Please enter a name before registering.")
            return

        self.current_name = name
        self.sample_count = 0
        self.mode = "register"
        self.status_var.set(f"Status: Registering '{name}'. Look at the camera...")

        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

    def _handle_registration(self, gray, faces, frame):
        if self.sample_count >= NUM_SAMPLES_PER_PERSON:
            self.mode = "idle"
            self.status_var.set(f"Status: Collected samples for {self.current_name}. Training model...")
            self.train_model()
            self.current_name = None
            return

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        # Take first detected face
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y + h, x:x + w]

        person_dir = os.path.join(DATASET_DIR, self.current_name)
        img_path = os.path.join(person_dir, f"{self.sample_count + 1}.jpg")
        cv2.imwrite(img_path, face_roi)
        self.sample_count += 1

        self.status_var.set(
            f"Status: Capturing {self.sample_count}/{NUM_SAMPLES_PER_PERSON} for {self.current_name}"
        )

    # --------------- Training model ---------------

    def train_model(self):
        faces = []
        labels = []

        label_map = {}
        current_id = 0

        for root_dir, dirs, files in os.walk(DATASET_DIR):
            for dirname in dirs:
                person_dir = os.path.join(root_dir, dirname)
                for filename in os.listdir(person_dir):
                    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    path = os.path.join(person_dir, filename)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    if dirname not in label_map:
                        label_map[dirname] = current_id
                        current_id += 1

                    faces.append(img)
                    labels.append(label_map[dirname])

        if len(faces) == 0:
            messagebox.showwarning("Training", "No face images found. Register at least one person first.")
            self.status_var.set("Status: Training failed (no data)")
            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(MODEL_PATH)

        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

        self.labels = label_map
        self.id_to_name = {v: k for k, v in self.labels.items()}

        self.status_var.set("Status: Training complete")
        messagebox.showinfo("Training", "Model trained successfully.")

    def _load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(MODEL_PATH)
                with open(LABELS_PATH, "rb") as f:
                    self.labels = pickle.load(f)
                self.id_to_name = {v: k for k, v in self.labels.items()}
                self.status_var.set("Status: Model loaded successfully")
            except Exception as e:
                self.status_var.set(f"Status: Failed to load model: {e}")
        else:
            self.status_var.set("Status: No trained model yet. Please register faces.")

    # --------------- Attendance ---------------

    def start_attendance(self):
        if self.recognizer is None:
            messagebox.showwarning("Attendance", "No model loaded. Train the model first.")
            return
        self.mode = "attendance"
        self.today_marked.clear()
        self.attendance_list.delete(0, tk.END)
        self.status_var.set("Status: Attendance mode. Looking for known faces...")

    def stop_attendance(self):
        self.mode = "idle"
        self.status_var.set("Status: Attendance stopped")

    def _handle_attendance(self, gray, faces, frame):
        if self.recognizer is None:
            cv2.putText(frame, "Model not trained", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (200, 200))

            label_id, confidence = self.recognizer.predict(roi_resized)

            if confidence < CONFIDENCE_THRESHOLD:
                name = self.id_to_name.get(label_id, "Unknown")
            else:
                name = "Unknown"

            if name != "Unknown":
                color = (0, 255, 0)
                self._mark_attendance(name)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _mark_attendance(self, name):
        if name in self.today_marked:
            return

        self.today_marked.add(name)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        filename = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="", encoding="utf-8") as f:
            if not file_exists:
                f.write("Name,Date,Time\n")
            f.write(f"{name},{date_str},{time_str}\n")

        self.attendance_list.insert(tk.END, f"{name} - {time_str}")
        self.status_var.set(f"Status: Marked present - {name} at {time_str}")

    # --------------- Cleanup ---------------

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()