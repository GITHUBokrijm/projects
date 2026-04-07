import cv2
import os
import re
import numpy as np
import pickle
import time
import sqlite3
import threading
from datetime import datetime

import customtkinter as ctk
from PIL import Image, ImageTk

# Optional TTS — gracefully degrade if unavailable
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ─── Paths and constants ───────────────────────────────────────────
DATASET_DIR = "dataset"
MODEL_PATH = "trainer.yml"
LABELS_PATH = "labels.pickle"
ATTENDANCE_DIR = "attendance_logs"
DB_PATH = "attendance.db"
NUM_SAMPLES_PER_PERSON = 20
CONFIDENCE_THRESHOLD = 80  # lower = stricter for LBPH
BLINK_FRAME_WINDOW = 90     # frames to wait for a blink (~3s at 30fps)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ─── Appearance ────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Glassmorphism palette
GLASS_BG = "#0d0d1a"           # deep dark background
GLASS_PANEL = "#1a1a2e"        # panel base
GLASS_PANEL_BORDER = "#ffffff0d"  # very subtle white border (hex+alpha won't work in tk, we fake it)
GLASS_ACCENT = "#00d4ff"       # cyan accent
GLASS_ACCENT_HOVER = "#00b8d9"
GLASS_GREEN = "#00e676"
GLASS_RED = "#ff1744"
GLASS_YELLOW = "#ffea00"
GLASS_TEXT = "#e0e0e0"
GLASS_TEXT_DIM = "#9e9e9e"
GLASS_SURFACE = "#25253d"      # elevated surface


# ═══════════════════════════════════════════════════════════════════
#  DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════
class DatabaseManager:
    """Lightweight SQLite wrapper for attendance records."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0
                )
            """)
            conn.commit()

    def insert(self, name, date_str, time_str, confidence=0.0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO attendance (name, date, time, confidence) VALUES (?, ?, ?, ?)",
                (name, date_str, time_str, confidence),
            )
            conn.commit()

    def get_by_date(self, date_str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name, date, time, confidence FROM attendance WHERE date = ? ORDER BY time",
                (date_str,),
            )
            return cursor.fetchall()

    def get_all_dates(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT date FROM attendance ORDER BY date DESC"
            )
            return [row[0] for row in cursor.fetchall()]


# ═══════════════════════════════════════════════════════════════════
#  TTS MANAGER
# ═══════════════════════════════════════════════════════════════════
class TTSManager:
    """Non-blocking text-to-speech via a daemon thread."""

    def __init__(self):
        self.enabled = TTS_AVAILABLE
        self.muted = False
        self._lock = threading.Lock()
        self._engine = None
        if self.enabled:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
                self._engine.setProperty("volume", 0.9)
            except Exception:
                self.enabled = False

    def speak(self, text):
        if not self.enabled or self.muted or not self._engine:
            return
        t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        t.start()

    def _speak_blocking(self, text):
        with self._lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.setProperty("volume", 0.9)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted


# ═══════════════════════════════════════════════════════════════════
#  BLINK  DETECTOR (Haar Cascade based — no dlib needed)
# ═══════════════════════════════════════════════════════════════════
class BlinkDetector:
    """
    Tracks eye presence across frames using haarcascade_eye.xml.
    A blink = eyes detected → eyes NOT detected → eyes detected again.
    """

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        # Per-person tracking: name → {state, frames_since_recognition, blink_count, last_eye_state}
        self.tracking = {}

    def reset(self, name=None):
        if name:
            self.tracking.pop(name, None)
        else:
            self.tracking.clear()

    def update(self, name, face_roi_gray):
        """
        Call each frame with the gray face ROI for a recognized person.
        Returns: (verified: bool, status_text: str)
        """
        if name not in self.tracking:
            self.tracking[name] = {
                "state": "waiting",  # waiting → verified
                "frames": 0,
                "blink_count": 0,
                "last_eye_state": None,  # True = eyes open, False = eyes closed
            }

        t = self.tracking[name]
        t["frames"] += 1

        # Detect eyes in the face ROI
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        eyes_open = len(eyes) >= 2  # both eyes detected

        if t["state"] == "verified":
            return True, "Verified"

        # Track state transitions
        if t["last_eye_state"] is not None:
            if t["last_eye_state"] is True and not eyes_open:
                pass  # eyes just closed — wait
            elif t["last_eye_state"] is False and eyes_open:
                # eyes reopened = blink completed
                t["blink_count"] += 1

        t["last_eye_state"] = eyes_open

        if t["blink_count"] >= 1:
            t["state"] = "verified"
            return True, "Verified"

        if t["frames"] > BLINK_FRAME_WINDOW:
            # Timeout — reset and try again
            t["frames"] = 0
            t["blink_count"] = 0
            t["last_eye_state"] = None

        return False, "Blink to verify..."


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════
class FaceAttendanceApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("1050x700")
        self.root.configure(fg_color=GLASS_BG)
        self.root.minsize(900, 600)

        # ── State ──
        self.cap = None
        self.running = False
        self.mode = "idle"
        self.current_name = None
        self.sample_count = 0
        self.recognizer = None
        self.labels = {}
        self.id_to_name = {}
        self.today_marked = set()
        self.current_frame = None
        self.last_capture_time = 0
        self.last_confidence = 0.0

        # ── Toggles ──
        self.liveness_enabled = True

        # ── Subsystems ──
        self.db = DatabaseManager()
        self.tts = TTSManager()
        self.blink_detector = BlinkDetector()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.status_var = ctk.StringVar(value="Status: Idle")

        self._load_model()
        self._build_layout()
        self.start_camera()

    # ──────────────────── UI LAYOUT ────────────────────────────────
    def _build_layout(self):
        # ── Left Sidebar (Glassmorphism Panel) ──
        self.sidebar = ctk.CTkFrame(
            self.root,
            fg_color=GLASS_PANEL,
            corner_radius=20,
            border_width=1,
            border_color="#ffffff12",
            width=260,
        )
        self.sidebar.pack(side="left", fill="y", padx=(15, 5), pady=15)
        self.sidebar.pack_propagate(False)

        # Inner glow effect — a thin highlight frame at the top
        glow_bar = ctk.CTkFrame(self.sidebar, fg_color=GLASS_ACCENT, height=3, corner_radius=2)
        glow_bar.pack(fill="x", padx=20, pady=(15, 0))

        # Header
        ctk.CTkLabel(
            self.sidebar,
            text="Face Attendance",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=GLASS_ACCENT,
        ).pack(pady=(12, 5))

        ctk.CTkLabel(
            self.sidebar,
            text="Intelligent Recognition System",
            font=ctk.CTkFont(size=11),
            text_color=GLASS_TEXT_DIM,
        ).pack(pady=(0, 15))

        # ── Name Input ──
        input_frame = ctk.CTkFrame(self.sidebar, fg_color=GLASS_SURFACE, corner_radius=12)
        input_frame.pack(fill="x", padx=15, pady=(5, 10))

        ctk.CTkLabel(
            input_frame, text="Person Name", font=ctk.CTkFont(size=11),
            text_color=GLASS_TEXT_DIM,
        ).pack(anchor="w", padx=12, pady=(8, 0))

        self.name_var = ctk.StringVar()
        self.name_entry = ctk.CTkEntry(
            input_frame, textvariable=self.name_var, height=36,
            font=ctk.CTkFont(size=13), corner_radius=8,
            fg_color="#16162b", border_color=GLASS_ACCENT, border_width=1,
            placeholder_text="Enter name...",
        )
        self.name_entry.pack(fill="x", padx=12, pady=(4, 10))

        # ── Action Buttons ──
        self._make_button(self.sidebar, "📷  Register Face", self.start_registration, GLASS_ACCENT)
        self._make_button(self.sidebar, "🔄  Retrain Model", self.train_model, "#7c4dff")

        # Separator
        ctk.CTkFrame(self.sidebar, fg_color="#ffffff08", height=1).pack(fill="x", padx=20, pady=12)

        self._make_button(self.sidebar, "✅  Start Attendance", self.start_attendance, GLASS_GREEN)
        self._make_button(self.sidebar, "⏹  Stop Attendance", self.stop_attendance, GLASS_RED)

        ctk.CTkFrame(self.sidebar, fg_color="#ffffff08", height=1).pack(fill="x", padx=20, pady=12)

        self._make_button(self.sidebar, "📊  View Records", self.open_records_window, "#ff9100")

        # ── Toggle Row ──
        toggle_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        toggle_frame.pack(fill="x", padx=15, pady=(10, 0))

        # Mute toggle
        self.mute_var = ctk.BooleanVar(value=False)
        mute_switch = ctk.CTkSwitch(
            toggle_frame, text="🔇 Mute TTS",
            variable=self.mute_var, command=self._toggle_mute,
            font=ctk.CTkFont(size=11), text_color=GLASS_TEXT_DIM,
            progress_color=GLASS_ACCENT,
        )
        mute_switch.pack(anchor="w", pady=2)

        # Liveness toggle
        self.liveness_var = ctk.BooleanVar(value=True)
        liveness_switch = ctk.CTkSwitch(
            toggle_frame, text="👁 Liveness Check",
            variable=self.liveness_var, command=self._toggle_liveness,
            font=ctk.CTkFont(size=11), text_color=GLASS_TEXT_DIM,
            progress_color=GLASS_GREEN,
        )
        liveness_switch.pack(anchor="w", pady=2)

        # ── Status ──
        self.status_label = ctk.CTkLabel(
            self.sidebar, textvariable=self.status_var,
            font=ctk.CTkFont(size=11), text_color=GLASS_ACCENT,
            wraplength=220, justify="left",
        )
        self.status_label.pack(fill="x", padx=15, pady=(15, 5), anchor="s")

        # Quit
        self._make_button(self.sidebar, "✕  Quit", self.on_close, "#424242", pady=(5, 15))

        # ── Right Main Area (Glassmorphism Panel) ──
        self.main_frame = ctk.CTkFrame(
            self.root,
            fg_color=GLASS_PANEL,
            corner_radius=20,
            border_width=1,
            border_color="#ffffff12",
        )
        self.main_frame.pack(side="right", fill="both", expand=True, padx=(5, 15), pady=15)

        # Camera preview
        self.video_label = ctk.CTkLabel(
            self.main_frame, text="", fg_color="#000000", corner_radius=14,
        )
        self.video_label.pack(fill="both", expand=True, padx=12, pady=(12, 6))

        # Attendance list (glass surface)
        list_container = ctk.CTkFrame(
            self.main_frame, fg_color=GLASS_SURFACE, corner_radius=12,
            border_width=1, border_color="#ffffff08",
        )
        list_container.pack(fill="x", padx=12, pady=(6, 12))

        ctk.CTkLabel(
            list_container, text="📋  Today's Attendance",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=GLASS_TEXT,
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.attendance_textbox = ctk.CTkTextbox(
            list_container, height=100, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#16162b", corner_radius=8, text_color=GLASS_TEXT,
            border_width=0, activate_scrollbars=True,
        )
        self.attendance_textbox.pack(fill="x", padx=12, pady=(0, 10))
        self.attendance_textbox.configure(state="disabled")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _make_button(self, parent, text, command, color, pady=(0, 6)):
        btn = ctk.CTkButton(
            parent, text=text, command=command,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=color, hover_color=self._darken(color),
            corner_radius=10, height=38,
        )
        btn.pack(fill="x", padx=15, pady=pady)
        return btn

    @staticmethod
    def _darken(hex_color, factor=0.8):
        """Darken a hex color for hover states."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) < 6:
            return "#333333"
        try:
            r = int(int(hex_color[0:2], 16) * factor)
            g = int(int(hex_color[2:4], 16) * factor)
            b = int(int(hex_color[4:6], 16) * factor)
            return f"#{r:02x}{g:02x}{b:02x}"
        except ValueError:
            return "#333333"

    def _toggle_mute(self):
        self.tts.muted = self.mute_var.get()

    def _toggle_liveness(self):
        self.liveness_enabled = self.liveness_var.get()
        self.blink_detector.reset()

    # ──────────────────── CAMERA ──────────────────────────────────
    def start_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Status: Cannot open camera!")
            self.cap = None
            return
        self.running = True
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Status: Failed to read from camera")
            self.root.after(30, self._update_frame)
            return

        # Normalize to 640x480 for consistent performance
        frame = cv2.resize(frame, (640, 480))
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

        # Convert to Tkinter-compatible image
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk  # prevent GC
        self.video_label.configure(image=imgtk)

        self.root.after(30, self._update_frame)

    # ──────────────────── REGISTRATION ────────────────────────────
    def start_registration(self):
        if self.mode == "register":
            return

        raw_name = self.name_var.get().strip()
        name = re.sub(r'[^a-zA-Z0-9 ]', '', raw_name)

        if not name:
            self.status_var.set("Status: Enter a valid name first!")
            return

        self.current_name = name
        self.sample_count = 0
        self.mode = "register"
        self.status_var.set(f"Status: Registering '{name}'. Look at the camera...")
        self.tts.speak(f"Starting registration for {name}")

        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

    def _handle_registration(self, gray, faces, frame):
        if self.sample_count >= NUM_SAMPLES_PER_PERSON:
            self.mode = "idle"
            name = self.current_name
            self.status_var.set(f"Status: Collected samples for {name}. Training model...")
            self.current_name = None
            self.root.after(100, self.train_model)
            return

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        (x, y, w, h) = faces[0]

        # Capture cooldown — force head movement for dataset variation
        current_time = time.time()
        if (current_time - self.last_capture_time) < 0.3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Move slightly...", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return

        self.last_capture_time = current_time
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))

        person_dir = os.path.join(DATASET_DIR, self.current_name)
        timestamp = datetime.now().strftime("%H%M%S%f")
        img_path = os.path.join(person_dir, f"sample_{self.sample_count + 1}_{timestamp}.jpg")
        cv2.imwrite(img_path, face_roi)
        self.sample_count += 1

        # Progress overlay
        progress = self.sample_count / NUM_SAMPLES_PER_PERSON
        bar_w = int(200 * progress)
        cv2.rectangle(frame, (10, 460), (10 + bar_w, 475), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 460), (210, 475), (255, 255, 255), 1)

        self.status_var.set(
            f"Status: Capturing {self.sample_count}/{NUM_SAMPLES_PER_PERSON} for {self.current_name}"
        )

    # ──────────────────── TRAINING ────────────────────────────────
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

                    img = cv2.resize(img, (200, 200))

                    if dirname not in label_map:
                        label_map[dirname] = current_id
                        current_id += 1

                    faces.append(img)
                    labels.append(label_map[dirname])

        if len(faces) == 0:
            self.status_var.set("Status: Training failed — no data")
            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(MODEL_PATH)

        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

        self.labels = label_map
        self.id_to_name = {v: k for k, v in self.labels.items()}

        self.status_var.set(f"Status: Training complete — {len(label_map)} people registered")
        self.tts.speak("Model training complete")

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
            self.status_var.set("Status: No trained model. Please register faces.")

    # ──────────────────── ATTENDANCE ──────────────────────────────
    def start_attendance(self):
        if self.recognizer is None:
            self.status_var.set("Status: No model loaded. Train first!")
            return
        self.mode = "attendance"
        self.blink_detector.reset()
        self.status_var.set("Status: Attendance mode — looking for faces...")
        self.tts.speak("Attendance mode started")

    def stop_attendance(self):
        self.mode = "idle"
        self.blink_detector.reset()
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
            self.last_confidence = confidence

            if confidence < CONFIDENCE_THRESHOLD:
                name = self.id_to_name.get(label_id, "Unknown")
            else:
                name = "Unknown"

            if name != "Unknown":
                # ── Liveness gate ──
                if self.liveness_enabled:
                    face_roi_for_blink = gray[y:y + h, x:x + w]
                    verified, blink_status = self.blink_detector.update(name, face_roi_for_blink)

                    if not verified:
                        # Yellow box — waiting for blink
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(frame, f"{name} — {blink_status}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        continue
                    else:
                        # Green flash — verified
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(frame, f"{name} - Verified!", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self._mark_attendance(name, confidence)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Unknown ({int(confidence)})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _mark_attendance(self, name, confidence=0.0):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        daily_key = f"{name}_{date_str}"
        if daily_key in self.today_marked:
            return

        self.today_marked.add(daily_key)

        # ── Write to CSV ──
        filename = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
        file_exists = os.path.exists(filename)

        try:
            with open(filename, "a", newline="", encoding="utf-8") as f:
                if not file_exists:
                    f.write("Name,Date,Time\n")
                f.write(f"{name},{date_str},{time_str}\n")
        except PermissionError:
            self.status_var.set("Status: CSV locked! (Close Excel?)")
            self.today_marked.remove(daily_key)
            return

        # ── Write to SQLite ──
        try:
            self.db.insert(name, date_str, time_str, confidence)
        except Exception:
            pass  # CSV already written — DB is secondary

        # ── Update UI ──
        self.attendance_textbox.configure(state="normal")
        self.attendance_textbox.insert("end", f"  ✓ {name}  —  {time_str}\n")
        self.attendance_textbox.configure(state="disabled")
        self.attendance_textbox.see("end")

        self.status_var.set(f"Status: Marked present — {name} at {time_str}")
        self.tts.speak(f"Welcome, {name}")

    # ──────────────────── VIEW RECORDS ────────────────────────────
    def open_records_window(self):
        win = ctk.CTkToplevel(self.root)
        win.title("Attendance Records")
        win.geometry("600x500")
        win.configure(fg_color=GLASS_BG)
        win.transient(self.root)
        win.grab_set()

        # Glass panel
        panel = ctk.CTkFrame(win, fg_color=GLASS_PANEL, corner_radius=16,
                             border_width=1, border_color="#ffffff12")
        panel.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(
            panel, text="📊  Attendance Records",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=GLASS_ACCENT,
        ).pack(pady=(15, 5))

        # Date selector
        date_frame = ctk.CTkFrame(panel, fg_color="transparent")
        date_frame.pack(fill="x", padx=20, pady=10)

        all_dates = self.db.get_all_dates()
        if not all_dates:
            all_dates = [datetime.now().strftime("%Y-%m-%d")]

        date_var = ctk.StringVar(value=all_dates[0])
        date_menu = ctk.CTkOptionMenu(
            date_frame, variable=date_var, values=all_dates,
            fg_color=GLASS_SURFACE, button_color=GLASS_ACCENT,
            button_hover_color=GLASS_ACCENT_HOVER, corner_radius=8,
            font=ctk.CTkFont(size=12),
        )
        date_menu.pack(side="left", padx=(0, 10))

        # Records display
        records_text = ctk.CTkTextbox(
            panel, font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#16162b", corner_radius=10, text_color=GLASS_TEXT,
        )
        records_text.pack(fill="both", expand=True, padx=20, pady=(5, 10))

        def load_records(*_):
            records_text.configure(state="normal")
            records_text.delete("1.0", "end")

            rows = self.db.get_by_date(date_var.get())
            if not rows:
                records_text.insert("end", "  No records for this date.\n")
            else:
                records_text.insert("end", f"  {'Name':<20} {'Time':<12} {'Confidence'}\n")
                records_text.insert("end", "  " + "─" * 48 + "\n")
                for name, date, time_val, conf in rows:
                    records_text.insert("end", f"  {name:<20} {time_val:<12} {conf:.1f}\n")
                records_text.insert("end", f"\n  Total: {len(rows)} records\n")

            records_text.configure(state="disabled")

        date_var.trace_add("write", load_records)

        load_btn = ctk.CTkButton(
            date_frame, text="Load", command=load_records,
            fg_color=GLASS_ACCENT, hover_color=GLASS_ACCENT_HOVER,
            corner_radius=8, width=80, font=ctk.CTkFont(size=12, weight="bold"),
        )
        load_btn.pack(side="left")

        # Export button
        def export_csv():
            rows = self.db.get_by_date(date_var.get())
            if not rows:
                return
            export_path = os.path.join(ATTENDANCE_DIR, f"export_{date_var.get()}.csv")
            with open(export_path, "w", newline="", encoding="utf-8") as f:
                f.write("Name,Date,Time,Confidence\n")
                for name, date, time_val, conf in rows:
                    f.write(f"{name},{date},{time_val},{conf}\n")
            self.status_var.set(f"Status: Exported to {export_path}")

        ctk.CTkButton(
            panel, text="📥  Export CSV", command=export_csv,
            fg_color="#ff9100", hover_color="#e68200", corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(0, 15))

        # Load initial data
        load_records()

    # ──────────────────── CLEANUP ─────────────────────────────────
    def on_close(self):
        self.stop_camera()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app_root = ctk.CTk()
    app = FaceAttendanceApp(app_root)
    app_root.mainloop()