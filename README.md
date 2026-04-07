# 🎭 Face Attendance System

A **production-grade face recognition attendance system** built with Python, featuring a glassmorphism UI, real-time camera processing, eye-blink anti-spoofing, text-to-speech feedback, and SQLite database storage.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📷 Face Registration** | Capture 20 face samples per person with automated cooldown for dataset variation |
| **✅ Real-Time Attendance** | LBPH face recognition with confidence scoring and instant CSV + SQLite logging |
| **👁 Blink Liveness Detection** | Anti-spoofing via eye-blink verification — blocks photo/video attacks |
| **🔊 Text-to-Speech** | Audible "Welcome, {name}" greetings via `pyttsx3` on background threads |
| **📊 Records Viewer** | Browse attendance by date, view confidence scores, and export to CSV |
| **🎨 Glassmorphism UI** | Modern dark-mode interface with frosted-glass panels, glowing accents, and rounded corners |
| **🗄️ Dual Storage** | Simultaneous CSV + SQLite persistence for maximum compatibility |

---

## 🖥️ Screenshots

> Run the app to see the full glassmorphism UI with live camera feed, liveness detection overlays, and the records viewer.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- A working **webcam**
- **Windows** (recommended for TTS; works on Linux/macOS with minor TTS adjustments)

### Installation

```bash
# Clone the repository
git clone https://github.com/lukeewarmcoder/projects.git
cd projects

# Install dependencies
pip install opencv-contrib-python numpy pillow customtkinter pyttsx3
```

### Run

```bash
python face_attendance.py
```

---

## 📖 How to Use

### 1. Register a Face
1. Type a person's name in the **"Person Name"** field
2. Click **"📷 Register Face"**
3. Look at the camera and move your head slightly between captures
4. 20 samples are collected automatically (~6 seconds)
5. The model auto-trains after registration

### 2. Take Attendance
1. Click **"✅ Start Attendance"**
2. Recognized faces appear with green bounding boxes
3. If **Liveness Check** is ON, you'll see a yellow **"Blink to verify..."** prompt — simply blink
4. Once verified, attendance is logged and the system says "Welcome, {name}"
5. Click **"⏹ Stop Attendance"** when done

### 3. View Records
1. Click **"📊 View Records"**
2. Select a date from the dropdown
3. Click **"Load"** to view attendance for that day
4. Click **"📥 Export CSV"** to save a copy

---

## 🏗️ Architecture

```
face_attendance.py
├── DatabaseManager      — SQLite wrapper (attendance.db)
├── TTSManager           — pyttsx3 on daemon threads
├── BlinkDetector        — Haar cascade eye-blink state machine
└── FaceAttendanceApp    — Main app (camera, UI, recognition)
```

### File Structure

```
projects/
├── face_attendance.py       # Main application
├── attendance.db            # SQLite database (auto-created)
├── trainer.yml              # Trained LBPH model (auto-created)
├── labels.pickle            # Label mapping (auto-created)
├── dataset/                 # Face samples per person
│   ├── John/
│   └── Alice/
├── attendance_logs/         # Daily CSV files
│   └── attendance_2026-04-07.csv
└── README.md
```

---

## 🔧 Configuration

Constants at the top of `face_attendance.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `NUM_SAMPLES_PER_PERSON` | `20` | Face samples captured per registration |
| `CONFIDENCE_THRESHOLD` | `80` | LBPH confidence cutoff (lower = stricter) |
| `BLINK_FRAME_WINDOW` | `90` | Frames to wait for a blink (~3s at 30fps) |

---

## 🛡️ Security Features

- **Eye-blink liveness detection** — Prevents photo/video spoofing attacks
- **Input sanitization** — Regex strips special characters from names to prevent path traversal
- **Excel lock handling** — Gracefully handles `PermissionError` when CSV is open in another program
- **Double-registration blocking** — Prevents overlapping registration sessions

---

## 🧰 Tech Stack

- **[OpenCV](https://opencv.org/)** — Face detection (Haar cascades) + LBPH recognition
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** — Modern dark-mode UI framework
- **[pyttsx3](https://pyttsx3.readthedocs.io/)** — Offline text-to-speech engine
- **[SQLite3](https://docs.python.org/3/library/sqlite3.html)** — Embedded database (Python standard library)
- **[Pillow](https://pillow.readthedocs.io/)** — Image processing for Tkinter display

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/lukeewarmcoder">lukeewarmcoder</a>
</p>
