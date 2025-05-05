# gesture_gui_app.py
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import shutil
import os

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Media Controller")
        self.root.geometry("400x350")

        self.video_path = tk.StringVar()

        tk.Label(root, text="Gesture Media Player", font=("Helvetica", 16)).pack(pady=10)

        tk.Button(root, text="1. Select Video", command=self.select_video, width=30).pack(pady=5)
        tk.Button(root, text="2. Capture Gestures", command=self.capture_gestures, width=30).pack(pady=5)
        tk.Button(root, text="3. Train Gesture Model", command=self.train_model, width=30).pack(pady=5)
        tk.Button(root, text="4. Run Gesture Control", command=self.run_gesture_control, width=30).pack(pady=5)

        self.status_label = tk.Label(root, text="Status: Ready", fg="green")
        self.status_label.pack(pady=20)

    def select_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv")])
        if filepath:
            self.video_path.set(filepath)
            video_dest = os.path.join("media", "your_video.mp4")
            shutil.copy(filepath, video_dest)
            self.status_label.config(text="Selected video copied to media/", fg="blue")

    def capture_gestures(self):
        self.status_label.config(text="Launching gesture capture...", fg="orange")
        try:
            subprocess.run(["python", "gestures/capture_gestures.py"], check=True)
            self.status_label.config(text="Gesture capture complete.", fg="green")
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Failed to launch gesture capture.")
            self.status_label.config(text="Capture failed.", fg="red")

    def train_model(self):
        self.status_label.config(text="Training model...", fg="orange")
        try:
            subprocess.run(["python", "gestures/train_model.py"], check=True)
            self.status_label.config(text="Model trained successfully.", fg="green")
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Failed to train the model.")
            self.status_label.config(text="Training failed.", fg="red")

    def run_gesture_control(self):
        self.status_label.config(text="Running gesture control...", fg="purple")
        try:
            subprocess.Popen(["python", "gesture_control.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start gesture control.\n{e}")
            self.status_label.config(text="Failed to run.", fg="red")

if __name__ == "__main__":
    if not os.path.exists("media"):
        os.makedirs("media")

    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
