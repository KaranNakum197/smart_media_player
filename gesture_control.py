import cv2
import numpy as np
from tensorflow.keras.models import load_model
import vlc
import time
import json


model = load_model("gestures/gesture_model.h5")

with open("gestures/labels.txt", "r") as f:
    label_map = {int(line.split()[0]): line.strip().split()[1] for line in f.readlines()}

with open("gestures/gesture_map.json", "r") as f:
    gesture_actions = json.load(f)

media_path = "media/your_video.mp4"  # Update this path
player = vlc.MediaPlayer(media_path)
player.play()
time.sleep(1)  # Give it a moment to load

def perform_action(action):
    if action == "play":
        player.play()
    elif action == "pause":
        player.pause()
    elif action == "volume_up":
        vol = player.audio_get_volume()
        player.audio_set_volume(min(100, vol + 10))
    elif action == "volume_down":
        vol = player.audio_get_volume()
        player.audio_set_volume(max(0, vol - 10))
    elif action == "previous":
        current_time = player.get_time()
        player.set_time(max(0, current_time - 10000))  # 10 sec back
    elif action == "next":
        current_time = player.get_time()
        player.set_time(current_time + 10000)  # 10 sec forward
    elif action == "exit":
        print("Exiting...")
        player.stop()
        exit()

cap = cv2.VideoCapture(0)
img_size = 64
cooldown = 2  # seconds between actions
last_action_time = time.time()

print("🖐️ Show a gesture in the green box...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (img_size, img_size)) / 255.0
    
    roi = np.stack([roi]*3, axis=-1)  # Shape: (64, 64, 3)
    roi_input = np.expand_dims(roi, axis=0)  # Shape: (1, 64, 64, 3)

    pred = model.predict(roi_input, verbose=0)
    class_id = np.argmax(pred)
    gesture = label_map[class_id]
    confidence = np.max(pred)

    cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 2)
    cv2.putText(frame, f"{gesture} ({confidence:.2f})", (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  

    cv2.imshow("Gesture Control", frame)

   
    if confidence > 0.70 and (time.time() - last_action_time) > cooldown:
        if gesture in gesture_actions:
            action = gesture_actions[gesture]
            print(f"🟢 Recognized: {gesture} → {action}")
            perform_action(action)
            last_action_time = time.time()

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()