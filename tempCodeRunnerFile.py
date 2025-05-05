import cv2
import os

# Config
gestures = ["play", "pause", "volume_up", "volume_down", "previous", "next", "exit"]
save_dir = "gestures/data"
num_samples = 300  # images per gesture

os.makedirs(save_dir, exist_ok=True)

def collect_images(gesture_name):
    cap = cv2.VideoCapture(0)
    print(f"Collecting for gesture: {gesture_name}. Press 's' to start, 'q' to quit.")

    count = 0
    gesture_path = os.path.join(save_dir, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 2)
        cv2.putText(frame, f"{gesture_name}: {count}/{num_samples}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Capture Gesture", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            roi = frame[50:250, 50:250]
            cv2.imwrite(f"{gesture_path}/{count}.jpg", roi)
            count += 1
        elif key == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images for {gesture_name}")

if __name__ == "__main__":
    for gesture in gestures:
        collect_images(gesture)