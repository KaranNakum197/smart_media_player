import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = "gestures/data"
labels = os.listdir(data_dir)
img_size = 64

X, y = [], []
label_map = {}

# Load images and labels
for idx, label in enumerate(labels):
    label_map[idx] = label
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(idx)

X = np.array(X) / 255.0  # Normalize
y = to_categorical(y)    # One-hot encode

# Save label names
with open("gestures/labels.txt", "w") as f:
    for idx in label_map:
        f.write(f"{idx} {label_map[idx]}\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save trained model
model.save("gesture_media_player/gestures/gesture_model.h5")
print("✅ Model trained and saved as gestures/gesture_model.h5")
