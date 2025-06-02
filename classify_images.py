import os
import cv2
import shutil

SOURCE_DIR = "passports"
VISA_DIR = "visa"
PERSONAL_DIR = "personal"

os.makedirs(VISA_DIR, exist_ok=True)
os.makedirs(PERSONAL_DIR, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def classify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Couldn't read image: {image_path}")
        return "unknown"

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"❌ No face detected in {os.path.basename(image_path)}")
        return "unknown"

    x, y, w, h = faces[0]
    face_area = w * h
    face_y_ratio = y / height
    face_size_ratio = face_area / (height * width)

    print(f"🧪 {os.path.basename(image_path)} → area: {round(face_size_ratio, 3)}, y_ratio: {round(face_y_ratio, 2)}")

    # Adjusted thresholds
    if face_size_ratio < 0.08 and face_y_ratio < 0.35:
        return "visa"
    else:
        return "personal"

# Process all images
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(".jpg"):
        full_path = os.path.join(SOURCE_DIR, filename)
        category = classify_image(full_path)

        if category == "visa":
            shutil.move(full_path, os.path.join(VISA_DIR, filename))
            print(f"📄 {filename} → visa")
        elif category == "personal":
            shutil.move(full_path, os.path.join(PERSONAL_DIR, filename))
            print(f"🧍 {filename} → personal")
        else:
            print(f"⚠️ {filename} → skipped")

print("✅ Classification complete.")