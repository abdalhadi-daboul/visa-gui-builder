import os
import cv2
import shutil
from tkinter import Tk, Button, Label, messagebox
from tkinter import filedialog as fd
import tensorflow
from tensorflow.keras.preprocessing import image  # <- this "warms up" tensorflow
from deepface import DeepFace
from scipy.spatial.distance import cosine
from itertools import product
import sys
print("🚀 Using Python executable:", sys.executable)

# Directories
PASSPORTS_DIR = "passports"
PERSONAL_DIR = "personal"
VISA_DIR = "visa"
PEOPLE_DIR = "people"
UNMATCHED_DIR = "unmatched"

for dir_path in [PASSPORTS_DIR, PERSONAL_DIR, VISA_DIR,
                 os.path.join(UNMATCHED_DIR, "personal"),
                 os.path.join(UNMATCHED_DIR, "visa"),
                 PEOPLE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # When bundled by PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

face_cascade = cv2.CascadeClassifier(resource_path("haarcascade_frontalface_default.xml"))

def classify_images():
    for filename in os.listdir(PASSPORTS_DIR):
        if filename.lower().endswith(".jpg"):
            full_path = os.path.join(PASSPORTS_DIR, filename)
            img = cv2.imread(full_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            face_area = w * h
            face_y_ratio = y / height
            face_size_ratio = face_area / (height * width)
            if face_size_ratio < 0.08 and face_y_ratio < 0.35:
                shutil.move(full_path, os.path.join(VISA_DIR, filename))
            else:
                shutil.move(full_path, os.path.join(PERSONAL_DIR, filename))
    messagebox.showinfo("Done", "✅ Classification complete")

# Matching

def get_embedding(path):
    try:
        result = DeepFace.represent(img_path=path, model_name="Facenet", enforce_detection=False)
        return result[0]['embedding']
    except:
        return None

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            embedding = get_embedding(path)
            if embedding:
                images.append({'filename': filename, 'path': path, 'embedding': embedding})
    return images

def match_images():
    personal_images = load_images(PERSONAL_DIR)
    visa_images = load_images(VISA_DIR)

    pairs = []
    for p, v in product(personal_images, visa_images):
        distance = cosine(p['embedding'], v['embedding'])
        pairs.append((distance, p, v))
    pairs.sort(key=lambda x: x[0])

    used_personals, used_visas = set(), set()
    person_id = 1
    for distance, personal, visa in pairs:
        if personal['filename'] in used_personals or visa['filename'] in used_visas:
            continue
        if distance >= 0.25:
            continue
        folder = os.path.join(PEOPLE_DIR, str(person_id))
        os.makedirs(folder, exist_ok=True)
        shutil.move(personal['path'], os.path.join(folder, personal['filename']))
        shutil.move(visa['path'], os.path.join(folder, visa['filename']))
        used_personals.add(personal['filename'])
        used_visas.add(visa['filename'])
        person_id += 1

    for p in personal_images:
        if p['filename'] not in used_personals:
            dst = os.path.join(UNMATCHED_DIR, "personal", p['filename'])
            if os.path.exists(dst):
                name, ext = os.path.splitext(p['filename'])
                i = 1
                while os.path.exists(os.path.join(UNMATCHED_DIR, "personal", f"{name}_{i}{ext}")):
                    i += 1
                dst = os.path.join(UNMATCHED_DIR, "personal", f"{name}_{i}{ext}")
            shutil.move(p['path'], dst)

    for v in visa_images:
        if v['filename'] not in used_visas:
            dst = os.path.join(UNMATCHED_DIR, "visa", v['filename'])
            if os.path.exists(dst):
                name, ext = os.path.splitext(v['filename'])
                i = 1
                while os.path.exists(os.path.join(UNMATCHED_DIR, "visa", f"{name}_{i}{ext}")):
                    i += 1
                dst = os.path.join(UNMATCHED_DIR, "visa", f"{name}_{i}{ext}")
            shutil.move(v['path'], dst)

    messagebox.showinfo("Done", "🎯 Matching complete")

# GUI
root = Tk()
root.title("Visa Matcher Tool")
root.geometry("300x200")

Label(root, text="Visa-Personal Matcher", font=("Arial", 14)).pack(pady=10)
Button(root, text="📂 Classify Images", width=25, command=classify_images).pack(pady=10)
Button(root, text="🧠 Match Visas", width=25, command=match_images).pack(pady=10)

root.mainloop()
