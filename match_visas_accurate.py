import os
import shutil
from deepface import DeepFace
from scipy.spatial.distance import cosine
from itertools import product

# Directories
PERSONAL_DIR = "personal"
VISA_DIR = "visa"
OUTPUT_DIR = "people"
UNMATCHED_DIR = "unmatched"

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(UNMATCHED_DIR, "personal"), exist_ok=True)
os.makedirs(os.path.join(UNMATCHED_DIR, "visa"), exist_ok=True)

# Get embedding for one image
def get_embedding(path):
    try:
        result = DeepFace.represent(img_path=path, model_name="Facenet", enforce_detection=False)
        return result[0]['embedding']
    except Exception as e:
        print(f"❌ Failed: {path} — {e}")
        return None

# Load images and generate embeddings
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            embedding = get_embedding(path)
            if embedding:
                images.append({
                    'filename': filename,
                    'path': path,
                    'embedding': embedding,
                    'matched': False
                })
    return images

personal_images = load_images_from_folder(PERSONAL_DIR)
visa_images = load_images_from_folder(VISA_DIR)

# Calculate all pair distances
pairs = []
for p, v in product(personal_images, visa_images):
    distance = cosine(p['embedding'], v['embedding'])
    pairs.append((distance, p, v))

# Sort by similarity (lower = more similar)
pairs.sort(key=lambda x: x[0])

person_id = 1
used_personals = set()
used_visas = set()

# Match based on confidence
for distance, personal, visa in pairs:
    if personal['filename'] in used_personals or visa['filename'] in used_visas:
        continue

    if distance >= 0.25:  # strict threshold
        continue

    folder = os.path.join(OUTPUT_DIR, str(person_id))
    os.makedirs(folder, exist_ok=True)

    shutil.move(personal['path'], os.path.join(folder, personal['filename']))
    shutil.move(visa['path'], os.path.join(folder, visa['filename']))

    used_personals.add(personal['filename'])
    used_visas.add(visa['filename'])

    print(f"✅ High-confidence match {personal['filename']} + {visa['filename']} → Folder {person_id} (score: {round(1 - distance, 3)})")
    person_id += 1

# Handle unmatched personals
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
        print(f"⚠️ Unmatched personal: {p['filename']}")

# Handle unmatched visas
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
        print(f"⚠️ Unmatched visa: {v['filename']}")

print("🎯 Strict confidence-based matching complete.")
