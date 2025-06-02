import os
import shutil
from deepface import DeepFace
from scipy.spatial.distance import cosine

PERSONAL_DIR = "personal"
VISA_DIR = "visa"
OUTPUT_DIR = "people"
UNMATCHED_DIR = "unmatched"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UNMATCHED_DIR + "/personal", exist_ok=True)
os.makedirs(UNMATCHED_DIR + "/visa", exist_ok=True)

def get_embedding(path):
    try:
        result = DeepFace.represent(img_path=path, model_name="Facenet", enforce_detection=False)
        return result[0]['embedding']
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(path)}: {e}")
        return None

def is_match(e1, e2, threshold=0.5):
    return cosine(e1, e2) < threshold

# Load all personal images
personal_images = []
for filename in os.listdir(PERSONAL_DIR):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(PERSONAL_DIR, filename)
        embedding = get_embedding(path)
        if embedding:
            personal_images.append({'filename': filename, 'path': path, 'embedding': embedding, 'matched': False})

# Load all visa images
visa_images = []
for filename in os.listdir(VISA_DIR):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(VISA_DIR, filename)
        embedding = get_embedding(path)
        if embedding:
            visa_images.append({'filename': filename, 'path': path, 'embedding': embedding, 'matched': False})

# Match each personal with a visa
person_id = 1
for personal in personal_images:
    best_match = None
    best_score = 1.0

    for visa in visa_images:
        if not visa['matched']:
            score = cosine(personal['embedding'], visa['embedding'])
            if score < best_score:
                best_score = score
                best_match = visa

    if best_match and best_score < 0.5:
        folder = os.path.join(OUTPUT_DIR, str(person_id))
        os.makedirs(folder, exist_ok=True)
        shutil.copy(personal['path'], os.path.join(folder, personal['filename']))
        shutil.copy(best_match['path'], os.path.join(folder, best_match['filename']))
        personal['matched'] = True
        best_match['matched'] = True
        print(f"✅ Matched {personal['filename']} with {best_match['filename']} → Folder {person_id}")
        person_id += 1
    else:
        print(f"⚠️ No match for {personal['filename']}")
        shutil.copy(personal['path'], os.path.join(UNMATCHED_DIR, "personal", personal['filename']))

# Save unmatched visa images
for visa in visa_images:
    if not visa['matched']:
        shutil.copy(visa['path'], os.path.join(UNMATCHED_DIR, "visa", visa['filename']))
        print(f"⚠️ Unmatched visa image: {visa['filename']}")

print("🎯 Matching complete.")
