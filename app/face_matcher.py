from deepface import DeepFace
import tempfile
import os

def compare_faces(img1_bytes, img2_bytes):
    # save images temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
        f1.write(img1_bytes)
        img1_path = f1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
        f2.write(img2_bytes)
        img2_path = f2.name

    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            enforce_detection=True
        )
        return result

    finally:
        os.remove(img1_path)
        os.remove(img2_path)
