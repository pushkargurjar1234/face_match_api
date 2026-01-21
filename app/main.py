from fastapi import FastAPI, UploadFile, File
from app.face_matcher import compare_faces

app = FastAPI(title="Face Match API")

@app.post("/compare-faces")
async def compare_faces_api(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()

    result = compare_faces(img1_bytes, img2_bytes)

    return {
        "matched": result["verified"],
        "confidence_distance": result["distance"],
        "threshold": result["threshold"]
    }
