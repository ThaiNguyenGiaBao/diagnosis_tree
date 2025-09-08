# health.router.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional

from .schema import DetectRequest, DetectResponse, Detection
from .service import HealthService
import os

import shutil
import uuid

router = APIRouter(prefix="/api/health", tags=["health"])

# create a singleton HealthService instance (or use DI in your app startup)
health_service = HealthService()

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "file")

@router.post("/detect", response_model=DetectResponse)
async def detect_endpoint(payload: DetectRequest = None):
    if payload is None or not payload.image_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image_url is required")

    try:
        print("Received detection request for URL:", payload.image_url)
        presigned, analysis_vn = health_service.detect_from_url(payload.image_url)

        return DetectResponse(presigned_url=presigned, analysis_vn=analysis_vn)
    except HTTPException:
        raise
    except Exception as exc:
        # log exception server-side in production
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")

@router.get("/", include_in_schema=False)
async def health_index():
    # adjust path to your project structure
    file_path = os.path.join(os.path.dirname(__file__), "view", "index.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(file_path, media_type="text/html")

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1] or ".png"
        filename = f"{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(PUBLIC_DIR, filename)

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Construct a URL (assuming you mount StaticFiles at /public)
        url = f"/api/health/public/{filename}"
        return {"image_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    

@router.get("/public/{filename}", include_in_schema=False)
async def serve_image(filename: str):
    file_path = os.path.join(PUBLIC_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/png")
