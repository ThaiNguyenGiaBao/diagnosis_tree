import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
from io import BytesIO
import logging as log
from typing import Optional, List, Union
from datetime import timedelta
import time
from PIL import Image

load_dotenv()

class MinioService:
    def __init__(self, bucket_name="smartfarm"):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost")
        self.port = int(os.getenv("MINIO_PORT", 9000))
        self.use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket_name = bucket_name
        
        self.client = Minio(
            f"{self.endpoint}:{self.port}",
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.use_ssl
        )
        
        print(f"Connected to MinIO at {self.endpoint}:{self.port}, SSL: {self.use_ssl}")
        
    def upload_resize_image(self, file_name: str, file_stream: bytes):
        sizes = {
            "small": 200,
            "medium": 800,
            "large": 1600,
        }

        try:
            original = Image.open(BytesIO(file_stream))
            upload_errors = []

            for size_label, width in sizes.items():
                img = original.copy()
                img.thumbnail((width, width * 10_000))  # keep aspect ratio, very tall max height
                buffer = BytesIO()
                img.save(buffer, format="AVIF")  # requires Pillow compiled with AVIF support
                avif_bytes = buffer.getvalue()

                try:
                    self.client.put_object(
                        self.bucket_name,
                        f"{size_label}/{file_name}",
                        BytesIO(avif_bytes),
                        length=len(avif_bytes),
                        content_type="image/avif",
                    )
                except S3Error as e:
                    log.error("Error uploading %s size for %s: %s", size_label, file_name, e)
                    upload_errors.append((size_label, str(e)))

            if upload_errors:
                raise Exception(f"Some uploads failed: {upload_errors}")

            return file_name
        except Exception as e:
            log.error("upload_resize_image error: %s", e)
            raise
    
    def get_presigned_url(
        self,
        file_key: Optional[str],
        size: str = "medium",
        expires_in: int = 60 * 60 * 24,
    ) -> str:
        if not file_key:
            raise ValueError("File key is required")

        try:
            return self.client.presigned_get_object(
                self.bucket_name,
                f"{size}/{file_key}",
                expires=timedelta(seconds=expires_in),
                response_headers={
                    "response-content-disposition": f'inline; filename="{size}/{file_key}"',
                    "response-content-type": "image/avif",
                },
            )
        except S3Error as e:
            log.error("get_presigned_url error: %s", e)
            raise
        
    def generate_file_key(self, original_name: str) -> str:
        ts = int(time.time() * 1000)
        # safe replacement of spaces
        safe = original_name.replace(" ", "_")
        return f"{ts}.{safe}"
    
            
            
minio_service = MinioService()

