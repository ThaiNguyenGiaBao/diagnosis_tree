from pydantic import BaseModel, Field
from typing import List, Optional

class Detection(BaseModel):
    label: str
    confidence: Optional[float] = None
    box_2d: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized 0..1000")


class DetectRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to analyze")
    
class DetectResponse(BaseModel):
    presigned_url: Optional[str] = Field(None, description="Presigned URL to the annotated image")
    analysis_vn: Optional[dict] = Field(None, description="Analysis results from the model")
