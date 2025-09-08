from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class AIModelInterface(ABC):
    @abstractmethod
    def generate_json_response(self, image_bytes: bytes, prompt) -> Tuple[List[dict], Dict[str, Any]]:
        pass
