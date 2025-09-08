# health.service.py
import os

from typing import List, Tuple, Optional, Dict, Any

import requests

from provider.minIO import minio_service  # adjust import according to your project

from .schema import Detection  # relative import
from common.utils import parse_detections , annotate_image 

from common.ai_model.implements.gemini import geminiModel


API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY or GENAI_API_KEY in your environment")


PROMPT = (
    "Bạn là chuyên gia nông học và sâu bệnh cây trồng. "
    "Dựa vào ảnh được gửi kèm, hãy phân tích và trả về DUY NHẤT một đối tượng JSON (KHÔNG kèm bất kỳ văn bản giải thích nào ngoài JSON). "
    "Các đề xuất hành động phải phù hợp ở Việt Nam, sử dụng tên thân thuộc, ưu tiên thuốc sinh học và biện pháp hữu cơ."
    "Định dạng JSON phải chính xác theo mô tả sau:\n\n"
    "1) \"detections\": mảng các đối tượng phát hiện (có thể rỗng []). Mỗi đối tượng detection có keys:\n"
    "   - label: (string) tên bệnh/sâu/hại (ví dụ: \"whiteflies\", \"rice_leaf_blast\")\n"
    "   - confidence: (number) độ tin cậy 0..1\n"
    "   - box_2d: (array of 4 numbers) [ymin, xmin, ymax, xmax] **normalized 0..1000** (integer hoặc float)\n\n"
    "2) \"analysis_vn\": object chứa các trường mô tả kết luận và khuyến nghị bằng TIẾNG VIỆT:\n"
    "   - prediction: (string) dự đoán ngắn gọn (ví dụ: \"Cây bị rệp trắng\")\n"
    "   - severity_level: (string) một trong [\"Thấp\",\"Trung bình\",\"Cao\", \"Rất cao\"]\n"
    "   - possible_causes: (array of objects) các nguyên nhân khả dĩ khác, mỗi item có keys:\n"
    "       * cause: (string) mô tả nguyên nhân,\n"
    "       * confidence: (number) độ tin cậy 0..1 cho nguyên nhân này,\n"
    "       * evidence: (string) bằng chứng/quan sát ngắn (ví dụ: 'vết vàng theo gân', 'lá cuốn lại', 'độ ẩm cao')\n"
    "   - recommended_actions: (array of objects) danh sách hành động theo bước, mỗi item có keys:\n"
    "       * name: (string) mô tả ngắn gọn, khoảng 10 từ (ví dụ: \"Phun thuốc sinh học Azadirachtin 3ml/l\"),\n"
    "       * timing: (string) khi nào thực hiện (ví dụ: \"Ngay lập tức\", \"Trong 3-5 ngày\"),\n"
    "       * description: (string, optional) mô tả chi tiết, lưu ý an toàn/độ pha/điều kiện, ...\n"
    "       * targetValue: (number) số lần cần thực hiện tronaa tuần (ví dụ: 1, 2, 3...)\n"
    "       * numOfWeeks: (number) số tuần cần thực hiện (ví dụ: 1, 2, 3...)\n"
    "   - chemical_recommendations: (array of strings) gợi ý thuốc hóa học (tên hoạt chất hoặc thương hiệu) hoặc []\n"
    "   - biological_recommendations: (array of strings) gợi ý biện pháp sinh học/thuốc hữu cơ hoặc []\n"
    "   - monitoring_plan: (string) kế hoạch theo dõi (ví dụ: \"Kiểm tra lại sau 7 ngày, tập trung vào tán lá phía dưới\")\n"
    "   - preventive_measures: (array of strings) các biện pháp phòng ngừa ngắn gọn\n"
    "   - additional_notes: (string) (tùy chọn)\n\n"
    "Ví dụ JSON mẫu (bắt buộc format tương tự):\n"
    "{\n"
    "  \"detections\": [\n"
    "    {\"label\":\"whiteflies\",\"confidence\":0.88,\"box_2d\":[120,250,260,420]}\n"
    "  ],\n"
    "  \"analysis_vn\": {\n"
    "    \"prediction\":\"Cây bị rệp trắng (whiteflies)\",\n"
    "    \"severity_level\":\"Cao\",\n"
    "    \"possible_causes\":[\n"
    "      {\"cause\":\"Điều kiện ấm ẩm, nhiều cỏ dại quanh gốc\",\"confidence\":0.75,\"evidence\":\"nhiều chấm trắng tập trung mặt dưới lá\"},\n"
    "      {\"cause\":\"Thiếu thông gió do cây trồng quá dày\",\"confidence\":0.45,\"evidence\":\"lá gần nhau, ít ánh sáng vào tán\"}\n"
    "    ],\n"
    "    \"recommended_actions\": [\n"
    "      {\"name\":\"Phun thuốc sinh học có hoạt chất Azadirachtin\",\"timing\":\"Ngay lập tức\",\"description\":\"Phun vào sáng sớm hoặc chiều mát\", \"targetValue\": 2, \"numOfWeeks\": 1},\n"
    "      {\"name\":\"Cắt bỏ lá bị nhiễm nặng và tiêu hủy\",\"timing\":\"Trong 1-2 ngày\",\"description\":\"Đeo găng tay, tiêu hủy xa vườn\", \"targetValue\": 2, \"numOfWeeks\": 1}\n"
    "    ],\n"
    "    \"chemical_recommendations\":[\"Imidacloprid (dùng thận trọng)\"],\n"
    "    \"biological_recommendations\":[\"Thả thiên địch (bọ rùa), dùng bẫy dính vàng\"],\n"
    "    \"monitoring_plan\":\"Kiểm tra lại sau 7 ngày; nếu số lượng không giảm, lặp lại phun theo hướng dẫn.\",\n"
    "    \"preventive_measures\":[\"Cắt tỉa thông thoáng\",\"Quản lý cỏ dại quanh gốc\"],\n"
    "    \"additional_notes\":\"Tránh phun lúc nắng gắt; tuân thủ liều lượng nhà sản xuất.\"\n"
    "  }\n"
    "}\n\n"
    "QUAN TRỌNG: Nếu không phát hiện được bất kỳ dấu hiệu nào, hãy trả:\n"
    "{\"detections\": [], \"analysis_vn\": {\"prediction\":\"Không phát hiện bệnh rõ rệt\",\"confidence_overall\":0.0, \"severity_level\":\"LOW\", \"severity_percent\":0, \"most_likely_cause\":\"\", \"possible_causes\":[], \"recommended_actions\":[], \"chemical_recommendations\":[], \"biological_recommendations\":[], \"monitoring_plan\":\"\", \"preventive_measures\":[], \"additional_notes\":\"\"}}\n\n"
    "KHÔNG được thêm text, chú thích hay giải thích ngoài JSON này. TẤT CẢ phần phân tích phải bằng tiếng Việt."
)


class HealthService:
    """
    Lightweight service for disease detection using a multimodal LLM (Gemini).
    """

    def __init__(self, minio=minio_service, model_name: str = "gemini-2.5-flash"):
        self.minio = minio

    def detect_from_url(self, image_url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Download an image, run detection, annotate, upload annotated image, return presigned URL and detection list.
        """
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        image_bytes = r.content
        return self._detect_and_store(image_bytes)

    def detect_from_bytes(self, image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Run detection on raw image bytes.
        """
        return self._detect_and_store(image_bytes)


    def _detect_and_store(self, image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Full pipeline: call model, annotate, upload to minio, return presigned url and typed detections.
        """
        raw_dets, analysis_vn  = geminiModel.generate_json_response(image_bytes, PROMPT)
        
        # convert to typed models
        typed = []
        for d in raw_dets:
            try:
                label = str(d.get("label") or "unknown")
                conf = d.get("confidence")
                if conf is not None:
                    conf = float(conf)
                    if conf > 1:
                        conf = conf / 100.0
                box = [float(x) for x in d.get("box_2d")[:4]]
                typed.append(Detection(label=label, confidence=conf, box_2d=box))
            except Exception:
                # skip malformed detection
                continue

        annotated_png = annotate_image(image_bytes, raw_dets)
        file_key = self.minio.generate_file_key("annotated.png")

        try:
            self.minio.upload_resize_image(file_key, annotated_png)
            presigned = self.minio.get_presigned_url(file_key,size="medium", expires_in=24 * 3600)
        except Exception:
            raise RuntimeError("Failed to upload or get presigned URL from MinIO")
        return presigned, analysis_vn
