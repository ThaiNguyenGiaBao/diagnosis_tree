import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

def parse_detections(text: str)-> Tuple[List[dict], Dict[str, Any]]:
    """
    Forgiving parser: attempt strict JSON first; else extract first [...] block and parse.
    Returns a list of dicts (possibly empty).
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            detections = obj.get("detections", []) or []
            analysis = obj.get("analysis_vn", {}) or {}
            return detections, analysis
        if isinstance(obj, list):
            return obj, {}
    except Exception:
        pass


def to_pixels(box: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized [ymin,xmin,ymax,xmax] on 0..1000 -> pixel x1,y1,x2,y2
    """
    ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) for v in box[:4]]
    y1, x1 = int(ymin / 1000.0 * H), int(xmin / 1000.0 * W)
    y2, x2 = int(ymax / 1000.0 * H), int(xmax / 1000.0 * W)
    x1, x2 = max(0, min(W - 1, min(x1, x2))), max(0, min(W - 1, max(x1, x2)))
    y1, y2 = max(0, min(H - 1, min(y1, y2))), max(0, min(H - 1, max(y1, y2)))
    return x1, y1, x2, y2


def annotate_image(image_bytes: bytes, detections: List[dict], target_width: int = 1200) -> bytes:
    """
    Rescale image so width == target_width (preserve aspect ratio),
    draw detections, and return PNG bytes of the annotated (scaled) image.
    """
    # load image
    base = Image.open(BytesIO(image_bytes)).convert("RGBA")
    orig_W, orig_H = base.size

    # compute scale to reach target_width
    if orig_W == 0:
        raise ValueError("Image width is zero")
    scale = target_width / orig_W
    # guard: avoid absurdly large sizes
    max_width = 4000
    if target_width > max_width:
        target_width = max_width
        scale = target_width / orig_W

    new_W = int(round(orig_W * scale))
    new_H = int(round(orig_H * scale))

    # resize using LANCZOS for quality
    if (new_W, new_H) != (orig_W, orig_H):
        base = base.resize((new_W, new_H), resample=Image.LANCZOS)

    overlay = Image.new("RGBA", (new_W, new_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # font size depends on scaled image size
    try:
        font_size = max(12, int(min(new_W, new_H) / 45))
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    thickness = max(1, int(round((new_W + new_H) / 900)))
    color_box = (255, 0, 0, 220)
    color_text = (255, 255, 255, 255)
    color_bg = (0, 0, 0, 200)

    for d in detections:
        box = d.get("box_2d") or d.get("bbox") or d.get("box")
        if not box or len(box) < 4:
            continue

        # convert normalized box -> pixels on the scaled image
        # reuse your to_pixels but pass new_W, new_H
        x1, y1, x2, y2 = to_pixels(box, new_W, new_H)
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue

        # draw rectangle (use width param if available)
        try:
            draw.rectangle([x1, y1, x2, y2], outline=color_box, width=thickness)
        except TypeError:
            for t in range(thickness):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color_box)

        # label + confidence formatting
        label = str(d.get("label") or "unknown")
        conf = d.get("confidence")
        try:
            if conf is not None:
                conf = float(conf)
                if conf > 1:
                    conf = conf / 100.0
        except Exception:
            conf = None
        text = f"{label}{'' if conf is None else f' {conf:.2f}'}"

        # measure text size robustly
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            try:
                tw, th = font.getsize(text)
            except Exception:
                tw, th = (len(text) * 6, 12)

        pad = max(4, th // 3)
        bx0 = x1
        by0 = max(0, y1 - th - 2 * pad)
        if by0 == 0:  # if not enough room above, place below
            by0 = y1 + pad
        bx1 = bx0 + tw + 2 * pad
        by1 = by0 + th + 2 * pad

        # draw label background and text
        draw.rectangle([bx0, by0, bx1, by1], fill=color_bg)
        draw.text((bx0 + pad, by0 + pad), text, fill=color_text, font=font)

    # composite overlay onto base and return PNG bytes
    out = Image.alpha_composite(base, overlay).convert("RGB")
    with BytesIO() as out_buf:
        out.save(out_buf, format="PNG", quality=95)
        return out_buf.getvalue()
