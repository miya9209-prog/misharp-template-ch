import io
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cv2
    CV2_OK = True
except Exception:
    cv2 = None
    CV2_OK = False
import numpy as np
import streamlit as st
from PIL import Image, ImageColor, ImageDraw, ImageFont

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except Exception:
    CANVAS_OK = False

APP_TITLE = "미샵 템플릿 OS"
APP_SUBTITLE = "상세페이지 자동화를 위한 이미지 편집 템플릿 제공"
COPYRIGHT = "made by MISHARP COMPANY, MIYAWA, 2026. 이 프로그램은 미샵컴퍼니 내부직원용이며 외부유출 및 무단 사용을 금합니다."
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
EXAMPLE_DIR = BASE_DIR / "examples"
OUTPUT_DIR = BASE_DIR / "outputs"
ASSET_DIR = BASE_DIR / "template_assets"
for d in [TEMPLATE_DIR, EXAMPLE_DIR, OUTPUT_DIR, ASSET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CANVAS_W = 900
DEFAULT_CANVAS_H = 1200
DEFAULT_FONT_SIZE = 42
SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------- base utils ----------
def safe_name(text: str) -> str:
    keep = []
    for ch in (text or "").strip():
        if ch.isalnum() or ch in "-_ ":
            keep.append(ch)
        elif ord(ch) > 127:
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out or datetime.now().strftime("template_%Y%m%d_%H%M%S")


def ensure_examples_loaded():
    zip_path = Path("/mnt/data/상페 상단 디자인.zip")
    if zip_path.exists() and not any(EXAMPLE_DIR.iterdir()):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                suffix = Path(name).suffix.lower()
                if suffix in SUPPORTED_IMAGE_EXT:
                    target = EXAMPLE_DIR / Path(name).name
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)


def list_example_images() -> List[Path]:
    ensure_examples_loaded()
    return sorted(
        [p for p in EXAMPLE_DIR.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXT],
        key=lambda p: p.name.lower(),
    )


def list_templates() -> List[Path]:
    return sorted(TEMPLATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_template(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_template(data: Dict) -> Path:
    data = dict(data)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    name = safe_name(data.get("template_name") or "template")
    path = TEMPLATE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    return ImageColor.getrgb(hex_color)


def locate_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


FONT_PATH = locate_font()


def get_font(size: int):
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


def fit_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0 or target_w <= 0 or target_h <= 0:
        return Image.new("RGB", (max(1, target_w), max(1, target_h)), (255, 255, 255))
    src_ratio = src_w / src_h
    dst_ratio = target_w / target_h
    if src_ratio > dst_ratio:
        new_h = target_h
        new_w = int(target_h * src_ratio)
    else:
        new_w = target_w
        new_h = int(target_w / src_ratio)
    resized = img.resize((new_w, new_h))
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    return resized.crop((left, top, left + target_w, top + target_h))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    if not text:
        return [""]
    text = str(text)
    if "\n" in text:
        lines = []
        for part in text.split("\n"):
            lines.extend(wrap_text(draw, part, font, max_width))
        return lines
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return [text]
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    final_lines = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= max_width:
            final_lines.append(line)
        else:
            buf = ""
            for ch in line:
                candidate = buf + ch
                if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    buf = candidate
                else:
                    if buf:
                        final_lines.append(buf)
                    buf = ch
            if buf:
                final_lines.append(buf)
    return final_lines


# ---------- AI recommendation ----------
def _clamp_box(x, y, bw, bh, w, h):
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    bw = max(1, min(int(bw), w - x))
    bh = max(1, min(int(bh), h - y))
    return x, y, bw, bh


def _fallback_suggest_regions(pil_img: Image.Image) -> Dict[str, List[Dict]]:
    """Non-cv2 fallback using brightness-change density across rows/cols."""
    img = np.array(pil_img.convert('L'))
    h, w = img.shape[:2]
    row_change = np.mean(np.abs(np.diff(img.astype(np.int16), axis=1)), axis=1)
    col_change = np.mean(np.abs(np.diff(img.astype(np.int16), axis=0)), axis=0)

    def smooth(arr, k):
        k = max(3, int(k) | 1)
        pad = k // 2
        arr2 = np.pad(arr, (pad, pad), mode='edge')
        return np.convolve(arr2, np.ones(k)/k, mode='valid')

    row_s = smooth(row_change, max(9, h // 40))
    col_s = smooth(col_change, max(9, w // 40))
    row_thr = float(np.percentile(row_s, 65)) if len(row_s) else 0
    col_thr = float(np.percentile(col_s, 65)) if len(col_s) else 0

    active_rows = np.where(row_s >= row_thr)[0]
    active_cols = np.where(col_s >= col_thr)[0]

    text_boxes = []
    if len(active_rows):
        top_band = active_rows[active_rows < int(h * 0.28)]
        if len(top_band):
            y1 = max(0, int(top_band.min()) - int(h * 0.02))
            y2 = min(h, int(top_band.max()) + int(h * 0.03))
            text_boxes.append((int(w * 0.08), y1, int(w * 0.84), max(int(h * 0.06), y2 - y1)))

        lower_band = active_rows[(active_rows > int(h * 0.62)) & (active_rows < int(h * 0.95))]
        if len(lower_band):
            y1 = max(0, int(lower_band.min()) - int(h * 0.015))
            y2 = min(h, int(lower_band.max()) + int(h * 0.02))
            text_boxes.append((int(w * 0.10), y1, int(w * 0.80), max(int(h * 0.05), y2 - y1)))

    image_boxes = []
    if len(active_cols) and len(active_rows):
        x1 = max(0, int(np.percentile(active_cols, 10)) - int(w * 0.03))
        x2 = min(w, int(np.percentile(active_cols, 90)) + int(w * 0.03))
        y1 = int(h * 0.17)
        y2 = int(h * 0.62)
        image_boxes.append((x1, y1, max(int(w * 0.45), x2 - x1), max(int(h * 0.24), y2 - y1)))
    else:
        image_boxes.append((int(w * 0.06), int(h * 0.18), int(w * 0.88), int(h * 0.42)))

    text_suggestions = []
    for idx, (x, y, bw, bh) in enumerate(text_boxes[:6], start=1):
        x, y, bw, bh = _clamp_box(x, y, bw, bh, w, h)
        text_suggestions.append({
            'id': f'text_{idx}', 'label': f'카피 {idx}', 'type': 'text',
            'x': x, 'y': y, 'w': bw, 'h': bh,
            'font_size': max(22, min(58, int(bh * 0.42))), 'font_color': '#111111' if y > h * 0.35 else '#FFFFFF',
            'align': 'center', 'default_text': f'카피 {idx} 입력', 'line_spacing': 8,
        })
    image_suggestions = []
    for idx, (x, y, bw, bh) in enumerate(image_boxes[:6], start=1):
        x, y, bw, bh = _clamp_box(x, y, bw, bh, w, h)
        image_suggestions.append({
            'id': f'image_{idx}', 'label': f'이미지 {idx}', 'type': 'image',
            'x': x, 'y': y, 'w': bw, 'h': bh, 'placeholder_rgb': [235, 235, 235],
        })
    return {'text_boxes': text_suggestions, 'image_slots': image_suggestions}

def _merge_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.15) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    picked = []
    boxes_arr = np.array(boxes)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = boxes_arr[:, 2] * boxes_arr[:, 3]
    idxs = np.argsort(areas)
    while len(idxs) > 0:
        last = idxs[-1]
        picked.append(tuple(boxes[last]))
        suppress = [len(idxs) - 1]
        for pos in range(len(idxs) - 1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            union = areas[last] + areas[i] - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return picked


def suggest_regions(pil_img: Image.Image) -> Dict[str, List[Dict]]:
    """Higher-precision block detector with cv2 when available; safe fallback otherwise."""
    if not CV2_OK:
        return _fallback_suggest_regions(pil_img)

    img = np.array(pil_img.convert('RGB'))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1) text candidates from blackhat + morphology
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 36), max(5, h // 120)))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    if grad_x.max() > grad_x.min():
        grad_x = 255 * ((grad_x - grad_x.min()) / (grad_x.max() - grad_x.min()))
    grad_x = grad_x.astype('uint8')
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 45), max(8, h // 90))))
    text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < w * h * 0.004 or bw < w * 0.12 or bh < h * 0.025:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.4:
            continue
        density = np.mean(text_mask[y:y+bh, x:x+bw] > 0)
        if density < 0.10:
            continue
        text_boxes.append((x, y, bw, bh))
    text_boxes = _merge_boxes(text_boxes, iou_threshold=0.12)

    # 2) image candidates from low-text, large, edge-bounded regions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 90), max(9, h // 90))), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < w * h * 0.018 or bw < w * 0.18 or bh < h * 0.10:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 4.5:
            continue
        roi = gray[y:y+bh, x:x+bw]
        if roi.size == 0:
            continue
        stdv = float(np.std(roi))
        if stdv < 10:
            continue
        overlap = False
        for tx, ty, tw, th in text_boxes:
            xx1, yy1 = max(x, tx), max(y, ty)
            xx2, yy2 = min(x+bw, tx+tw), min(y+bh, ty+th)
            inter = max(0, xx2-xx1) * max(0, yy2-yy1)
            if inter / area > 0.28:
                overlap = True
                break
        if not overlap:
            image_boxes.append((x, y, bw, bh))
    image_boxes = _merge_boxes(image_boxes, iou_threshold=0.18)

    # 3) whitespace-aware structural priors for detail-page top sections
    row_mean = gray.mean(axis=1)
    row_var = gray.var(axis=1)
    bright = row_mean > np.percentile(row_mean, 60)
    calm = row_var < np.percentile(row_var, 55)
    white_rows = np.where(bright & calm)[0]
    if len(white_rows):
        top_rows = white_rows[white_rows < int(h * 0.24)]
        if len(top_rows):
            y1 = max(0, int(top_rows.min()) - int(h * 0.015))
            y2 = min(h, int(top_rows.max()) + int(h * 0.02))
            candidate = (int(w * 0.08), y1, int(w * 0.84), max(int(h * 0.06), y2 - y1))
            text_boxes.append(candidate)
    text_boxes = _merge_boxes(text_boxes, iou_threshold=0.18)

    if not image_boxes:
        image_boxes = [(int(w * 0.06), int(h * 0.18), int(w * 0.88), int(h * 0.42))]
    if not text_boxes:
        text_boxes = [
            (int(w * 0.08), int(h * 0.06), int(w * 0.84), int(h * 0.09)),
            (int(w * 0.12), int(h * 0.72), int(w * 0.76), int(h * 0.08)),
        ]

    text_suggestions = []
    for idx, (x, y, bw, bh) in enumerate(sorted(text_boxes, key=lambda b: (b[1], b[0]))[:6], start=1):
        x, y, bw, bh = _clamp_box(x, y, bw, bh, w, h)
        text_suggestions.append({
            'id': f'text_{idx}', 'label': f'카피 {idx}', 'type': 'text',
            'x': x, 'y': y, 'w': bw, 'h': bh,
            'font_size': max(22, min(64, int(bh * 0.42))),
            'font_color': '#111111' if y > h * 0.35 else '#FFFFFF',
            'align': 'center', 'default_text': f'카피 {idx} 입력', 'line_spacing': 8,
        })
    image_suggestions = []
    for idx, (x, y, bw, bh) in enumerate(sorted(image_boxes, key=lambda b: (b[1], b[0]))[:6], start=1):
        x, y, bw, bh = _clamp_box(x, y, bw, bh, w, h)
        image_suggestions.append({
            'id': f'image_{idx}', 'label': f'이미지 {idx}', 'type': 'image',
            'x': x, 'y': y, 'w': bw, 'h': bh, 'placeholder_rgb': [235, 235, 235],
        })
    return {'text_boxes': text_suggestions, 'image_slots': image_suggestions}


# ---------- template/render ----------
def render_template(template: Dict, image_map: Dict[str, Image.Image], text_map: Dict[str, str]) -> Image.Image:
    width = int(template.get("canvas_width", DEFAULT_CANVAS_W))
    height = int(template.get("canvas_height", DEFAULT_CANVAS_H))
    bg = template.get("background_color", "#F5F3EF")
    canvas = Image.new("RGB", (width, height), hex_to_rgb(bg))
    draw = ImageDraw.Draw(canvas)

    ref_rel = template.get("reference_image_rel")
    if ref_rel:
        ref_path = BASE_DIR / ref_rel
        if ref_path.exists():
            ref = Image.open(ref_path).convert("RGB")
            ref = fit_crop(ref, width, height)
            alpha = int(float(template.get("reference_overlay_alpha", 0.14)) * 255)
            overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            ref_rgba = ref.convert("RGBA")
            ref_rgba.putalpha(alpha)
            overlay.alpha_composite(ref_rgba)
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)

    for slot in template.get("image_slots", []):
        x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
        sid = slot["id"]
        if sid in image_map and image_map[sid] is not None:
            pasted = fit_crop(image_map[sid].convert("RGB"), w, h)
            canvas.paste(pasted, (x, y))
        else:
            fill = tuple(slot.get("placeholder_rgb", [230, 230, 230]))
            draw.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill, outline=(160, 160, 160), width=2)
            font = get_font(max(18, min(28, w // 8)))
            label = slot.get("label", sid)
            bb = draw.textbbox((0, 0), label, font=font)
            tx = x + max((w - (bb[2] - bb[0])) // 2, 8)
            ty = y + max((h - (bb[3] - bb[1])) // 2, 8)
            draw.text((tx, ty), label, fill=(100, 100, 100), font=font)

    for box in template.get("text_boxes", []):
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        text = text_map.get(box["id"], box.get("default_text", ""))
        font_size = int(box.get("font_size", DEFAULT_FONT_SIZE))
        font_color = box.get("font_color", "#111111")
        align = box.get("align", "center")
        line_spacing = int(box.get("line_spacing", 8))
        font = get_font(font_size)
        lines = wrap_text(draw, text, font, w)
        line_heights = []
        for line in lines:
            bb = draw.textbbox((0, 0), line or "A", font=font)
            line_heights.append(bb[3] - bb[1])
        total_h = sum(line_heights) + max(0, len(lines) - 1) * line_spacing
        cur_y = y + max((h - total_h) // 2, 0)
        for idx, line in enumerate(lines):
            bb = draw.textbbox((0, 0), line, font=font)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
            if align == "left":
                tx = x
            elif align == "right":
                tx = x + w - tw
            else:
                tx = x + max((w - tw) // 2, 0)
            draw.text((tx, cur_y), line, fill=font_color, font=font)
            cur_y += th + line_spacing

    return canvas


def draw_guide_preview(base_img: Image.Image, image_slots: List[Dict], text_boxes: List[Dict], alpha: float = 0.32) -> Image.Image:
    preview = base_img.convert("RGBA")
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    f = get_font(22)
    for slot in image_slots:
        x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, outline=(52, 152, 219, 255), fill=(52, 152, 219, int(255 * alpha)), width=3)
        draw.text((x + 8, y + 8), slot.get("label", slot["id"]), fill=(255, 255, 255, 255), font=f)
    for box in text_boxes:
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, outline=(231, 76, 60, 255), fill=(231, 76, 60, int(255 * alpha)), width=3)
        draw.text((x + 8, y + 8), box.get("label", box["id"]), fill=(255, 255, 255, 255), font=f)
    return Image.alpha_composite(preview, overlay).convert("RGB")


def copy_uploaded_reference(uploaded_file, target_stem: str) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    target = ASSET_DIR / f"{safe_name(target_stem)}{suffix}"
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def template_payload(template_name: str, canvas_w: int, canvas_h: int, bg_color: str, ref_asset_path: Path | None,
                     image_slots: List[Dict], text_boxes: List[Dict], note: str = "") -> Dict:
    return {
        "template_name": template_name,
        "canvas_width": int(canvas_w),
        "canvas_height": int(canvas_h),
        "background_color": bg_color,
        "reference_image_rel": str(ref_asset_path.relative_to(BASE_DIR)) if ref_asset_path else "",
        "reference_overlay_alpha": 0.12,
        "image_slots": image_slots,
        "text_boxes": text_boxes,
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app": APP_TITLE,
    }


# ---------- export ----------
def build_jsx(template: Dict, text_values: Dict[str, str], image_files: Dict[str, str]) -> str:
    width = int(template.get("canvas_width", DEFAULT_CANVAS_W))
    height = int(template.get("canvas_height", DEFAULT_CANVAS_H))
    bg = hex_to_rgb(template.get("background_color", "#F5F3EF"))

    def esc(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "#target photoshop",
        "app.displayDialogs = DialogModes.NO;",
        f'var doc = app.documents.add({width}, {height}, 72, "{esc(template.get("template_name", APP_TITLE))}", NewDocumentMode.RGB, DocumentFill.WHITE);',
        f"doc.selection.select([[0,0],[{width},0],[{width},{height}],[0,{height}]]);",
        f"var bg = new SolidColor(); bg.rgb.red={bg[0]}; bg.rgb.green={bg[1]}; bg.rgb.blue={bg[2]};",
        "doc.selection.fill(bg); doc.selection.deselect();",
    ]

    for slot in template.get("image_slots", []):
        sid = slot["id"].replace("-", "_")
        x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
        file_name = image_files.get(slot["id"], "")
        label = slot.get("label", slot["id"])
        lines += [
            f'var file_{sid} = File("./assets/{esc(file_name)}");',
            f"if (file_{sid}.exists) {{",
            f"  var opened_{sid} = open(file_{sid});",
            f"  opened_{sid}.resizeImage(UnitValue({w},'px'), UnitValue({h},'px'), 72, ResampleMethod.BICUBIC);",
            f"  opened_{sid}.selection.selectAll();",
            f"  opened_{sid}.selection.copy();",
            f"  opened_{sid}.close(SaveOptions.DONOTSAVECHANGES);",
            f"  app.activeDocument = doc;",
            f"  doc.paste();",
            f"  var lyr_{sid} = doc.activeLayer;",
            f'  lyr_{sid}.name = "{esc(label)}";',
            f"  lyr_{sid}.translate({x}, {y});",
            "}",
        ]

    for box in template.get("text_boxes", []):
        tid = box["id"].replace("-", "_")
        text = text_values.get(box["id"], box.get("default_text", ""))
        font_size = int(box.get("font_size", DEFAULT_FONT_SIZE))
        color = hex_to_rgb(box.get("font_color", "#111111"))
        x = int(box["x"])
        y = int(box["y"]) + font_size
        lines += [
            f"var txt_{tid} = doc.artLayers.add();",
            f"txt_{tid}.kind = LayerKind.TEXT;",
            f'txt_{tid}.name = "{esc(box.get("label", box["id"]))}";',
            f'txt_{tid}.textItem.contents = "{esc(text)}";',
            f"txt_{tid}.textItem.position = [{x}, {y}];",
            f"txt_{tid}.textItem.size = {font_size};",
            f"var color_{tid} = new SolidColor();",
            f"color_{tid}.rgb.red = {color[0]}; color_{tid}.rgb.green = {color[1]}; color_{tid}.rgb.blue = {color[2]};",
            f"txt_{tid}.textItem.color = color_{tid};",
        ]

    lines.append("alert('미샵 템플릿 JSX 패키지 생성 완료. 포토샵에서 추가 편집 후 PSD로 저장하세요.');")
    return "\n".join(lines)


def build_export_zip(template: Dict, rendered_img: Image.Image, image_map: Dict[str, Image.Image], text_values: Dict[str, str]) -> Path:
    out_name = safe_name(template.get("template_name", "misharp_template")) + "_package.zip"
    out_path = OUTPUT_DIR / out_name
    asset_bytes = {}
    asset_file_names = {}
    for slot in template.get("image_slots", []):
        sid = slot["id"]
        if image_map.get(sid) is not None:
            fname = f"{sid}.png"
            asset_file_names[sid] = fname
            asset_bytes[fname] = pil_to_bytes(image_map[sid].convert("RGB"), fmt="PNG")

    jsx = build_jsx(template, text_values, asset_file_names)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("output/rendered.jpg", pil_to_bytes(rendered_img.convert("RGB"), fmt="JPEG"))
        zf.writestr("photoshop/misharp_template.jsx", jsx.encode("utf-8"))
        zf.writestr("template/template.json", json.dumps(template, ensure_ascii=False, indent=2).encode("utf-8"))
        zf.writestr("template/text_values.json", json.dumps(text_values, ensure_ascii=False, indent=2).encode("utf-8"))
        for fname, b in asset_bytes.items():
            zf.writestr(f"photoshop/assets/{fname}", b)
        zf.writestr(
            "README.txt",
            (
                f"{APP_TITLE}\n"
                f"{APP_SUBTITLE}\n\n"
                "[구성]\n"
                "1. output/rendered.jpg : 즉시 사용용 JPG\n"
                "2. photoshop/misharp_template.jsx : 포토샵 실행 스크립트\n"
                "3. photoshop/assets/* : 포토샵에서 불러올 이미지 자산\n"
                "4. template/template.json : 템플릿 원본 구조\n\n"
                "[사용법]\n"
                "1. 포토샵에서 File > Scripts > Browse 로 misharp_template.jsx 선택\n"
                "2. assets 폴더와 jsx 파일이 같은 상위구조에 있도록 압축을 푼 뒤 실행\n"
                "3. 포토샵에서 위치/서체/크기 최종 수정 후 PSD 저장\n\n"
                f"{COPYRIGHT}\n"
            ).encode("utf-8")
        )
    return out_path


# ---------- presets ----------
def preset_blocks(canvas_w: int, canvas_h: int) -> Tuple[List[Dict], List[Dict]]:
    image_slots = [
        {"id": "hero_image", "label": "대표 이미지", "x": int(canvas_w * 0.06), "y": int(canvas_h * 0.18), "w": int(canvas_w * 0.88), "h": int(canvas_h * 0.42), "placeholder_rgb": [235, 235, 235]},
        {"id": "detail_image", "label": "디테일 컷", "x": int(canvas_w * 0.06), "y": int(canvas_h * 0.66), "w": int(canvas_w * 0.42), "h": int(canvas_h * 0.22), "placeholder_rgb": [235, 235, 235]},
        {"id": "faq_image", "label": "FAQ/서브컷", "x": int(canvas_w * 0.52), "y": int(canvas_h * 0.66), "w": int(canvas_w * 0.42), "h": int(canvas_h * 0.22), "placeholder_rgb": [235, 235, 235]},
    ]
    text_boxes = [
        {"id": "hook", "label": "3초 훅", "x": int(canvas_w * 0.08), "y": int(canvas_h * 0.05), "w": int(canvas_w * 0.84), "h": int(canvas_h * 0.08), "font_size": 44, "font_color": "#111111", "align": "center", "default_text": "3초 훅 카피 입력", "line_spacing": 8},
        {"id": "subcopy", "label": "서브 카피", "x": int(canvas_w * 0.12), "y": int(canvas_h * 0.12), "w": int(canvas_w * 0.76), "h": int(canvas_h * 0.05), "font_size": 26, "font_color": "#333333", "align": "center", "default_text": "서브 카피 입력", "line_spacing": 6},
        {"id": "usp", "label": "USP", "x": int(canvas_w * 0.08), "y": int(canvas_h * 0.90), "w": int(canvas_w * 0.84), "h": int(canvas_h * 0.07), "font_size": 24, "font_color": "#111111", "align": "center", "default_text": "USP 3~5줄 입력", "line_spacing": 6},
    ]
    return image_slots, text_boxes


# ---------- session ----------
def init_state():
    defaults = {
        "creator_boxes": [],
        "creator_result_key": None,
        "creator_last_bg": "#F5F3EF",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------- UI helpers ----------
def top_header():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="wide")
    st.markdown(
        f"""
        <style>
        .block-container {{padding-top: 1.2rem; padding-bottom: 1.4rem; max-width: 1400px;}}
        .misharp-title {{font-size: 34px; font-weight: 800; margin-bottom: 6px;}}
        .misharp-sub {{font-size: 15px; color: #666; margin-bottom: 22px;}}
        .misharp-foot {{font-size: 12px; color: #666; text-align:center; margin-top:28px; padding-top:16px; border-top:1px solid #ddd;}}
        .tiny-muted {{font-size: 12px; color: #777;}}
        .stTabs [data-baseweb="tab-list"] {{gap: 8px;}}
        .stTabs [data-baseweb="tab"] {{height: 48px; white-space: pre-wrap; background: #f5f5f5; border-radius: 12px; padding: 8px 18px;}}
        .template-card {{background:#fff; border:1px solid #e9e9e9; border-radius:20px; padding:14px; box-shadow:0 4px 18px rgba(0,0,0,0.04); height:100%;}}
        .template-meta {{font-size:12px; color:#666; margin-top:4px;}}
        .pill {{display:inline-block; background:#f2f2f2; border-radius:999px; padding:4px 10px; font-size:11px; margin-right:6px; margin-top:6px;}}
        </style>
        <div class="misharp-title">{APP_TITLE}</div>
        <div class="misharp-sub">{APP_SUBTITLE}</div>
        """,
        unsafe_allow_html=True,
    )


def info_card(text: str):
    st.info(text)


# ---------- app ----------
def run_creator_tab():
    st.subheader("템플릿 생성")
    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        template_name = st.text_input("템플릿 이름", value=f"미샵_템플릿_{datetime.now().strftime('%m%d_%H%M')}")
        bg_color = st.color_picker("배경색", value=st.session_state.get("creator_last_bg", "#F5F3EF"))
        st.session_state["creator_last_bg"] = bg_color
        use_preset = st.checkbox("미샵 기본 블록 프리셋으로 시작", value=True)
        uploaded_ref = st.file_uploader("참조 JPG/PNG 업로드", type=["jpg", "jpeg", "png", "webp"], key="creator_ref")
        ex_files = list_example_images()
        example_names = [p.name for p in ex_files]
        picked_example = st.selectbox("또는 예시 디자인 선택", ["선택안함"] + example_names, index=0)
        use_auto = st.checkbox("AI로 이미지/카피 영역 자동 추천", value=True)
        if not CV2_OK:
            st.caption("현재 OpenCV 없이 실행 중입니다. 앱은 정상 동작하며, AI 자동 추천은 안전한 기본 분석 모드로 작동합니다.")
        note = st.text_area("템플릿 메모", height=120, placeholder="예: 상단형, 대표컷 중심, 3초 훅 강조")

        if uploaded_ref is not None:
            pil_ref = Image.open(uploaded_ref).convert("RGB")
            ref_source_name = Path(uploaded_ref.name).stem
            ref_asset = copy_uploaded_reference(uploaded_ref, ref_source_name)
        elif picked_example != "선택안함":
            ref_path = EXAMPLE_DIR / picked_example
            pil_ref = Image.open(ref_path).convert("RGB")
            ref_asset = ref_path
        else:
            pil_ref = Image.new("RGB", (DEFAULT_CANVAS_W, DEFAULT_CANVAS_H), hex_to_rgb(bg_color))
            ref_asset = None

        canvas_w, canvas_h = pil_ref.size
        st.caption(f"기준 캔버스: {canvas_w} × {canvas_h}px")

        if st.button("AI 추천 영역 불러오기", type="primary", use_container_width=True):
            image_slots, text_boxes = (preset_blocks(canvas_w, canvas_h) if use_preset else ([], []))
            if use_auto and pil_ref is not None:
                ai = suggest_regions(pil_ref)
                # merge preset + ai suggestions without exact duplicates
                existing = {(b["x"], b["y"], b["w"], b["h"]) for b in image_slots + text_boxes}
                for slot in ai["image_slots"]:
                    key = (slot["x"], slot["y"], slot["w"], slot["h"])
                    if key not in existing:
                        image_slots.append(slot)
                        existing.add(key)
                for box in ai["text_boxes"]:
                    key = (box["x"], box["y"], box["w"], box["h"])
                    if key not in existing:
                        text_boxes.append(box)
                        existing.add(key)
            st.session_state["creator_boxes"] = image_slots + text_boxes
            st.success("AI 추천 영역을 불러왔습니다. 필요하면 아래에서 직접 박스를 추가하세요.")

        if st.button("미샵 기본 블록만 불러오기", use_container_width=True):
            image_slots, text_boxes = preset_blocks(canvas_w, canvas_h)
            st.session_state["creator_boxes"] = image_slots + text_boxes
            st.success("미샵 기본 블록 프리셋을 불러왔습니다.")

        if st.button("박스 전체 초기화", use_container_width=True):
            st.session_state["creator_boxes"] = []
            st.rerun()

    with right:
        st.markdown("**영역 확인 / 수동 추가**")
        current_boxes = st.session_state.get("creator_boxes", [])
        image_slots = [b for b in current_boxes if b.get("type") == "image"]
        text_boxes = [b for b in current_boxes if b.get("type") == "text"]
        preview = draw_guide_preview(fit_crop(pil_ref, canvas_w, canvas_h), image_slots, text_boxes)
        st.image(preview, use_container_width=True)

        st.markdown("**박스 수동 추가**")
        if CANVAS_OK:
            canvas_result = st_canvas(
                fill_color="rgba(231, 76, 60, 0.18)",
                stroke_width=3,
                stroke_color="#E74C3C",
                background_image=pil_ref,
                update_streamlit=True,
                height=min(canvas_h, 1000),
                width=min(canvas_w, 1000),
                drawing_mode="rect",
                key="creator_draw_canvas",
            )
            new_objects = []
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                for obj in canvas_result.json_data["objects"]:
                    if obj.get("type") == "rect":
                        x = int(obj.get("left", 0))
                        y = int(obj.get("top", 0))
                        w = int(obj.get("width", 0) * obj.get("scaleX", 1))
                        h = int(obj.get("height", 0) * obj.get("scaleY", 1))
                        if w > 10 and h > 10:
                            new_objects.append((x, y, w, h))
            if new_objects:
                obj_pick = st.selectbox("추가할 최신 박스", [f"{i+1}. x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}" for i, b in enumerate(new_objects)], index=len(new_objects)-1)
                obj = new_objects[int(obj_pick.split('.')[0]) - 1]
                add_type = st.radio("박스 타입", ["image", "text"], horizontal=True)
                label = st.text_input("라벨", value="이미지" if add_type == "image" else "카피")
                if st.button("선택 박스 추가", use_container_width=True):
                    x, y, w, h = obj
                    new_box = {
                        "id": f"{add_type}_{uuid.uuid4().hex[:6]}",
                        "label": label,
                        "type": add_type,
                        "x": x, "y": y, "w": w, "h": h,
                    }
                    if add_type == "image":
                        new_box["placeholder_rgb"] = [235, 235, 235]
                    else:
                        new_box.update({
                            "font_size": max(22, min(64, int(h * 0.42))),
                            "font_color": "#111111",
                            "align": "center",
                            "default_text": f"{label} 입력",
                            "line_spacing": 8,
                        })
                    st.session_state["creator_boxes"] = st.session_state.get("creator_boxes", []) + [new_box]
                    st.success("박스를 추가했습니다.")
                    st.rerun()
        else:
            st.warning("drawable canvas 라이브러리가 없어서 드래그 추가는 비활성화되었습니다. requirements 설치 후 사용하세요.")

    st.markdown("---")
    st.markdown("**영역 목록 편집**")
    boxes = st.session_state.get("creator_boxes", [])
    if not boxes:
        st.caption("아직 등록된 박스가 없습니다.")
    else:
        for i, box in enumerate(boxes):
            with st.expander(f"{i+1}. [{box.get('type')}] {box.get('label', box['id'])}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                box["label"] = c1.text_input("라벨", value=box.get("label", ""), key=f"label_{i}")
                box["x"] = c2.number_input("x", min_value=0, value=int(box.get("x", 0)), key=f"x_{i}")
                box["y"] = c3.number_input("y", min_value=0, value=int(box.get("y", 0)), key=f"y_{i}")
                box["w"] = c4.number_input("w", min_value=1, value=int(box.get("w", 100)), key=f"w_{i}")
                c5, c6, c7, c8 = st.columns(4)
                box["h"] = c5.number_input("h", min_value=1, value=int(box.get("h", 60)), key=f"h_{i}")
                if box.get("type") == "text":
                    box["font_size"] = c6.number_input("font size", min_value=10, value=int(box.get("font_size", 32)), key=f"fs_{i}")
                    box["font_color"] = c7.color_picker("font color", value=box.get("font_color", "#111111"), key=f"fc_{i}")
                    box["align"] = c8.selectbox("align", ["left", "center", "right"], index=["left", "center", "right"].index(box.get("align", "center")), key=f"al_{i}")
                    box["default_text"] = st.text_input("기본 문구", value=box.get("default_text", ""), key=f"dt_{i}")
                remove = st.button("이 박스 삭제", key=f"remove_{i}")
                if remove:
                    st.session_state["creator_boxes"] = [b for idx, b in enumerate(st.session_state["creator_boxes"]) if idx != i]
                    st.rerun()

    st.markdown("---")
    if st.button("현재 구조로 템플릿 저장", type="primary", use_container_width=True):
        boxes = st.session_state.get("creator_boxes", [])
        image_slots = [b for b in boxes if b.get("type") == "image"]
        text_boxes = [b for b in boxes if b.get("type") == "text"]
        payload = template_payload(template_name, canvas_w, canvas_h, bg_color, ref_asset, image_slots, text_boxes, note=note)
        saved = save_template(payload)
        st.success(f"템플릿 저장 완료: {saved.name}")


def run_manage_tab():
    st.subheader("템플릿 관리")
    files = list_templates()
    if not files:
        st.caption("저장된 템플릿이 없습니다.")
        return

    cols = st.columns(3, gap="large")
    for idx, path in enumerate(files):
        data = load_template(path)
        col = cols[idx % 3]
        with col:
            preview_img = None
            ref_rel = data.get('reference_image_rel')
            if ref_rel and (BASE_DIR / ref_rel).exists():
                try:
                    ref = Image.open(BASE_DIR / ref_rel).convert('RGB')
                    preview_img = draw_guide_preview(
                        fit_crop(ref, data.get('canvas_width', DEFAULT_CANVAS_W), data.get('canvas_height', DEFAULT_CANVAS_H)),
                        data.get('image_slots', []),
                        data.get('text_boxes', []),
                        alpha=0.22,
                    )
                except Exception:
                    preview_img = None
            if preview_img is None:
                preview_img = Image.new('RGB', (900, 1200), hex_to_rgb(data.get('background_color', '#F5F3EF')))

            st.markdown('<div class="template-card">', unsafe_allow_html=True)
            st.image(preview_img, use_container_width=True)
            st.markdown(f"**{data.get('template_name', path.stem)}**")
            st.markdown(
                f"<div class='template-meta'>{path.name}<br>{data.get('canvas_width')}×{data.get('canvas_height')} · 이미지 {len(data.get('image_slots', []))}개 · 카피 {len(data.get('text_boxes', []))}개</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='pill'>배경 {data.get('background_color', '#F5F3EF')}</span><span class='pill'>업데이트 {data.get('updated_at', '')[:10]}</span>",
                unsafe_allow_html=True,
            )
            if data.get('note'):
                st.caption(data['note'])
            c1, c2 = st.columns(2)
            if c1.button('복제', key=f'dup_{path.name}', use_container_width=True):
                clone = dict(data)
                clone['template_name'] = f"{data.get('template_name', path.stem)}_복제"
                save_template(clone)
                st.rerun()
            c2.download_button(
                'JSON 다운로드',
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=path.name,
                mime='application/json',
                key=f'dl_{path.name}',
                use_container_width=True,
            )
            if st.button('삭제', key=f'del_{path.name}', use_container_width=True):
                path.unlink(missing_ok=True)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def run_use_tab():
    st.subheader("템플릿 활용 / PSD용 패키지 출력")
    files = list_templates()
    if not files:
        st.caption("먼저 템플릿을 저장해 주세요.")
        return
    selected_name = st.selectbox("템플릿 선택", [p.name for p in files])
    template = load_template(TEMPLATE_DIR / selected_name)

    ref_rel = template.get("reference_image_rel")
    if ref_rel and (BASE_DIR / ref_rel).exists():
        base_preview = Image.open(BASE_DIR / ref_rel).convert("RGB")
    else:
        base_preview = Image.new("RGB", (template.get("canvas_width", DEFAULT_CANVAS_W), template.get("canvas_height", DEFAULT_CANVAS_H)), hex_to_rgb(template.get("background_color", "#F5F3EF")))

    st.caption(f"캔버스 {template.get('canvas_width')}×{template.get('canvas_height')} · 이미지 {len(template.get('image_slots', []))}개 / 카피 {len(template.get('text_boxes', []))}개")

    left, right = st.columns([0.4, 0.6], gap="large")
    image_map: Dict[str, Image.Image] = {}
    text_values: Dict[str, str] = {}

    with left:
        st.markdown("**이미지 입력**")
        for slot in template.get("image_slots", []):
            up = st.file_uploader(f"{slot.get('label', slot['id'])}", type=["jpg", "jpeg", "png", "webp"], key=f"use_{selected_name}_{slot['id']}")
            if up is not None:
                image_map[slot["id"]] = Image.open(up).convert("RGB")

        st.markdown("**카피 입력**")
        for box in template.get("text_boxes", []):
            text_values[box["id"]] = st.text_area(box.get("label", box["id"]), value=box.get("default_text", ""), height=100 if box.get("h", 80) > 120 else 68, key=f"text_{selected_name}_{box['id']}")

    with right:
        rendered = render_template(template, image_map, text_values)
        st.image(rendered, use_container_width=True)
        jpg_bytes = pil_to_bytes(rendered.convert("RGB"), fmt="JPEG")
        st.download_button("JPG 다운로드", data=jpg_bytes, file_name=f"{safe_name(template.get('template_name'))}.jpg", mime="image/jpeg", use_container_width=True)
        package_path = build_export_zip(template, rendered, image_map, text_values)
        st.download_button("PSD용 패키지 ZIP 다운로드", data=package_path.read_bytes(), file_name=package_path.name, mime="application/zip", use_container_width=True)
        st.caption("ZIP 안에는 JPG + JSX + assets + 템플릿 JSON이 함께 포함됩니다.")


def main():
    init_state()
    ensure_examples_loaded()
    top_header()
    tabs = st.tabs(["템플릿 생성", "템플릿 관리", "템플릿 활용"])
    with tabs[0]:
        run_creator_tab()
    with tabs[1]:
        run_manage_tab()
    with tabs[2]:
        run_use_tab()
    st.markdown(f'<div class="misharp-foot">{COPYRIGHT}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
