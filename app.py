import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image, ImageColor, ImageDraw, ImageFont

APP_TITLE = "미샵 템플릿 OS"
APP_SUBTITLE = "상세페이지 자동화를 위한 이미지 편집 템플릿 제공"
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
EXAMPLE_DIR = BASE_DIR / "examples"
OUTPUT_DIR = BASE_DIR / "outputs"
for d in [TEMPLATE_DIR, EXAMPLE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PAGE_WIDTH = 1100
DEFAULT_FONT_SIZE = 42


# ---------- utils ----------
def safe_name(text: str) -> str:
    keep = []
    for ch in text.strip():
        if ch.isalnum() or ch in "-_ ":
            keep.append(ch)
        elif ord(ch) > 127:
            keep.append(ch)
    return "".join(keep).strip().replace(" ", "_") or datetime.now().strftime("template_%Y%m%d_%H%M%S")


def list_templates() -> List[Path]:
    return sorted(TEMPLATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_template(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_template(data: Dict) -> Path:
    path = TEMPLATE_DIR / f"{safe_name(data['template_name'])}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    return ImageColor.getrgb(hex_color)


def fit_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
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


def locate_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
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


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    if not text:
        return [""]
    words = text.split(" ")
    lines = []
    current = ""
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
    return final_lines or [text]


def render_template(template: Dict, image_map: Dict[str, Image.Image], text_map: Dict[str, str]) -> Image.Image:
    width = int(template.get("canvas_width", 900))
    height = int(template.get("canvas_height", 900))
    bg = template.get("background_color", "#F5F3EF")
    canvas = Image.new("RGB", (width, height), hex_to_rgb(bg))
    draw = ImageDraw.Draw(canvas)

    ref_path = template.get("reference_image")
    if ref_path and Path(ref_path).exists():
        ref = Image.open(ref_path).convert("RGB")
        ref = fit_crop(ref, width, height)
        overlay_alpha = int(float(template.get("reference_overlay_alpha", 0.18)) * 255)
        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        ref_rgba = ref.convert("RGBA")
        ref_rgba.putalpha(overlay_alpha)
        overlay.alpha_composite(ref_rgba)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)

    for slot in template.get("image_slots", []):
        x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
        slot_id = slot["id"]
        if slot_id in image_map and image_map[slot_id] is not None:
            pasted = fit_crop(image_map[slot_id].convert("RGB"), w, h)
            canvas.paste(pasted, (x, y))
        else:
            fill = tuple(slot.get("placeholder_rgb", [230, 230, 230]))
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=(170, 170, 170), width=2)
            f = get_font(max(16, min(28, w // 10)))
            label = slot.get("label", slot_id)
            bbox = draw.textbbox((0, 0), label, font=f)
            tx = x + max((w - (bbox[2] - bbox[0])) // 2, 8)
            ty = y + max((h - (bbox[3] - bbox[1])) // 2, 8)
            draw.text((tx, ty), label, fill=(110, 110, 110), font=f)

    for box in template.get("text_boxes", []):
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        text = text_map.get(box["id"], box.get("default_text", ""))
        font_size = int(box.get("font_size", DEFAULT_FONT_SIZE))
        color = box.get("font_color", "#FFFFFF")
        align = box.get("align", "center")
        line_spacing = int(box.get("line_spacing", 8))
        font = get_font(font_size)
        lines = []
        for para in text.split("\n"):
            lines.extend(wrap_text(draw, para, font, w))
        if not lines:
            lines = [""]
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
            draw.text((tx, cur_y), line, fill=color, font=font)
            cur_y += th + line_spacing

    return canvas


def image_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    bio = io.BytesIO()
    save_kwargs = {"quality": 96} if fmt.upper() == "JPEG" else {}
    img.save(bio, format=fmt, **save_kwargs)
    return bio.getvalue()


def build_jsx(template: Dict, text_values: Dict[str, str], image_files: Dict[str, str]) -> str:
    width = int(template.get("canvas_width", 900))
    height = int(template.get("canvas_height", 900))
    bg = hex_to_rgb(template.get("background_color", "#F5F3EF"))

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "#target photoshop",
        "app.displayDialogs = DialogModes.NO;",
        f'var doc = app.documents.add({width}, {height}, 72, "{esc(template.get("template_name", "MISHARP Template"))}", NewDocumentMode.RGB, DocumentFill.WHITE);',
        f"doc.selection.select([[0,0],[{width},0],[{width},{height}],[0,{height}]]);",
        f"var bg = new SolidColor(); bg.rgb.red={bg[0]}; bg.rgb.green={bg[1]}; bg.rgb.blue={bg[2]};",
        "doc.selection.fill(bg); doc.selection.deselect();",
    ]

    for slot in template.get("image_slots", []):
        sid = slot["id"]
        x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
        file_name = image_files.get(sid, "")
        label = slot.get("label", sid)
        lines += [
            f'// IMAGE SLOT: {esc(label)}',
            f'var imgFile_{sid.replace("-", "_")} = File("./assets/{esc(file_name)}");',
            f'if (imgFile_{sid.replace("-", "_")}.exists) {{',
            f'  var placedDoc_{sid.replace("-", "_")} = open(imgFile_{sid.replace("-", "_")});',
            f'  placedDoc_{sid.replace("-", "_")}.resizeImage(UnitValue({w},"px"), UnitValue({h},"px"), 72, ResampleMethod.BICUBIC);',
            f'  placedDoc_{sid.replace("-", "_")}.selection.selectAll();',
            f'  placedDoc_{sid.replace("-", "_")}.selection.copy();',
            f'  placedDoc_{sid.replace("-", "_")}.close(SaveOptions.DONOTSAVECHANGES);',
            '  app.activeDocument = doc;',
            '  doc.paste();',
            '  var lyr = doc.activeLayer;',
            f'  lyr.name = "{esc(label)}";',
            f'  lyr.translate({x}, {y});',
            '}',
        ]

    for box in template.get("text_boxes", []):
        tid = box["id"].replace("-", "_")
        text = text_values.get(box["id"], box.get("default_text", ""))
        lines += [
            f'var textLayer_{tid} = doc.artLayers.add();',
            f'textLayer_{tid}.kind = LayerKind.TEXT;',
            f'textLayer_{tid}.name = "{esc(box.get("label", box["id"]))}";',
            f'textLayer_{tid}.textItem.contents = "{esc(text)}";',
            f'textLayer_{tid}.textItem.position = [{int(box["x"])}, {int(box["y"]) + int(box.get("font_size", DEFAULT_FONT_SIZE))}];',
            f'textLayer_{tid}.textItem.size = {int(box.get("font_size", DEFAULT_FONT_SIZE))};',
        ]
    lines += [
        "alert('미샵 템플릿 JSX 생성 완료. 필요 시 포토샵에서 위치/서체를 미세 조정하세요.');"
    ]
    return "\n".join(lines)


def package_outputs(template_name: str, jpg_bytes: bytes, json_bytes: bytes, jsx_str: str, assets: List[Tuple[str, bytes]]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{template_name}.jpg", jpg_bytes)
        zf.writestr(f"{template_name}.json", json_bytes)
        zf.writestr(f"{template_name}.jsx", jsx_str.encode("utf-8"))
        for fname, content in assets:
            zf.writestr(f"assets/{fname}", content)
    return bio.getvalue()


def make_seed_examples():
    if any(EXAMPLE_DIR.iterdir()):
        return
    zip_path = Path("/mnt/data/상페 상단 디자인.zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".jpg"):
                    out = EXAMPLE_DIR / Path(name).name
                    out.write_bytes(zf.read(name))


make_seed_examples()


# ---------- page ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    .misharp-title {font-size: 34px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px;}
    .misharp-sub {font-size: 16px; color: #555; margin-bottom: 18px;}
    .misharp-card {background: #faf8f4; border: 1px solid #ece7dc; border-radius: 18px; padding: 18px 20px;}
    .small-help {color:#666; font-size: 13px;}
    .footer-note {margin-top: 36px; font-size: 12px; color:#666; text-align:center; line-height:1.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="misharp-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="misharp-sub">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.write(
        "참조 상세페이지 이미지를 기준으로 템플릿을 등록하고, 등록된 템플릿에 이미지와 카피를 넣어 JPG와 포토샵용 JSX 패키지로 내보내는 구조입니다."
    )


tab1, tab2, tab3 = st.tabs(["템플릿 생성", "템플릿 활용", "템플릿 모음"])

with tab1:
    st.subheader("템플릿 생성부")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        ref_file = st.file_uploader("참조 JPG 업로드", type=["jpg", "jpeg", "png"], key="ref_upload")
        template_name = st.text_input("템플릿 이름", value="미샵 상단형 템플릿")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            canvas_width = st.number_input("캔버스 가로", min_value=300, max_value=3000, value=900, step=10)
        with cc2:
            canvas_height = st.number_input("캔버스 세로", min_value=300, max_value=5000, value=900, step=10)
        with cc3:
            background_color = st.color_picker("배경색", value="#F2EFE9")
        overlay_alpha = st.slider("참조 이미지 오버레이 강도", 0.0, 0.6, 0.18, 0.01)

        st.markdown("#### 이미지 슬롯 설정")
        image_slot_count = st.number_input("이미지 슬롯 수", 0, 12, 1, 1)
        image_slots = []
        for i in range(image_slot_count):
            with st.expander(f"이미지 슬롯 {i+1}", expanded=(i == 0)):
                a, b, c = st.columns(3)
                label = a.text_input("라벨", value=f"대표이미지 {i+1}", key=f"img_label_{i}")
                x = b.number_input("x", 0, 5000, 90, 1, key=f"img_x_{i}")
                y = c.number_input("y", 0, 5000, 90, 1, key=f"img_y_{i}")
                a2, b2, c2 = st.columns(3)
                w = a2.number_input("w", 10, 5000, 720, 1, key=f"img_w_{i}")
                h = b2.number_input("h", 10, 5000, 720, 1, key=f"img_h_{i}")
                ph_color = c2.color_picker("플레이스홀더 색", value="#E4E4E4", key=f"img_ph_{i}")
                image_slots.append(
                    {
                        "id": f"image_slot_{i+1}",
                        "label": label,
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "placeholder_rgb": list(hex_to_rgb(ph_color)),
                    }
                )

        st.markdown("#### 카피 박스 설정")
        text_box_count = st.number_input("카피 박스 수", 0, 12, 1, 1)
        text_boxes = []
        for i in range(text_box_count):
            with st.expander(f"카피 박스 {i+1}", expanded=(i == 0)):
                a, b, c = st.columns(3)
                label = a.text_input("라벨", value=f"카피 {i+1}", key=f"txt_label_{i}")
                x = b.number_input("x", 0, 5000, 120, 1, key=f"txt_x_{i}")
                y = c.number_input("y", 0, 5000, 700, 1, key=f"txt_y_{i}")
                a2, b2, c2 = st.columns(3)
                w = a2.number_input("w", 10, 5000, 660, 1, key=f"txt_w_{i}")
                h = b2.number_input("h", 10, 5000, 120, 1, key=f"txt_h_{i}")
                font_size = c2.number_input("폰트 크기", 8, 200, 40, 1, key=f"txt_fs_{i}")
                a3, b3, c3 = st.columns(3)
                font_color = a3.color_picker("글자색", value="#FFFFFF", key=f"txt_fc_{i}")
                align = b3.selectbox("정렬", ["left", "center", "right"], index=1, key=f"txt_al_{i}")
                line_spacing = c3.number_input("줄간격", 0, 40, 8, 1, key=f"txt_ls_{i}")
                default_text = st.text_area("기본 문구", value="TITLE COPY", key=f"txt_def_{i}")
                text_boxes.append(
                    {
                        "id": f"text_box_{i+1}",
                        "label": label,
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "font_size": int(font_size),
                        "font_color": font_color,
                        "align": align,
                        "line_spacing": int(line_spacing),
                        "default_text": default_text,
                    }
                )

        save_btn = st.button("템플릿 저장", type="primary", use_container_width=True)

        if save_btn:
            if ref_file is None:
                st.error("참조 이미지를 먼저 업로드해 주세요.")
            else:
                ref_name = f"ref_{safe_name(template_name)}_{ref_file.name}"
                ref_path = EXAMPLE_DIR / ref_name
                ref_path.write_bytes(ref_file.getvalue())
                template = {
                    "template_name": template_name,
                    "canvas_width": int(canvas_width),
                    "canvas_height": int(canvas_height),
                    "background_color": background_color,
                    "reference_overlay_alpha": float(overlay_alpha),
                    "reference_image": str(ref_path),
                    "image_slots": image_slots,
                    "text_boxes": text_boxes,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                save_path = save_template(template)
                st.success(f"템플릿 저장 완료: {save_path.name}")

    with c2:
        st.markdown("#### 미리보기")
        if ref_file is not None:
            ref_path = EXAMPLE_DIR / f"_temp_preview_{ref_file.name}"
            ref_path.write_bytes(ref_file.getvalue())
            preview_template = {
                "template_name": template_name,
                "canvas_width": int(canvas_width),
                "canvas_height": int(canvas_height),
                "background_color": background_color,
                "reference_overlay_alpha": float(overlay_alpha),
                "reference_image": str(ref_path),
                "image_slots": image_slots,
                "text_boxes": text_boxes,
            }
            preview = render_template(preview_template, {}, {})
            st.image(preview, use_container_width=True)
        else:
            st.info("좌측에서 참조 이미지를 올리면 템플릿 프리뷰가 표시됩니다.")

with tab2:
    st.subheader("템플릿 활용부")
    template_files = list_templates()
    if not template_files:
        st.info("먼저 템플릿 생성 탭에서 템플릿을 하나 저장해 주세요.")
    else:
        option_map = {p.stem: p for p in template_files}
        selected_name = st.selectbox("사용할 템플릿 선택", list(option_map.keys()))
        template = load_template(option_map[selected_name])

        left, right = st.columns([0.95, 1.05])
        image_inputs: Dict[str, Image.Image] = {}
        image_payloads: List[Tuple[str, bytes]] = []
        text_values: Dict[str, str] = {}
        image_file_names: Dict[str, str] = {}

        with left:
            st.markdown("#### 이미지 입력")
            for slot in template.get("image_slots", []):
                up = st.file_uploader(
                    f"{slot.get('label', slot['id'])}",
                    type=["jpg", "jpeg", "png"],
                    key=f"use_{selected_name}_{slot['id']}",
                )
                if up is not None:
                    img = Image.open(io.BytesIO(up.getvalue())).convert("RGB")
                    image_inputs[slot["id"]] = img
                    image_payloads.append((up.name, up.getvalue()))
                    image_file_names[slot["id"]] = up.name

            st.markdown("#### 카피 입력")
            for box in template.get("text_boxes", []):
                text_values[box["id"]] = st.text_area(
                    box.get("label", box["id"]),
                    value=box.get("default_text", ""),
                    key=f"use_text_{selected_name}_{box['id']}",
                    height=100,
                )

        with right:
            preview = render_template(template, image_inputs, text_values)
            st.markdown("#### 결과 미리보기")
            st.image(preview, use_container_width=True)

            jpg_bytes = image_to_bytes(preview, "JPEG")
            json_bytes = json.dumps(template, ensure_ascii=False, indent=2).encode("utf-8")
            jsx_text = build_jsx(template, text_values, image_file_names)
            bundle_bytes = package_outputs(
                safe_name(template["template_name"]),
                jpg_bytes,
                json_bytes,
                jsx_text,
                image_payloads,
            )

            cdl1, cdl2, cdl3 = st.columns(3)
            cdl1.download_button(
                "JPG 다운로드",
                data=jpg_bytes,
                file_name=f"{safe_name(template['template_name'])}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
            cdl2.download_button(
                "JSX 다운로드",
                data=jsx_text.encode("utf-8"),
                file_name=f"{safe_name(template['template_name'])}.jsx",
                mime="text/plain",
                use_container_width=True,
            )
            cdl3.download_button(
                "전체 패키지 ZIP",
                data=bundle_bytes,
                file_name=f"{safe_name(template['template_name'])}_package.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.caption("현재 버전은 실무 안정성을 위해 JPG + 포토샵용 JSX 패키지로 제공합니다. 포토샵에서 JSX 실행 후 최종 PSD 저장이 가능합니다.")

with tab3:
    st.subheader("템플릿 모음")
    files = list_templates()
    if not files:
        st.info("저장된 템플릿이 없습니다.")
    else:
        cols = st.columns(3)
        for idx, path in enumerate(files):
            template = load_template(path)
            preview = render_template(template, {}, {})
            with cols[idx % 3]:
                st.image(preview, caption=template.get("template_name", path.stem), use_container_width=True)
                st.caption(f"크기 {template.get('canvas_width')} × {template.get('canvas_height')}")
                st.code(path.name)

st.markdown(
    '<div class="footer-note">made by MISHARP COMPANY, MIYAWA, 2026.<br>이 프로그램은 미샵컴퍼니 내부직원용이며 외부유출 및 무단 사용을 금합니다.</div>',
    unsafe_allow_html=True,
)
