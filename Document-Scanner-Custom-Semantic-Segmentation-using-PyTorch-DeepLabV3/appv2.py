import os
import gc
import io
import cv2
import base64
import pathlib
import zipfile
import tempfile
import numpy as np
import mimetypes
import pytesseract 
import re
import shutil
import PIL.ImageEnhance as ImageEnhance
from PIL import ImageFilter
from PyPDF2 import PdfReader
from skimage.filters import threshold_local
from pathlib import Path
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# @st.cache(allow_output_mutation=True)
def load_model(num_classes=2, model_name="mbv3", device=torch.device("cpu")):
    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")

    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    _ = model(torch.randn((1, 3, 384, 384)))

    return model


def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms


def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def scan(image_true=None, trained_model=None, image_size=384, BUFFER=10):
    global preprocess_transforms

    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    with torch.no_grad():
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # ==========================================
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # check if corners are inside.
    # if not find smallest enclosing box, expand_image then extract document
    # else extract document

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        #     box_corners = minimum_bounding_rectangle(corners)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        # new image with additional zeros pixels
        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)

        # adjust original image within the new 'image_extended'
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)

    return final

def enhance_magic_color(cv_img):
    """Magic color: contrast + saturation + local sharpening"""
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    # increase contrast
    pil = ImageEnhance.Contrast(pil).enhance(1.2)
    # increase color
    pil = ImageEnhance.Color(pil).enhance(1.15)
    # slight sharpness
    pil = ImageEnhance.Sharpness(pil).enhance(1.1)
    # optional slight unsharp mask (PIL filter)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def enhance_bw(cv_img):
    """High-contrast B&W for crisp text (binary style). Good for documents with clear text."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # adaptive threshold for uneven lighting
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 10)
    # convert back to BGR for downstream (but OCR can accept gray too)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def enhance_grayscale(cv_img):
    """Smooth grayscale (reduce color noise)"""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # apply histogram equalization to improve contrast
    eq = cv2.equalizeHist(gray)
    # slight denoise
    eq = cv2.fastNlMeansDenoising(eq, None, 10, 7, 21)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def enhance_lighten(cv_img):
    """Lighten: brighten shadows to reveal faint ink (useful for faded or low-light scans)."""
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(1.25)
    pil = ImageEnhance.Contrast(pil).enhance(1.05)
    # slight denoise blur + unsharp
    pil = pil.filter(ImageFilter.MedianFilter(size=3))
    pil = ImageEnhance.Sharpness(pil).enhance(1.05)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def enhance_original(cv_img):
    """Return as-is"""
    return cv_img

# mapping
ENHANCE_MODES = {
    "Magic Color": enhance_magic_color,
    "B&W": enhance_bw,
    "Grayscale": enhance_grayscale,
    "Lighten": enhance_lighten,
    "Original": enhance_original
}

def extract_text_from_image(path: Path) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang="ind+eng")
    return text.strip()

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages[:2]:
            text += page.extract_text() or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        import pdf2image
        pages = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1)
        if pages:
            tmp = np.array(pages[0])
            text = pytesseract.image_to_string(tmp, lang="ind+eng")
            return text.strip()
    except Exception:
        pass
    return ""

# def find_nomor_surat(text: str) -> str:
#     """Cari pola nomor surat umum"""
#     patterns = [
#         r"\bNo\.?\s*[:\-]?\s*([A-Za-z0-9\/\.\-]+)",
#         r"\bNomor\s*[:\-]?\s*([A-Za-z0-9\/\.\-]+)",
#         r"\bNo\s*Surat\s*[:\-]?\s*([A-Za-z0-9\/\.\-]+)",
#         r"\bSurat\s*Nomor\s*[:\-]?\s*([A-Za-z0-9\/\.\-]+)",
#         r"\b([A-Z]{2,5}\d{2,6}\/\d{2,6})\b",
#     ]
#     for p in patterns:
#         m = re.search(p, text, flags=re.IGNORECASE)
#         if m:
#             nomor = re.sub(r"[^\w\-_/\.]", "_", m.group(1))
#             return nomor
#     return ""
def find_nomor_surat(text: str, image_path: Path = None) -> str:
    """
    Robust nomor detection with extra handwriting (blue/ink) pass.
    - image_path: Path to PNG (warped scanned image) to run focused header OCR.
    - returns sanitized nomor or empty string.
    """
    import re, cv2
    import pytesseract
    from pathlib import Path

    def sanitize_nomor(s: str) -> str:
        s = s.strip()
        s = re.sub(r"[^A-Za-z0-9\.\-_/\\]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    # helper: try OCR on provided image region using given config
    def ocr_image(img, config):
        try:
            txt = pytesseract.image_to_string(img, lang='ind+eng', config=config)
            return txt.strip()
        except Exception:
            return ""

    # 1) Header crop word-level approach (same as before)
    if image_path and Path(image_path).exists():
        try:
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            crop = img[int(0.02*h): int(0.24*h), int(0.03*w): int(0.85*w)].copy()
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cfg = r'--oem 3 --psm 6'
            data = pytesseract.image_to_data(th, lang='ind+eng', config=cfg, output_type=pytesseract.Output.DICT)
            # find "nomor" tokens
            import numpy as np
            idxs = [i for i, w in enumerate(data['text']) if w and re.search(r'^(no|no\.|nomor|nom|nr)$', w, flags=re.IGNORECASE)]
            for i in idxs:
                parts = []
                for j in range(i+1, min(i+1+12, len(data['text']))):
                    tok = data['text'][j].strip()
                    if not tok:
                        continue
                    # stop tokens (labels)
                    if re.search(r'^(sifat|lampiran|perihal|kepada|tanggal|kalianda|kalianda,)$', tok, flags=re.IGNORECASE):
                        break
                    parts.append(tok)
                    joined = " ".join(parts)
                    if re.search(r'[\d]+\.[\d]+|\/|[Vv]\.?\d+|[0-9]{3,4}', joined):
                        cand = sanitize_nomor(joined)
                        if re.search(r'\d', cand):
                            return cand
                if parts:
                    cand = sanitize_nomor(" ".join(parts))
                    if re.search(r'\d', cand):
                        return cand
        except Exception:
            pass

        # 1b) Additional: try OCR directly on crop with numeric whitelist (single-line)
        try:
            gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, th2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cfg_num = r'-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./-()[]: --psm 7 --oem 3'
            txt_num = ocr_image(th2, cfg_num)
            if txt_num:
                m = re.search(r'([0-9\.\-/\\\sA-Za-z]{3,80})', txt_num)
                if m:
                    cand = sanitize_nomor(m.group(1))
                    if re.search(r'\d', cand):
                        return cand
        except Exception:
            pass

        # 2) HANDWRITING/INK PASS: isolate colored ink (blue/black) and OCR it
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hchan, schan, vchan = cv2.split(hsv)
            # blue-ish mask (tune ranges if other ink colors appear)
            lower_blue = (90, 40, 30)
            upper_blue = (140, 255, 255)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            # also create dark ink mask (low V)
            mask_dark = cv2.inRange(vchan, 0, 80)

            mask = cv2.bitwise_or(mask_blue, mask_dark)
            # morphology to connect strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.medianBlur(mask, 3)

            # create enhanced image emphasizing ink: bitwise on grayscale
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ink_only = cv2.bitwise_and(gray_crop, gray_crop, mask=mask)
            # amplify contrast
            ink_eq = cv2.equalizeHist(ink_only)
            # threshold to binary
            _, ink_th = cv2.threshold(ink_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR with whitelist single-line (numbers, ., /, letters)
            cfg_hand = r'-c tessedit_char_whitelist=0123456789./-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz() --psm 7 --oem 3'
            txt_hand = ocr_image(ink_th, cfg_hand)
            if txt_hand:
                # join digits and punctuation
                m = re.search(r'([0-9\.\-/\\\sA-Za-z]{2,80})', txt_hand)
                if m:
                    cand = sanitize_nomor(m.group(1))
                    if re.search(r'\d', cand):
                        return cand
        except Exception:
            pass

    # 3) Fallback: regex on full-page OCR text
    if text:
        patterns = [
            r"\bNo\.?\s*[:\-]?\s*([A-Za-z0-9\.\-/\\]+)",
            r"\bNomor\s*[:\-]?\s*([A-Za-z0-9\.\-/\\]+)",
            r"\bNo\s*Surat\s*[:\-]?\s*([A-Za-z0-9\.\-/\\]+)",
            r"\bSurat\s*Nomor\s*[:\-]?\s*([A-Za-z0-9\.\-/\\]+)",
            r"([0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}\.[0-9]+(?:\s*\/\s*[A-Za-z0-9\.\-\\\/]+){1,3})",
            r"([A-Za-z0-9\.\-]+\/[A-Za-z0-9\.\-]+\/\d{4})"
        ]
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                cand = sanitize_nomor(m.group(1))
                if re.search(r'\d', cand):
                    return cand

    return ""


def find_jenis_surat(text: str) -> str:
    """Cari jenis surat (nota dinas, surat undangan, surat tugas, dll)"""
    patterns = {
        "nota_dinas": r"\bnota\s+dinas\b",
        "surat_undangan": r"\bsurat\s+undangan\b",
        "surat_perintah_tugas": r"\bsurat\s+perintah\s+tugas\b",
        "surat_tugas": r"\bsurat\s+tugas\b",
        "surat_edaran": r"\bsurat\s+edaran\b",
        "berita_acara": r"\bberita\s+acara\b",
        "laporan": r"\blaporan\b",
        "surat_pengantar": r"\bsurat\s+pengantar\b"
    }
    for jenis, patt in patterns.items():
        if re.search(patt, text, flags=re.IGNORECASE):
            return jenis
    return "dokumen_umum"

def find_perihal(text: str) -> str:
    """Deteksi perihal atau hal dari isi surat."""
    if not text:
        return ""
    patterns = [
        r"\bPerihal\s*[:\-]?\s*(.+)",
        r"\bHal\s*[:\-]?\s*(.+)",
        r"\bTentang\s*[:\-]?\s*(.+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            # Ambil maksimal 6 kata setelah 'Perihal'
            perihal = m.group(1).strip()
            perihal = re.sub(r"[^A-Za-z0-9\s]", " ", perihal)
            words = perihal.split()
            perihal_short = "_".join(words[:6])
            return perihal_short.lower()
    # fallback: ambil 3 kata pertama isi surat
    words = re.findall(r"[A-Za-z0-9]+", text)
    if words:
        return "_".join(words[:3]).lower()
    return "tanpa_perihal"


def short_title_from_text(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return "_".join(words[:4]) if words else "dokumen"
def sanitize_for_filename(s: str, max_len: int = 80) -> str:
    import re, unicodedata
    if not s:
        return "noname"
    s = unicodedata.normalize("NFKD", s)
    # Ganti slash jadi dash
    s = s.replace("/", "-").replace("\\", "-")
    # Hapus karakter ilegal
    s = re.sub(r'[:*?"<>|]+', "_", s)
    # Hapus karakter aneh lain
    s = re.sub(r"[^A-Za-z0-9\.\-_ ]+", "_", s)
    # Gabungkan underscore berulang
    s = re.sub(r"[\s_]+", "_", s)
    s = s.strip("_.- ")
    if len(s) > max_len:
        s = s[:max_len]
    if not s:
        s = "noname"
    return s
# Generating a link to download a particular image file.
# def get_image_download_link(img, filename, text):
#     buffered = io.BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
#     return href
def get_download_link(data, filename, text, file_type="auto"):
    """
    Membuat tautan download HTML untuk Streamlit.

    Parameter:
    - data: bisa berupa PIL.Image, bytes, atau path file
    - filename: nama file saat diunduh
    - text: teks yang muncul pada tombol/link
    - file_type: 'image', 'pdf', 'zip', atau 'auto'

    Output:
    - HTML anchor tag <a> siap digunakan dengan st.markdown(..., unsafe_allow_html=True)
    """

    # 1️⃣ Jika input berupa gambar (PIL)
    if hasattr(data, "save"):
        buffered = io.BytesIO()
        data.save(buffered, format="PNG")
        data_bytes = buffered.getvalue()
        mime = "image/png"

    # 2️⃣ Jika input berupa path file (PDF/ZIP)
    elif isinstance(data, (str, bytes, io.IOBase)):
        if isinstance(data, str):
            # Baca file dari path
            with open(data, "rb") as f:
                data_bytes = f.read()
            mime = mimetypes.guess_type(data)[0] or "application/octet-stream"
        elif isinstance(data, bytes):
            data_bytes = data
            mime = "application/octet-stream"
        else:
            data_bytes = data.read()
            mime = "application/octet-stream"
    else:
        raise TypeError("data harus berupa PIL.Image, path file, bytes, atau IO stream")

    # 3️⃣ Encode base64
    b64 = base64.b64encode(data_bytes).decode()

    # 4️⃣ Buat link HTML download
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{text}</a>'
    return href

# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Batch Document Scanner — DeepLabV3")

# === Upload multiple files ===
uploaded_files = st.file_uploader(
    "Upload document images (png/jpg/jpeg). You can select many files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)
# Default parameter values
DEFAULTS = {
    "magic_contrast": 1.2,
    "magic_color": 1.15,
    "magic_sharp": 1.1,
    "unsharp_radius": 1.0,
    "unsharp_percent": 120,
    "unsharp_threshold": 3,
    "bw_blocksize": 15,
    "bw_C": 10,
    "gr_denoise_h": 10,
    "light_brightness": 1.25,
    "light_contrast": 1.05,
    "light_sharp": 1.05
}
method = st.radio("Select model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)

col_left, col_right = st.columns((1, 1))

with col_left:
    st.markdown("**Settings**")
    max_preview = st.number_input("Show preview of first N results", min_value=1, max_value=12, value=4)
    mode = st.selectbox(
        "Enhancement mode (CamScanner-like):",
        ("Magic Color", "B&W", "Grayscale", "Lighten", "Original")
    )
    st.markdown("**Enhancement parameters**")
    st.caption("Tweak sliders, then preview (bottom) or run batch (Start processing).")
    # Magic color params
    st.markdown("**Magic Color params**")
    magic_contrast = st.slider("Magic: Contrast", 0.5, 2.0, float(DEFAULTS["magic_contrast"]), 0.05)
    magic_color = st.slider("Magic: Color/Saturation", 0.5, 2.0, float(DEFAULTS["magic_color"]), 0.05)
    magic_sharp = st.slider("Magic: Sharpness", 0.5, 3.0, float(DEFAULTS["magic_sharp"]), 0.05)
    unsharp_radius = st.slider("Unsharp: Radius", 0.5, 3.0, float(DEFAULTS["unsharp_radius"]), 0.1)
    unsharp_percent = st.slider("Unsharp: Percent", 0, 300, int(DEFAULTS["unsharp_percent"]), 5)
    unsharp_threshold = st.slider("Unsharp: Threshold", 0, 10, int(DEFAULTS["unsharp_threshold"]))

    # B&W params
    st.markdown("**B&W params**")
    bw_blocksize = st.slider("B&W: blockSize (odd)", 3, 51, int(DEFAULTS["bw_blocksize"]), 2)
    # ensure odd
    if bw_blocksize % 2 == 0:
        bw_blocksize += 1
    bw_C = st.slider("B&W: C (subtract)", -50, 50, int(DEFAULTS["bw_C"]))

    # Grayscale params
    st.markdown("**Grayscale params**")
    gr_denoise_h = st.slider("Grayscale: denoise strength (h)", 0, 30, int(DEFAULTS["gr_denoise_h"]))

    # Lighten params
    st.markdown("**Lighten params**")
    light_brightness = st.slider("Lighten: Brightness", 0.5, 2.0, float(DEFAULTS["light_brightness"]), 0.05)
    light_contrast = st.slider("Lighten: Contrast", 0.5, 2.0, float(DEFAULTS["light_contrast"]), 0.05)
    light_sharp = st.slider("Lighten: Sharpness", 0.5, 3.0, float(DEFAULTS["light_sharp"]), 0.05)
    process_button = st.button("Start processing")

with col_right:
    st.markdown("**Model**")
    st.caption("Model will be loaded once and reused for all images.")
# Enhancement functions that use parameters above
def enhance_magic_color(cv_img, params):
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Contrast(pil).enhance(params["magic_contrast"])
    pil = ImageEnhance.Color(pil).enhance(params["magic_color"])
    pil = ImageEnhance.Sharpness(pil).enhance(params["magic_sharp"])
    pil = pil.filter(ImageFilter.UnsharpMask(radius=params["unsharp_radius"],
                                             percent=int(params["unsharp_percent"]),
                                             threshold=int(params["unsharp_threshold"])))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def enhance_bw(cv_img, params):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # adaptive threshold uses blockSize and C
    # blocksize must be odd and >=3 (already ensured by slider)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, params["bw_blocksize"], params["bw_C"])
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def enhance_grayscale(cv_img, params):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    eq = cv2.fastNlMeansDenoising(eq, None, params["gr_denoise_h"], 7, 21)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def enhance_lighten(cv_img, params):
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(params["light_brightness"])
    pil = ImageEnhance.Contrast(pil).enhance(params["light_contrast"])
    pil = pil.filter(ImageFilter.MedianFilter(size=3))
    pil = ImageEnhance.Sharpness(pil).enhance(params["light_sharp"])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def enhance_original(cv_img, params):
    return cv_img

# map that uses the param-dict at runtime
def get_enhancer(mode_name, params):
    if mode_name == "Magic Color":
        return lambda img: enhance_magic_color(img, params)
    if mode_name == "B&W":
        return lambda img: enhance_bw(img, params)
    if mode_name == "Grayscale":
        return lambda img: enhance_grayscale(img, params)
    if mode_name == "Lighten":
        return lambda img: enhance_lighten(img, params)
    return lambda img: enhance_original(img, params)

# package current params into a dict for runtime
current_params = {
    "magic_contrast": magic_contrast,
    "magic_color": magic_color,
    "magic_sharp": magic_sharp,
    "unsharp_radius": unsharp_radius,
    "unsharp_percent": unsharp_percent,
    "unsharp_threshold": unsharp_threshold,
    "bw_blocksize": bw_blocksize,
    "bw_C": bw_C,
    "gr_denoise_h": gr_denoise_h,
    "light_brightness": light_brightness,
    "light_contrast": light_contrast,
    "light_sharp": light_sharp
}

ENHANCE_MODES = {
    "Magic Color": get_enhancer("Magic Color", current_params),
    "B&W": get_enhancer("B&W", current_params),
    "Grayscale": get_enhancer("Grayscale", current_params),
    "Lighten": get_enhancer("Lighten", current_params),
    "Original": get_enhancer("Original", current_params)
}

# If files selected: show quick summary
if uploaded_files:
    st.info(f"{len(uploaded_files)} files selected. Click **Start processing** to run.")
    # show filenames (first 20)
    names_preview = [f.name for f in uploaded_files[:20]]
    model_name = "mbv3" if method == "MobilenetV3-Large" else "r50"
    model = load_model(model_name=model_name)  # cached resourc
    st.write("Files (first 20):", names_preview)
    try:
        preview_file = uploaded_files[0]
        file_bytes = np.asarray(bytearray(preview_file.read()), dtype=np.uint8)
        preview_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # run warp (use model if you want; here we'll preview enhancement on the warped result if 'processed' exists)
        # if model is loaded already, do perspective warp; otherwise preview enhancement on raw crop-resized image
        try:
            # try to warp quickly (use scan) — but avoid running heavy model if not loaded
            if 'model' in globals():
                warped = scan(image_true=preview_img, trained_model=model, image_size=IMAGE_SIZE)
            else:
                # simple centered resize for preview only
                warped = cv2.resize(preview_img, (int(preview_img.shape[1]*0.8), int(preview_img.shape[0]*0.8)))
        except Exception:
            warped = cv2.resize(preview_img, (int(preview_img.shape[1]*0.8), int(preview_img.shape[0]*0.8)))
        enhancer_fn = ENHANCE_MODES.get(mode, ENHANCE_MODES["Original"])
        preview_enh = enhancer_fn(warped)
        col_preview_a, col_preview_b = st.columns(2)
        with col_preview_a:
            st.markdown("**Preview: Before (warped)**")
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col_preview_b:
            st.markdown(f"**Preview: After ({mode})**")
            st.image(cv2.cvtColor(preview_enh, cv2.COLOR_BGR2RGB), use_column_width=True)
        # Reset file pointer so main loop can read uploaded_files[0] again
        preview_file.seek(0)
    except Exception as e:
        st.warning(f"Preview generation failed: {e}")

# Process on button click
if process_button and uploaded_files:
    model_name = "mbv3" if method == "MobilenetV3-Large" else "r50"
    model = load_model(model_name=model_name)  # cached resource
    

    # Use temporary directory on disk to avoid high memory use
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        processed_paths = []
        pdf_paths = []
        pil_images_for_pdf = []  # store PIL images for PDF creation
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status_text.text(f"Processing {i}/{total}: {uploaded_file.name}")

            # read file -> OpenCV BGR
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.warning(f"Failed to read {uploaded_file.name}, skipping.")
                progress_bar.progress(int(i/total*100))
                continue

            # process (your actual scan function)
            processed = scan(image_true=img, trained_model=model, image_size=IMAGE_SIZE)
            # convert BGR -> RGB for saving with PIL
            # enhancer = ENHANCE_MODES.get(mode, ENHANCE_MODES["Magic Color"])
            # processed_enh = enhancer(processed)
            processed_enh = enhance_magic_color(processed)
            processed_rgb = cv2.cvtColor(processed_enh, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(processed_rgb)

            # safe filename
            safe_name = Path(uploaded_file.name).stem
            out_name = tmpdir / f"{safe_name}_scanned.png"
            # pil_img.save(out_name, format="PNG")
            # processed_paths.append(out_name)
            # convert BGR->RGB for PIL/PNG saving
            pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            pil_img.save(out_name, format="PNG")
            processed_paths.append(out_name)

            # 4) save per-image PDF (one PDF per page)
            out_pdf = tmpdir / f"{safe_name}_scanned.pdf"
            pil_img.convert("RGB").save(out_pdf, "PDF", resolution=300.0)
            pdf_paths.append(out_pdf)

            progress_bar.progress(int(i/total*100))

        status_text.text("Creating ZIP archive...")

        # create zip on disk
        # zip_path = tmpdir / "scanned_results.zip"
        # with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        #     for p in processed_paths:
        #         # store without tmpdir prefix: just the filename
        #         zf.write(p, arcname=p.name)

        # status_text.text("Done ✅")
        # show images input
        status_text.text("Creating individual PDFs and ZIP archive...")
        # Create zip: all (PNG + PDFs)
        zip_all_path = tmpdir / "scanned_all_results.zip"
        with zipfile.ZipFile(zip_all_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in processed_paths + pdf_paths:
                zf.write(p, arcname=p.name)

        # Create zip: PDFs only
        zip_pdf_path = tmpdir / "scanned_pdfs_only.zip"
        with zipfile.ZipFile(zip_pdf_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in pdf_paths:
                zf.write(p, arcname=p.name)

        # Create zip: Images only
        zip_img_path = tmpdir / "scanned_images_only.zip"
        with zipfile.ZipFile(zip_img_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in processed_paths:
                zf.write(p, arcname=p.name)

        status_text.text("✅ Done — auto-color applied and files exported.")
        # Show previews before and after (if many, just first N)
        st.subheader("Preview (Before → After)")
        cols = st.columns(min(max_preview, len(processed_paths)))
        for i, p in enumerate(processed_paths[:max_preview]):
            with cols[i]:
                a, b = st.columns(2)
                with a:
                    st.caption("Before")
                    st.image(uploaded_files[i], use_container_width=True)
                with b:
                    st.caption("After")
                    st.image(str(p), use_container_width=True)
                st.caption(uploaded_files[i].name)

        # st.subheader("Before (first results)")
        # cols = st.columns(min(max_preview, len(processed_paths)))
        # for i, p in enumerate(processed_paths[:max_preview]):
        #     with cols[i]:
        #         # load original uploaded file for preview
        #         orig = Image.open(uploaded_files[i])
        #         st.image(orig, use_container_width=True)
        #         st.caption(uploaded_files[i].name)
        # st.subheader("Preview (first results)")
        # cols = st.columns(min(max_preview, len(processed_paths)))
        # for i, p in enumerate(processed_paths[:max_preview]):
        #     with cols[i]:
        #         st.image(str(p), use_container_width=True)
        #         st.caption(p.name)

        # Provide download button (stream file from disk)
        # with open(zip_path, "rb") as f:
        #     zip_bytes = f.read()
        #     st.download_button(
        #         label=f"Download all processed images ({len(processed_paths)} files) as ZIP",
        #         data=zip_bytes,
        #         file_name="scanned_results.zip",
        #         mime="application/zip"
        #     )
        # === Tombol download ZIP & PDF ===
        # with open(zip_path, "rb") as f:
        #     zip_bytes = f.read()
        #     st.download_button(
        #         label=f"📦 Download all results as ZIP ({len(processed_paths)} files + PDF)",
        #         data=zip_bytes,
        #         file_name="scanned_results.zip",
        #         mime="application/zip"
        #     )

        # if pdf_path.exists():
        #     with open(pdf_path, "rb") as f:
        #         pdf_bytes = f.read()
        #         st.download_button(
        #             label=f"🖨️ Download combined PDF ({len(pil_images_for_pdf)} pages)",
        #             data=pdf_bytes,
        #             file_name="scanned_results.pdf",
        #             mime="application/pdf"
        #         )
        # === TIGA TOMBOL DOWNLOAD: ALL ZIPS ===
        # === Tahap OCR + Struktur Folder ===
        # === Tahap OCR + Struktur Folder dengan Parent Folder, Nomor & Perihal ===
        st.info("🔍 Deteksi teks, nomor, jenis surat, dan perihal...")

        ocr_output_dir = tmpdir / "grouped_docs"
        ocr_output_dir.mkdir(exist_ok=True)
        group_map = {}

        # Counter autonumber per jenis surat
        auto_counter = {}

        def get_autonumber(jenis: str):
            """Berikan nomor urut otomatis untuk jenis surat tanpa nomor."""
            if jenis not in auto_counter:
                auto_counter[jenis] = 1
            else:
                auto_counter[jenis] += 1
            return f"{auto_counter[jenis]:03d}"
        # Pastikan semua path hasil scan masih ada
        processed_paths = [p for p in processed_paths if Path(p).exists()]
        pdf_paths = [p for p in pdf_paths if Path(p).exists()]
        for img_path, pdf_path in zip(processed_paths, pdf_paths):
            # OCR dari gambar & PDF
            text_img = extract_text_from_image(img_path)
            text_pdf = extract_text_from_pdf(pdf_path)
            combined_text = (text_img + "\n" + text_pdf).strip()

            # Deteksi jenis, nomor, dan perihal
            nomor = find_nomor_surat(combined_text, image_path=Path(img_path))
            jenis = find_jenis_surat(combined_text)
            perihal = find_perihal(combined_text)
            nomor_safe = sanitize_for_filename(nomor)
            jenis_safe = sanitize_for_filename(jenis)
            perihal_safe = sanitize_for_filename(perihal)
            # Tentukan parent & subfolder
            if jenis_safe == "dokumen_umum":
                parent_folder = ocr_output_dir / "dokumen_umum"
                if nomor_safe:
                    sub_name_raw = f"{nomor_safe}_{perihal_safe}" if perihal_safe else nomor_safe
                else:
                    sub_name_raw = f"dokumen_{get_autonumber('dokumen_umum')}_{perihal_safe}" if perihal_safe else f"dokumen_{get_autonumber('dokumen_umum')}"
            else:
                parent_folder = ocr_output_dir / jenis_safe
                if nomor_safe:
                    sub_name_raw = f"{jenis_safe}_{nomor_safe}_{perihal_safe}" if perihal_safe else f"{jenis_safe}_{nomor_safe}"
                else:
                    sub_name_raw = f"{jenis_safe}_{get_autonumber(jenis_safe)}_{perihal_safe}" if perihal_safe else f"{jenis_safe}_{get_autonumber(jenis_safe)}"

            # Final sanitize nama folder
            sub_name = sanitize_for_filename(sub_name_raw, max_len=100)
            sub_folder = parent_folder / sub_name
            sub_folder.mkdir(parents=True, exist_ok=True)

            # === Safe copy (tanpa error WinError 3) ===
            def safe_copy(src_path, dst_folder, base_name):
                if not src_path:
                    return None
                src = Path(src_path)
                if not src.exists():
                    print(f"⚠️ File hilang, dilewati: {src}")
                    return None
                dst_folder.mkdir(parents=True, exist_ok=True)
                ext = src.suffix.lower() or ".png"
                base_name = sanitize_for_filename(base_name)
                dst = dst_folder / f"{base_name}{ext}"
                counter = 1
                while dst.exists():
                    dst = dst_folder / f"{base_name}_{counter}{ext}"
                    counter += 1
                try:
                    shutil.copy2(str(src), str(dst))
                    return dst
                except Exception as e:
                    st.error(f"❌ Gagal menyalin {src} -> {dst}: {e}")
                    return None

            # === Simpan file hasil ===
            base_name = sub_name
            saved_files = []
            for src in [img_path, pdf_path]:
                saved = safe_copy(src, sub_folder, base_name)
                if saved:
                    saved_files.append(saved)

            # Tambahkan ke map hasil
            group_map.setdefault(str(sub_folder.relative_to(ocr_output_dir)), []).extend(saved_files)

        # === Buat ZIP hasil akhir ===
        final_zip_path = tmpdir / "scanned_grouped_hierarchy.zip"
        with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in ocr_output_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=str(file.relative_to(ocr_output_dir)))

        # === Tampilkan hasil ===
        st.success(f"✅ Struktur folder lengkap dengan nomor & perihal selesai dibuat ({len(group_map)} subfolder).")

        for folder, files in group_map.items():
            st.markdown(f"📂 **{folder}** — {len(files)} file")
            for f in files[:1]:
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    st.image(str(f), width=240)

        # # === Buat ZIP berisi semua hasil (Images + PDFs) ===
        # zip_all_path = tmpdir / "scanned_all_results.zip"
        # with zipfile.ZipFile(zip_all_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        #     for p in processed_paths + pdf_paths:
        #         zf.write(p, arcname=p.name)

        # # === Buat ZIP hanya PDF ===
        # zip_pdf_path = tmpdir / "scanned_pdfs_only.zip"
        # with zipfile.ZipFile(zip_pdf_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        #     for p in pdf_paths:
        #         zf.write(p, arcname=p.name)

        # # === Buat ZIP hanya gambar (PNG) ===
        # zip_img_path = tmpdir / "scanned_images_only.zip"
        # with zipfile.ZipFile(zip_img_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        #     for p in processed_paths:
        #         zf.write(p, arcname=p.name)

        # === Tiga tombol sejajar ===
        col1, col2, col3, col4 = st.columns(4)

        # # 🔹 1. Download semua hasil (PDF + PNG)
        # with col1:
        #     with open(zip_all_path, "rb") as f:
        #         zip_bytes = f.read()
        #         st.download_button(
        #             label=f"📦 Download All Results (PDF + Image) [{len(pdf_paths)+len(processed_paths)} files]",
        #             data=zip_bytes,
        #             file_name="scanned_all_results.zip",
        #             mime="application/zip",
        #             use_container_width=True
        #         )

        # # 🔹 2. Download semua PDF saja
        # with col2:
        #     with open(zip_pdf_path, "rb") as f:
        #         zip_bytes = f.read()
        #         st.download_button(
        #             label=f"🖨️ Download All PDFs Only ({len(pdf_paths)} files)",
        #             data=zip_bytes,
        #             file_name="scanned_pdfs_only.zip",
        #             mime="application/zip",
        #             use_container_width=True
        #         )

        # # 🔹 3. Download semua gambar saja
        # with col3:
        #     with open(zip_img_path, "rb") as f:
        #         zip_bytes = f.read()
        #         st.download_button(
        #             label=f"🖼️ Download All Images Only ({len(processed_paths)} files)",
        #             data=zip_bytes,
        #             file_name="scanned_images_only.zip",
        #             mime="application/zip",
        #             use_container_width=True
        #         )
        # 🔹 4. Download struktur folder berdasarkan OCR
        # === Tombol Download ZIP Akhir ===
        col4 = st.columns(1)[0]
        with col4:
            with open(final_zip_path, "rb") as f:
                st.download_button(
                    "📦 Download Folder Berdasarkan Jenis, Nomor & Perihal (ZIP)",
                    data=f.read(),
                    file_name="scanned_grouped_hierarchy.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        status_text.text("✅ Processing complete!")

        # Cleanup handled by TemporaryDirectory context
else:
    if not uploaded_files:
        st.info("No files uploaded yet.")


# IMAGE_SIZE = 384
# preprocess_transforms = image_preprocess_transforms()
# image = None
# final = None
# result = None

# st.set_page_config(initial_sidebar_state="collapsed")

# st.title("Document Scanner: Semantic Segmentation using DeepLabV3-PyTorch")

# uploaded_file = st.file_uploader("Upload Document Image :", type=["png", "jpg", "jpeg"])

# method = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)

# col1, col2 = st.columns((6, 5))

# if uploaded_file is not None:

#     # Convert the file to an opencv image.
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)
#     h, w = image.shape[:2]

#     if method == "MobilenetV3-Large":
#         model = load_model(model_name="mbv3")
#     else:
#         model = load_model(model_name="r50")

#     with col1:
#         st.title("Input")
#         st.image(image, channels="BGR", width='stretch')

#     with col2:
#         st.title("Scanned")
#         final = scan(image_true=image, trained_model=model, image_size=IMAGE_SIZE)
#         st.image(final, channels="BGR", width='stretch')

#     if final is not None:
#         # Display link.
#         result = Image.fromarray(final[:, :, ::-1])
#         st.markdown(get_image_download_link(result, "output.png", "Download " + "Output"), unsafe_allow_html=True)
