#!/usr/bin/env python3
"""
auto_detect_corners.py

- Baca semua gambar di folder `data_mentah/`
- Lakukan preprocessing (grayscale, blur, adaptive threshold / Canny)
- Temukan kontur terbesar yang menyerupai kertas
- Coba approxPolyDP -> jika menghasilkan 4 titik, gunakan itu
- Jika tidak, pakai bounding box dari minAreaRect sebagai fallback
- Urutkan titik jadi TL, TR, BR, BL
- Gambar titik + label (x,y) + polygon pada gambar asli
- Simpan hasil ke `data_output/<name>_corners.jpg`

Usage:
    pip install opencv-python numpy
    python auto_detect_corners.py
"""
import cv2
import numpy as np
from pathlib import Path

# ---------- Konfigurasi ----------
INPUT_DIR = Path("data_mentah2")
OUTPUT_DIR = Path("data_output2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canny / preprocessing parameters
USE_ADAPTIVE_THRESH = False   # kalau True gunakan adaptive threshold, else Canny
GAUSSIAN_BLUR_KSIZE = (5,5)
CANNY_LOW = 50
CANNY_HIGH = 150

# approxPolyDP parameter (epsilon ratio)
APPROX_EPS_RATIO = 0.02

# filter area: abaikan kontur yang terlalu kecil (sebagai rasio terhadap image area)
MIN_CONTOUR_AREA_RATIO = 0.05  # 5% luas gambar

# ---------- Fungsi util ----------
def order_points_clockwise(pts):
    """Urutkan 4 titik jadi [TL, TR, BR, BL]"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def largest_quad_from_contours(contours, img_area, min_area_ratio=MIN_CONTOUR_AREA_RATIO):
    """Cari kontur yang setelah approx menjadi 4 titik dan terbesar"""
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_RATIO * peri, True)
        if len(approx) == 4:
            candidates.append((area, approx.reshape(4,2)))
    if not candidates:
        return None
    # pilih area terbesar
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def fallback_min_area_rect(cnt):
    """Jika approx tidak menghasilkan 4 titik, gunakan minAreaRect -> boxPoints"""
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # 4x2 float
    return box.astype("float32")

def detect_corners_in_image(img_bgr):
    """Deteksi sudut; kembalikan coords (4,2) float TL,TR,BR,BL atau None"""
    h, w = img_bgr.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSIZE, 0)

    if USE_ADAPTIVE_THRESH:
        th = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        edges = th
    else:
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    # optional: morphology to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # find contours (external)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges_closed

    # sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # try to find a quad among top contours
    quad = largest_quad_from_contours(contours, img_area)
    if quad is not None:
        ordered = order_points_clockwise(quad)
        return ordered, edges_closed

    # fallback: take the largest contour and approximate using minAreaRect
    largest_cnt = contours[0]
    if cv2.contourArea(largest_cnt) < img_area * MIN_CONTOUR_AREA_RATIO:
        # too small
        return None, edges_closed

    box = fallback_min_area_rect(largest_cnt)
    ordered = order_points_clockwise(box)
    return ordered, edges_closed

def annotate_and_save(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print("Gagal baca:", img_path)
        return

    corners, edges_vis = detect_corners_in_image(img)
    out_img = img.copy()

    base = img_path.stem
    out_file = OUTPUT_DIR / f"{base}_corners.jpg"
    out_edges = OUTPUT_DIR / f"{base}_edges.jpg"

    # save edges visualization for debugging
    cv2.imwrite(str(out_edges), edges_vis)

    if corners is None:
        # beri tanda gagal
        cv2.putText(out_img, "Corners not found", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_file), out_img)
        print(f"{img_path.name}: corners NOT found. Saved edges -> {out_edges.name}, annotated -> {out_file.name}")
        return

    # gambar polygon + titik + label koordinat
    pts = corners.astype(int)
    # polygon
    cv2.polylines(out_img, [pts], isClosed=True, color=(0,255,0), thickness=3)
    labels = ["TL","TR","BR","BL"]
    for i, (x,y) in enumerate(pts):
        cv2.circle(out_img, (x,y), 8, (0,0,255), -1)
        txt = f"{labels[i]} ({x},{y})"
        # teks background
        cv2.putText(out_img, txt, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(out_img, txt, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    # simpan
    cv2.imwrite(str(out_file), out_img)
    print(f"{img_path.name}: corners detected and saved -> {out_file.name}")

# ---------- Main ----------
def main():
    if not INPUT_DIR.exists():
        print("Folder data_mentah/ tidak ditemukan. Buat dan isi gambar di folder tersebut.")
        return

    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tiff")])
    if not files:
        print("Tidak ada gambar di data_mentah/")
        return

    for f in files:
        annotate_and_save(f)

    print("Selesai. Semua gambar diproses. Hasil ada di folder data_output/")

if __name__ == "__main__":
    main()
