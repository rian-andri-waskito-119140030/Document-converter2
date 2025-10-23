#!/usr/bin/env python3
"""
draw_grid.py

- Membaca semua gambar di folder `data_mentah/`
- Menggambar garis vertikal & horizontal (grid) di setiap gambar
- Menambahkan label koordinat pixel pada sumbu atas (x) dan sumbu kiri (y)
- Menyimpan hasil ke folder `data_output/` sebagai <nama>_grid.jpg

Konfigurasi:
- Ganti NX/NY untuk jumlah garis vertikal/horizontal (bukan termasuk tepi)
- Atau gunakan STEP_PX (jika >0, NX/NY diabaikan)
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INPUT_FOLDER = Path("data_mentah")
OUTPUT_FOLDER = Path("data_output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Opsi grid: jika STEP_PX > 0, gunakan spacing pixel ; jika 0, gunakan NX / NY pembagian
NX = 10      # jumlah kolom pembagian (garis vertikal = NX-1); jika NX=10 => 9 garis vertikal internal
NY = 10      # jumlah baris pembagian (garis horizontal = NY-1)
STEP_PX = 0  # jika >0, gunakan jarak pixel tetap antar garis (mis. 100). kalau 0, pakai NX/NY.

# Gaya gambar
LINE_COLOR = (0, 255, 0)        # warna garis (BGR) -> hijau
LINE_THICK = 1
MAJOR_LINE_COLOR = (0, 200, 200)
MAJOR_LINE_THICK = 2
TEXT_COLOR = (255, 255, 255)    # putih (PIL RGB)
TEXT_BG = (0, 0, 0, 160)        # latar teks semi-transparan (RGBA)
FONT_SIZE = 14
SHOW_CORNERS = True             # tunjukkan koordinat sudut (TL, TR, BR, BL)

def draw_grid_on_image_cv2(img_bgr, nx=NX, ny=NY, step_px=STEP_PX):
    h, w = img_bgr.shape[:2]

    # convert to PIL for nicer text rendering
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    # Tentukan posisi garis
    if step_px and step_px > 0:
        xs = list(range(step_px, w, step_px))
        ys = list(range(step_px, h, step_px))
    else:
        # NX/NY pembagian -> buat garis di bagian equal
        if nx <= 1:
            xs = []
        else:
            xs = [int(i * w / nx) for i in range(1, nx)]
        if ny <= 1:
            ys = []
        else:
            ys = [int(i * h / ny) for i in range(1, ny)]

    # Gambar garis (cv2 untuk performa)
    img_out = img_bgr.copy()
    # garis mayor (tepi gambar)
    cv2.rectangle(img_out, (0,0), (w-1,h-1), MAJOR_LINE_COLOR, 1)

    # draw verticals
    for i, x in enumerate(xs, start=1):
        # thicker line every Nx/5 roughly (visual)
        thick = LINE_THICK
        color = LINE_COLOR
        if (len(xs) >= 4) and (i % max(1, len(xs)//4) == 0):
            thick = MAJOR_LINE_THICK
            color = MAJOR_LINE_COLOR
        cv2.line(img_out, (x,0), (x,h-1), color, thick)

        # label di bagian atas (pakai PIL agar teks rapi)
        label = str(x)
        bbox = draw.textbbox((0,0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = x - tw//2
        ty = 2
        # draw background rectangle
        draw.rectangle([tx-3, ty-1, tx+tw+3, ty+th+1], fill=TEXT_BG)
        draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

    # draw horizontals
    for j, y in enumerate(ys, start=1):
        thick = LINE_THICK
        color = LINE_COLOR
        if (len(ys) >= 4) and (j % max(1, len(ys)//4) == 0):
            thick = MAJOR_LINE_THICK
            color = MAJOR_LINE_COLOR
        cv2.line(img_out, (0,y), (w-1,y), color, thick)

        # label di kiri
        label = str(y)
        bbox = draw.textbbox((0,0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = 2
        ty = y - th//2
        draw.rectangle([tx-3, ty-1, tx+tw+3, ty+th+1], fill=TEXT_BG)
        draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

    # Mark corners if diminta
    if SHOW_CORNERS:
        corners = {
            "TL": (0,0),
            "TR": (w-1, 0),
            "BR": (w-1, h-1),
            "BL": (0, h-1)
        }
        for name, (cx, cy) in corners.items():
            # small circle (cv2)
            cv2.circle(img_out, (cx, cy), 4, (0,0,255), -1)
            # label
            label = f"{name} ({cx},{cy})"
            bbox = draw.textbbox((0,0), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            tx = cx + 6 if cx + 6 + tw < w else cx - tw - 6
            ty = cy + 6 if cy + 6 + th < h else cy - th - 6
            draw.rectangle([tx-3, ty-1, tx+tw+3, ty+th+1], fill=TEXT_BG)
            draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

    # gabungkan PIL text + gambar cv2
    result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    # copy non-text parts from img_out where lines/circles were drawn (we drew lines on img_out)
    # simplify: overlay img_out (lines) on result but keep text from result
    # We will paint the lines from img_out onto result by taking non-background pixels
    line_mask = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    # where line_mask differs significantly from original gray (i.e. lines exist), copy pixels
    # To be robust, copy any non-white/near-white pixel from img_out
    mask = (line_mask < 250)  # True where not almost-white
    result[mask] = img_out[mask]

    return result

def process_folder():
    if not INPUT_FOLDER.exists():
        print("Folder data_mentah tidak ditemukan. Buat folder dan masukkan gambar.")
        return
    images = sorted([p for p in INPUT_FOLDER.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tiff")])
    if not images:
        print("Tidak ada gambar di data_mentah/")
        return
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print("Gagal baca:", p.name)
            continue
        out_img = draw_grid_on_image_cv2(img)
        out_path = OUTPUT_FOLDER / f"{p.stem}_grid.jpg"
        cv2.imwrite(str(out_path), out_img)
        print("Simpan:", out_path)
    print("Selesai. Semua disimpan di", OUTPUT_FOLDER.resolve())

if __name__ == "__main__":
    process_folder()
