#!/usr/bin/env python3
"""
color_corners_from_csv.py

Baca file summary_all_corners.csv yang berisi koordinat sudut (TL, TR, BR, BL),
kemudian gambar keempat sudut dengan warna berbeda pada gambar aslinya.

Warna:
- TL (Top Left)  = Merah
- TR (Top Right) = Biru
- BR (Bottom Right) = Hijau
- BL (Bottom Left) = Kuning

Output disimpan ke folder data_output/colored_corners/
"""

import cv2
import csv
import numpy as np
from pathlib import Path

# Lokasi file
CSV_PATH = Path("data_output3/summary_all_corners.csv")
INPUT_DIR = Path("data_mentah2")
OUTPUT_DIR = Path("data_output3/colored_corners")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Warna (BGR)
COLORS = {
    "TL": (0, 0, 255),     # merah
    "TR": (255, 0, 0),     # biru
    "BR": (0, 255, 0),     # hijau
    "BL": (0, 255, 255)    # kuning
}

# Ukuran titik dan tebal garis
RADIUS = 10
THICKNESS = 3

def draw_colored_corners(image_path, corners_dict, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Gagal membaca {image_path.name}")
        return

    h, w = img.shape[:2]
    for label, (x, y) in corners_dict.items():
        if x == "-" or y == "-":
            continue
        try:
            x, y = int(x), int(y)
        except:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue

        color = COLORS.get(label, (255, 255, 255))
        cv2.circle(img, (x, y), RADIUS, color, -1)
        cv2.putText(img, label, (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, THICKNESS, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)
    print(f"✅ Disimpan: {output_path.name}")

def main():
    if not CSV_PATH.exists():
        print("❌ File summary_all_corners.csv tidak ditemukan di data_output/")
        return

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["file"]
            input_path = INPUT_DIR / filename
            if not input_path.exists():
                print(f"⚠️ Gambar {filename} tidak ditemukan di {INPUT_DIR}")
                continue

            # ambil nilai koordinat
            corners = {
                "TL": (row["TL_x"], row["TL_y"]),
                "TR": (row["TR_x"], row["TR_y"]),
                "BR": (row["BR_x"], row["BR_y"]),
                "BL": (row["BL_x"], row["BL_y"])
            }

            out_path = OUTPUT_DIR / f"{Path(filename).stem}_colored.jpg"
            draw_colored_corners(input_path, corners, out_path)

    print("\n🎨 Semua gambar telah diberi warna pada 4 sudut.")
    print(f"Hasil ada di folder: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
