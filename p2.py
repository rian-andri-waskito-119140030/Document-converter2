#!/usr/bin/env python3
"""
detect_corners_robust.py

Robust corner detection with multiple fallback strategies.
Simpan gambar di ./data_mentah/ dan jalankan. Hasil ada di ./data_output/

Outputs per file:
- <name>_annotated.jpg    : annotated image with corners if found (or message)
- <name>_debug_*.jpg      : debug images for each method
- summary_all_corners.csv : CSV summary (TL,TR,BR,BL or "-" if not found)

Dependencies:
    pip install opencv-python numpy
"""
import cv2
import numpy as np
from pathlib import Path
import csv
import math

INPUT_DIR = Path("data_mentah2")
OUTPUT_DIR = Path("data_output3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- parameters (sesuaikan jika perlu) ----------
CANNY_LOW, CANNY_HIGH = 50, 150
MORPH_KERNEL = (7,7)
MIN_AREA_RATIO = 0.02   # kontur minimal relatif terhadap area gambar (2%)
APPROX_EPS_RATIO = 0.01
SUBPIX = True

# Hough params (untuk fallback)
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESH = 80
HOUGH_MINLEN = 60
HOUGH_MAXGAP = 20

# cornerSubPix
SUBPIX_WIN = (5,5)
SUBPIX_ZERO = (-1,-1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def order_points(pts):
    pts = np.array(pts, dtype="float32").reshape(4,2)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def subpix_refine(gray, pts):
    if not SUBPIX:
        return pts
    try:
        pts_for = pts.reshape(-1,1,2).astype(np.float32)
        cv2.cornerSubPix(gray, pts_for, SUBPIX_WIN, SUBPIX_ZERO, SUBPIX_CRITERIA)
        return pts_for.reshape(-1,2)
    except Exception:
        return pts


def save_annotated(img_bgr, corners, out_path):
    vis = img_bgr.copy()
    h,w = vis.shape[:2]
    if corners is None:
        cv2.putText(vis, "Corners NOT found", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    else:
        pts = corners.astype(int).tolist()
        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], True, (0,255,0), 3)
        labels = ["TL","TR","BR","BL"]
        for i,(x,y) in enumerate(pts):
            cv2.circle(vis, (x,y), 6, (0,0,255), -1)
            txt = f"{labels[i]} ({x},{y})"
            cv2.putText(vis, txt, (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


# ---------- Method 1: adaptive threshold -> find largest quad ----------
def method_adaptive_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive using mean
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 10)
    # close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    return thr


def find_quad_from_binary(bin_img, orig_img):
    # find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h,w = bin_img.shape[:2]
    img_area = h*w
    # sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:8]:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_AREA_RATIO:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_RATIO * peri, True)
        if len(approx) == 4:
            quad = order_points(approx)
            # additional aspect ratio or area check optional
            return quad
    return None


# ---------- Method 2: Canny + morph (edges only) ----------
def method_canny_morph(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed


# ---------- Method 3: fill white regions (flood) -> largest white region bounding box ----------
def method_fill_and_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold bright areas
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # dilate to merge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, th
    h,w = th.shape[:2]
    img_area = h*w
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # choose largest white area with reasonable aspect ratio
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_AREA_RATIO:
            continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        # take rectangle corners
        tl = (x,y); tr = (x+ww, y); br = (x+ww, y+hh); bl = (x, y+hh)
        pts = np.array([tl,tr,br,bl], dtype="float32")
        return pts, th
    return None, th


# ---------- Method 4: Hough outer lines (take min/max vertical/horizontal) ----------
def method_hough_outer(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # HoughLinesP
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH,
                            minLineLength=HOUGH_MINLEN, maxLineGap=HOUGH_MAXGAP)
    if lines is None:
        return None, edges
    verts = []
    hors = []
    for l in lines.reshape(-1,4):
        x1,y1,x2,y2 = l
        ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
        if ang > 80 and ang < 100:  # vertical-ish
            verts.append((x1+x2)//2)
        elif ang < 10 or ang > 170: # horizontal-ish
            hors.append((y1+y2)//2)
    if not verts or not hors:
        return None, edges
    left = min(verts); right = max(verts)
    top = min(hors); bottom = max(hors)
    h,w = gray.shape[:2]
    # validate area
    area = (right-left)*(bottom-top)
    if area < 0.02 * (w*h):
        return None, edges
    pts = np.array([[left,top],[right,top],[right,bottom],[left,bottom]], dtype="float32")
    return pts, edges


# ---------- master routine tries methods in order ----------
def detect_corners_robust(img, debug_prefix):
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method A: adaptive threshold -> find quad
    thr_adapt = method_adaptive_threshold(img)
    cv2.imwrite(str(OUTPUT_DIR / f"{debug_prefix}_debug_adapt.jpg"), thr_adapt)
    quad = find_quad_from_binary(thr_adapt, img)
    if quad is not None:
        # refine with subpix
        quad = subpix_refine(gray, quad)
        return quad, "adaptive_threshold"

    # Method B: Canny + morph -> find quad
    edges_closed = method_canny_morph(img)
    cv2.imwrite(str(OUTPUT_DIR / f"{debug_prefix}_debug_edges.jpg"), edges_closed)
    quad = find_quad_from_binary(edges_closed, img)
    if quad is not None:
        quad = subpix_refine(gray, quad)
        return quad, "canny_morph"

    # Method C: fill->bbox
    bbox_pts, thfill = method_fill_and_bbox(img)
    cv2.imwrite(str(OUTPUT_DIR / f"{debug_prefix}_debug_fill.jpg"), thfill)
    if bbox_pts is not None:
        quad = order_points(bbox_pts)
        quad = subpix_refine(gray, quad)
        return quad, "fill_bbox"

    # Method D: Hough outer lines
    pts_hough, hough_edges = method_hough_outer(img)
    cv2.imwrite(str(OUTPUT_DIR / f"{debug_prefix}_debug_hough.jpg"), hough_edges)
    if pts_hough is not None:
        quad = order_points(pts_hough)
        quad = subpix_refine(gray, quad)
        return quad, "hough_outer"

    # Method E: last fallback - take largest contour and minAreaRect
    # try processing edges_closed to get contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > MIN_AREA_RATIO * w * h:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            quad = order_points(box)
            quad = subpix_refine(gray, quad)
            return quad, "minAreaRect_fallback"

    return None, None


def process_all():
    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".tiff",".bmp")])
    if not files:
        print("No images in", INPUT_DIR)
        return

    summary = OUTPUT_DIR / "summary_all_corners.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "method", "TL_x","TL_y","TR_x","TR_y","BR_x","BR_y","BL_x","BL_y"])
        for p in files:
            print("Processing:", p.name)
            img = cv2.imread(str(p))
            quad, method = detect_corners_robust(img, p.stem)
            out_annot = OUTPUT_DIR / f"{p.stem}_annotated.jpg"
            if quad is None:
                print("  -> corners NOT found for", p.name)
                save_annotated(img, None, out_annot)
                writer.writerow([p.name, "none"] + ["-"]*8)
            else:
                quad_int = quad.astype(int)
                save_annotated(img, quad_int, out_annot)
                flat = quad_int.reshape(-1).tolist()
                print(f"  -> method={method}, corners={flat}")
                writer.writerow([p.name, method] + flat)
    print("Done. Summary CSV at:", summary.resolve())


if __name__ == "__main__":
    process_all()
