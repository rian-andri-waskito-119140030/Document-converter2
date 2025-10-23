import cv2
import numpy as np

# 1️⃣ Baca gambar
img = cv2.imread("data_mentah/data_a1.jpg")

# 2️⃣ Ubah jadi grayscale dan threshold inverse
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Threshold agar teks = putih (255), background = hitam (0)
mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 15, 10)

cv2.imshow("Step 1 - Binary Inverse", mask)
cv2.waitKey(500)

# 3️⃣ Repeated Closing (hapus isi dokumen)
kernel = np.ones((5,5), np.uint8)
close = mask.copy()

for i in range(1, 6):  # lakukan 5 kali berturut-turut
    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=3)
    # cv2.imshow(f"Step {i+1} - Closing iteration {i}", close)
    # cv2.waitKey(500)

# 4️⃣ Simpan hasil akhir
cv2.imwrite("data_output4/closed_result.jpg", close)
cv2.destroyAllWindows()
print("Hasil disimpan di data_output/closed_result.jpg")

