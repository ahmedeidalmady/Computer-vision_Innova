# =========================================================
# Task 2.1 
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image safely
image_path = "E:\Courses\computer Vision - Innova Bots\IMG_20211228_173953_156.jpg"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# Image Properties
print("\n========== IMAGE PROPERTIES ==========")
print("Shape :", img.shape)
print("dtype :", img.dtype)
print("ndim  :", img.ndim)
print("size  :", img.size)

H, W = img.shape[:2]
print("Height:", H)
print("Width :", W)

# =========================================================
# Task 2.2 - Access Center Pixel
# =========================================================
cy = H // 2
cx = W // 2

center_pixel = img[cy, cx]

print("\n========== CENTER PIXEL ==========")
print("Center Pixel [B,G,R] :", center_pixel)

print("Blue  :", img[cy, cx, 0])
print("Green :", img[cy, cx, 1])
print("Red   :", img[cy, cx, 2])

# Modify Pixel
modified_img = img.copy()

# Make center pixel pure red
modified_img[cy, cx] = [0, 0, 255]

# =========================================================
# Task 2.3 - Draw Red Square using slicing
# =========================================================
modified_img[100:200, 100:200] = [0, 0, 255]


# =========================================================
# Task 2.4 - Crop ROI (Region Of Interest)
# =========================================================
roi = img[100:300, 200:400]

print("\n========== ROI ==========")
print("ROI Shape :", roi.shape)


# =========================================================
# Task 2.4 - Split Channels
# =========================================================
B, G, R = cv2.split(img)

# =========================================================
# Task 2.5 - Convert to float32
# =========================================================
float_img = img.astype(np.float32)

print("\n========== DATA TYPE CONVERSION ==========")
print("Before :", img.dtype)
print("After  :", float_img.dtype)

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("\n========== GRAYSCALE ==========")
print("Gray Shape :", gray.shape)
print("Gray ndim  :", gray.ndim)


# =========================================================
# Task 2.6 - Save Images
# =========================================================
cv2.imwrite("roi.jpg", roi)
cv2.imwrite("modified_image.jpg", modified_img)
cv2.imwrite("gray.jpg", gray)


# Display Results using Matplotlib
# Convert BGR -> RGB for matplotlib
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 8))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(rgb)
plt.title("Original Image")
plt.axis("off")

# ROI
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.axis("off")

# Grayscale
plt.subplot(2, 3, 3)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")

# Blue channel
plt.subplot(2, 3, 4)
plt.imshow(B, cmap='gray')
plt.title("Blue Channel")
plt.axis("off")

# Green channel
plt.subplot(2, 3, 5)
plt.imshow(G, cmap='gray')
plt.title("Green Channel")
plt.axis("off")

# Red channel
plt.subplot(2, 3, 6)
plt.imshow(R, cmap='gray')
plt.title("Red Channel")
plt.axis("off")


# Save Figure BEFORE show
plt.savefig("task2_output.png")
plt.show()