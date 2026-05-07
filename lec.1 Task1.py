# =========================================================
#Task 1.1
# =========================================================
import cv2, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Print versions
print("OpenCV:  " + cv2.__version__)
print("NumPy:   " + np.__version__)
print("Pillow:  " + Image.__version__)

# Create & display a test image
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[50:250, 100:300] = [0, 194, 203]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Setup Successful!")
plt.show()

# =========================================================
#Task 1.2
# =========================================================
img = cv2.imread('E:\Courses\computer Vision - Innova Bots\FB_IMG_1635710627740.jpg')
if img is None:
    raise FileNotFoundError

# Image properties
shape= img.shape
dtype= img.dtype
total_pixil_count= img.shape[0] * img.shape[1]

# Center pixel coordinates
H, W = img.shape[:2]
cy = H // 2
cx = W // 2

# Exact BGR center pixel
center_pixel = img[cy, cx]

# Print results
print("\n========== IMG DATA ==========")
print('Shape:', shape)
print('Data Type:', dtype)
print('Total Pixels:', total_pixil_count)
print("Center Pixel (BGR):", center_pixel)

# Load 3 different images
img1 = cv2.imread("E:\Courses\computer Vision - Innova Bots\IMG_20211224_194528_022.jpg", 1)
img2 = cv2.imread("E:\Courses\computer Vision - Innova Bots\IMG_20211228_173953_156.jpg", 2)
img3 = cv2.imread("E:\Courses\computer Vision - Innova Bots\IMG_20211228_173954_595.jpg", 3)

# =========================================================
# Task 1.3 - float32 conversion pipeline
# =========================================================
float_img = img1.astype(np.float32)

print("\n========== FLOAT32 CONVERSION ==========")
print("Before Conversion :", img1.dtype)
print("After Conversion  :", float_img.dtype)

# =========================================================
# Task 1.4 - Split BGR channels
# =========================================================
B, G, R = cv2.split(img1)

# Display channels as grayscale subplots
plt.figure(figsize=(12, 4))

# Blue channel
plt.subplot(1, 3, 1)
plt.imshow(B, cmap='gray')
plt.title("Blue Channel")
plt.axis("off")

# Green channel
plt.subplot(1, 3, 2)
plt.imshow(G, cmap='gray')
plt.title("Green Channel")
plt.axis("off")

# Red channel
plt.subplot(1, 3, 3)
plt.imshow(R, cmap='gray')
plt.title("Red Channel")
plt.axis("off")

# =========================================================
# Task 1.5 - Save BEFORE show
# =========================================================
plt.savefig("channels_output.png")

plt.show()
