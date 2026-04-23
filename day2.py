import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('E:\Courses\computer Vision - Innova Bots\lecture 1\800002023.jpg')
if img is None:
    raise FileNotFoundError('E:\Courses\computer Vision - Innova Bots\lecture 1\800002023.jpg')

print (img.shape)
print (img.dtype)
print (img.ndim)
print (img.size)
H, W = img.shape[:2]

cx, cy = W//2 , H//2
print (img[cy,cx])
print (img[cy,cx,0])
print (img[cy,cx,1])
print (img[cy,cx,2])

cv2.imshow('Dog', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,6))
plt.imshow(rgb)
plt.title(f'Shape: {img.shape}')
plt.axis('off')
plt.show()

B, G, R = cv2.split(img)
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for ax,ch,name in zip(axes,[B,G,R],['B','G','R']):
    ax.imshow(ch, cmap='gray')   # always cmap!
    ax.set_title(name); ax.axis('off')
plt.tight_layout(); plt.show() 