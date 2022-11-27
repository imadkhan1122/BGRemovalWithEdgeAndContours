import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (255.0, 255.0, 255.0)  # In BGR format
img = cv2.imread("2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)
contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contours = contour_info[0]
mask = np.zeros(edges.shape)
#cv2.fillConvexPoly(mask, max_contour[0], (255))
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
img = img.astype('float32') / 255.0  # for easy blending
masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
masked = (masked * 255).astype('uint8')  # Convert back to 8-bit
plt.imsave('output/1.jpg',masked)
c_red,c_green,c_blue=cv2.split(img)
img_a=cv2.merge((c_red,c_green,c_blue,mask.astype('float32')/255.0))
plt.imshow(img_a)
plt.show()
cv2.imwrite('output/1_1.jpg',img_a*255)
plt.imsave('output/1_2.jpg',img_a)
cv2.imshow('img', masked)  # Display
cv2.waitKey()