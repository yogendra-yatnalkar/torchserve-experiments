import cv2
import numpy as np

random_arr = np.zeros((512, 512, 3), dtype = np.uint8)
cv2.imwrite("0.png", random_arr)
