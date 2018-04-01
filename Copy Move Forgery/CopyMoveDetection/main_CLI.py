
import CopyMoveDetection
import cv2
from PIL import Image
"""
Main Code
"""

# to detect all images under a directory, use detect_dir
# CopyMoveDetection.detect_dir('../testcase_image/', '../testcase_result/', 32)

# to detect single image, use detect
# CopyMoveDetection.detect('../testcase_image/', '01_barrier_copy.png', '../testcase_result/', blockSize=32)

# example
from PIL import Image
img = Image.open('test_images/Test-2.png').convert('LA')
width, height = img.size
img=img.resize((int(width/2),int(height/2)))
img.save('test_images/Test-2.png',optimize=True,quality=95)


CopyMoveDetection.detect('test_images/', 'Test-2.png', 'results/', blockSize=32)
