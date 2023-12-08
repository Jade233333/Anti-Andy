from Tesseract import *
from annotation import *
from pre_image import *

# Example usage
image_path = 'test/4.png'
preprocessed_image = preprocess_image(image_path)
cv2.imwrite("processed_image.png", preprocessed_image)
pytesseract_image_to_data("processed_image.png")
image_data = pd.read_csv('output.csv')
draw_rectangles_on_blocks('processed_image.png', image_data, 'modified_image.png')
