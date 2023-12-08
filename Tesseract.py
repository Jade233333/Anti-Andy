from PIL import Image
import pytesseract

print(pytesseract.image_to_data(Image.open('test/1.png')))
