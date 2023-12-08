import pandas as pd
from PIL import Image
import pytesseract

def pytesseract_image_to_data(image_path):
    image_data = pytesseract.image_to_data(Image.open(image_path))

    df = pd.DataFrame([line.split('\t') for line in image_data.split('\n')], columns=image_data.split('\n')[0].split('\t')).iloc[1:]

    df.to_csv('output.csv', index=False, header=True)
