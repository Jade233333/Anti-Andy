import os
import cv2
import numpy as np
import concurrent.futures
from pdf2image import convert_from_path
from tqdm import tqdm

def convert_pdf_to_image(pdf_file, pdf_dir, save_dir):
    # Convert the PDF to images
    images = convert_from_path(os.path.join(pdf_dir, pdf_file))

    # Iterate over every image (i.e., every page of the PDF)
    for i, image in enumerate(images):
        # Skip the first three pages
        if i < 3:
            continue

        # Convert the PIL Image to an OpenCV Image (in BGR format)
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the image
        cv2.imwrite(os.path.join(save_dir, f'{pdf_file}_{i}.png'), cv2_image)

def convert_pdf_to_images_concurrent(pdf_dir, save_dir):
    # Get a list of all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    # Use a ProcessPoolExecutor to parallelize the work
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use list to force execution and tqdm for progress bar
        list(tqdm(executor.map(convert_pdf_to_image, pdf_files, [pdf_dir]*len(pdf_files), [save_dir]*len(pdf_files)), total=len(pdf_files)))

def main():
    # Call the function
    convert_pdf_to_images_concurrent('raw_question_bank/multi/pdf', 'raw_question_bank/multi/images')

if __name__ == '__main__':
    main()
