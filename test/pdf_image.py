import os
import pdf2image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def convert_pdf_to_images(args):
    filename, folder_path, output_folder_path = args
    if filename.endswith(".pdf"):
        # Extract file name without extension
        file_name_without_ext = os.path.splitext(filename)[0]

        # Get the full path of the PDF file
        pdf_path = os.path.join(folder_path, filename)

        # Convert PDF pages to JPEG images
        images = pdf2image.convert_from_path(pdf_path)

        # Loop through each page
        for i, image in enumerate(images):
            # Define the output filename for the JPEG
            output_filename = f"{file_name_without_ext}_page_{i+1}.jpeg"

            # Save the JPEG image
            image.save(os.path.join(output_folder_path, output_filename))

    # Return a result to signify that the task is done
    return 1

if __name__ == "__main__":
    # Define the folder containing the PDF files
    folder_path = "data/pre_pdf"

    # Define the output folder for the converted JPEGs
    output_folder_path = "data/images"

    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Get all files in the directory
    all_files = [(file, folder_path, output_folder_path) for file in os.listdir(folder_path) if file.endswith(".pdf")]

    # Use multiprocessing to process multiple files at once
    with Pool(cpu_count() - 1) as p:
        for _ in tqdm(p.imap_unordered(convert_pdf_to_images, all_files), total=len(all_files)):
            pass

    print("PDF pages converted to JPEG successfully!")
