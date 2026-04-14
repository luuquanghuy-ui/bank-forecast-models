# Image Reader from Word Document
# Usage: python doc/read_images.py

import os
import zipfile
import shutil
from pathlib import Path

def extract_images_from_docx(docx_path, output_dir="doc/extracted_images"):
    """Extract images from a Word document."""
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(docx_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith("word/media/"):
                img_name = os.path.basename(file)
                img_path = os.path.join(output_dir, img_name)
                with zip_ref.open(file) as source, open(img_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                print(f"Extracted: {img_name}")

    return output_dir

if __name__ == "__main__":
    docx_file = "doc/image_review_template.docx"

    if not os.path.exists(docx_file):
        print(f"Error: {docx_file} not found!")
        print("Put your images in the Word doc and save it as: doc/image_review_template.docx")
    else:
        output = extract_images_from_docx(docx_file)
        print(f"\nImages extracted to: {output}")
        print("Now send me the images to read!")
