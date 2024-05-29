# Full Script with Setup Instructions

# 1. Create a Virtual Environment
#    python -m venv pdf_test_env

# 2. Activate the Virtual Environment
#    On Windows:
#    pdf_test_env\Scripts\activate
#    On macOS/Linux:
#    source pdf_test_env/bin/activate

# 3. Install the Required Packages
#    pip install pymupdf pillow numpy

# 4. Save this script as validate_pdf.py and run it
#    python validate_pdf.py

import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np

def extract_images_from_pdf(pdf_path):
    """
    Extracts images from a PDF file and returns them as a list of PIL Image objects.
    """
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    
    return images

def compare_images(img1, img2, threshold=0.60):
    """
    Compares two images and returns True if they are similar enough based on the given threshold.
    """
    img1 = img1.resize((100, 100)).convert("L")  # Resize and convert to grayscale
    img2 = img2.resize((100, 100)).convert("L")
    
    img1_array = np.array(img1).astype(np.float32)
    img2_array = np.array(img2).astype(np.float32)
    
    # Normalize the image arrays
    img1_array /= 255.0
    img2_array /= 255.0
    
    # Calculate the mean squared error between the images
    mse = np.mean((img1_array - img2_array) ** 2)
    
    # Calculate similarity
    similarity = 1 - mse
    
    return similarity >= threshold

def validate_pdf_images(pdf_path, expected_images_paths):
    """
    Validates the images in the PDF against a list of expected images.
    """
    extracted_images = extract_images_from_pdf(pdf_path)
    expected_images = [Image.open(path) for path in expected_images_paths]
    
    if len(extracted_images) != len(expected_images):
        print("Number of extracted images does not match the number of expected images.")
        return False
    
    for i in range(len(extracted_images)):
        if not compare_images(extracted_images[i], expected_images[i]):
            print(f"Image {i+1} does not match the expected image.")
            return False
    
    print("All images match the expected images.")
    return True

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF file and returns it as a list of strings.
    """
    doc = fitz.open(pdf_path)
    text_content = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text_content.append(text)
    
    return text_content

def normalize_text(text):
    """
    Normalizes text by removing extra whitespace and converting to lowercase.
    """
    return ' '.join(text.lower().split())

def validate_partial_text_in_single_page(pdf_path, expected_texts):
    """
    Validates that each expected text snippet is present in the single page of the PDF.
    """
    extracted_texts = extract_text_from_pdf(pdf_path)
    
    if len(extracted_texts) != 1:
        print("PDF does not contain exactly one page.")
        return False
    
    extracted_text_normalized = normalize_text(extracted_texts[0])
    
    for expected_text in expected_texts:
        expected_text_normalized = normalize_text(expected_text)
        if expected_text_normalized not in extracted_text_normalized:
            print(f"Expected text snippet not found in the PDF:")
            print("Extracted text:")
            print(extracted_texts[0])
            print("Expected text snippet:")
            print(expected_text)
            return False
    
    print("All expected text snippets are found in the PDF.")
    return True

# Paths to the PDF file, expected images, and expected text snippets
pdf_path = "jan_klaasen_20240020.pdf"
expected_images_paths = ["image.png"]
expected_texts = [
    "Invoice date: 5/6/2024",
    "Invoice num.: 20240020",
    "Jan klaasen 25G",
    "The Netherlands",
    "€8000.00",
    "VAT: NL 23.80.12.34.B01",
    "info@bsure-digital.n44"
]

# Validate the images in the PDF
validate_pdf_images(pdf_path, expected_images_paths)

# Validate partial text snippets in the single-page PDF
validate_partial_text_in_single_page(pdf_path, expected_texts)
