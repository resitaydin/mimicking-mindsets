import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfWriter
import io

def create_searchable_pdf(input_pdf, output_pdf, language='tur'):
    # Convert PDF pages to images
    images = convert_from_path(input_pdf)
    
    # Create a PDF writer
    pdf_writer = PdfWriter()
    
    for page_num, image in enumerate(images, start=1):
        # Perform OCR on each image and get searchable PDF page
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension='pdf', lang=language)
        
        # Append OCR output to the final PDF
        pdf_writer.append(io.BytesIO(pdf_bytes))
        print(f"Processed page {page_num}/{len(images)}")
    
    # Write the final searchable PDF to a file
    with open(output_pdf, "wb") as f:
        pdf_writer.write(f)

# Example usage
input_pdf = 'Erol-Gungor-Islam-Tasavvufunun-Meseleler.pdf'
output_pdf = 'output_searchable.pdf'
create_searchable_pdf(input_pdf, output_pdf)
