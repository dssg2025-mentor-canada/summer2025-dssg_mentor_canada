# for OCR
import pathlib
from pathlib import Path

from pdf2image import convert_from_path

# to iterate over multiple PDF files
from glob import glob

# convert pdf to png
def pdf_to_image(pdf_path, output_folder="rag/pdf_images"):
    """
    function to convert pdfs to images
    """
    # make output folder if it DNE
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(pdf_path, output_folder=output_folder, fmt="png")

    return pages

# convert all existing items in the folder
ALL_PDF_PATHS = glob("rag/raw_pdfs/*.pdf")

from pdf2image.exceptions import PDFPageCountError

for pdf_file in ALL_PDF_PATHS:
    # print(f"converting {pdf_file} into .png file")
    # pdf_to_image(pdf_path=pdf_file, output_folder="rag/pdf_images")
    try:
        print(f"Processing: {pdf_file}")
        pdf_to_image(pdf_path=pdf_file, output_folder="rag/pdf_images")
    except PDFPageCountError:
        print(f"❌ Skipping unreadable PDF: {pdf_file}")
    except Exception as e:
        print(f"❌ Unexpected error with {pdf_file}: {e}")

# print(f"Successfully converted {len(ALL_PDF_PATHS)} files :)")
print(ALL_PDF_PATHS)