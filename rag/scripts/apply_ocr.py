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
    # # make output folder if it DNE
    # if not Path(output_folder).exists():
    #     Path(output_folder).mkdir(parents=True, exist_ok=True)

    # pages = convert_from_path(pdf_path, output_folder=output_folder, fmt="png")

    # return pages
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get base name (e.g., "mentoring_report" from "mentoring_report.pdf")
    base_name = pdf_path.stem
    expected_first_image = output_folder / f"{base_name}_page_1.png"

    if expected_first_image.exists():
        print(f"✅ Skipping {pdf_path.name} (already converted)")
        return 0

    # Convert PDF to list of images (one per page)
    images = convert_from_path(str(pdf_path))

    for i, img in enumerate(images):
        output_file = output_folder / f"{base_name}_page_{i+1}.png"
        img.save(output_file, "PNG")

    print(f"✅ Converted {pdf_path.name} ({len(images)} pages)")
    return len(images)

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

print(f"Successfully converted {len(ALL_PDF_PATHS)} files :)")
