import os
import io
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import easyocr
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import torch

# Optional OpenCV (used if available for better preprocessing + deskew)
try:
    import cv2
except ImportError:
    cv2 = None


def _deskew_pil(pil_image, max_correction_deg=15):
    """
    Deskew image using OpenCV if available; otherwise returns the image unchanged.
    Conservative limits to avoid over-rotation.
    """
    if cv2 is None:
        return pil_image
    try:
        gray = np.array(pil_image.convert("L"))
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel = np.ones((3, 3), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pil_image

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return pil_image

        rect = cv2.minAreaRect(largest)
        angle = rect[-1]
        if angle < -45:
            angle = angle + 90
        if abs(angle) > max_correction_deg:
            return pil_image

        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(np.array(pil_image), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception:
        return pil_image


def _preprocess_base(pil_image):
    """
    Your original preprocessing: grayscale + contrast.
    """
    gray_image = ImageOps.grayscale(pil_image)
    enhancer = ImageEnhance.Contrast(gray_image)
    processed_image = enhancer.enhance(2.0)
    return np.array(processed_image)


def _preprocess_cv_strong(pil_image):
    """
    Stronger OpenCV pipeline (if available): denoise, adaptive threshold, morphology.
    Falls back to base preprocessing when OpenCV is missing.
    """
    if cv2 is None:
        return _preprocess_base(pil_image)
    img = np.array(pil_image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.fastNlMeansDenoising(img, h=15)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    return img


def _score_easyocr_results(results):
    """
    Score results: prefer higher total confidence; fallback to longer text.
    """
    if not results:
        return 0.0, ""
    texts = []
    total_conf = 0.0
    for item in results:
        try:
            # (bbox, text, conf)
            _, t, c = item
            texts.append(t)
            if c is not None:
                total_conf += float(c)
        except Exception:
            # detail=0 fallback: strings only
            if isinstance(item, str):
                texts.append(item)
    combined = " ".join(texts).strip()
    return (total_conf if total_conf > 0 else len(combined)), combined


def generate_handwriting_pdf(image_path, output_filename="handwritten_text_report.pdf", use_gpu=True):
    """
    Extracts text from a handwritten image using EasyOCR and generates a PDF report.
    Extracted text is printed horizontally with line wrapping and automatic page breaks.
    """
    print("--- Starting Handwriting Recognition Process ---")

    try:
        # 1️⃣ Initialize EasyOCR Reader
        device = 'cuda' if use_gpu else 'cpu'
        print(f"[STEP 1/5] Loading EasyOCR model on {device}...")
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("EasyOCR model loaded successfully.")

        # 2️⃣ Load the image
        print(f"[STEP 2/5] Loading image from: {image_path}")
        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            return

        # 3️⃣ Preprocess image (deskew + resize, then two variants)
        print("[STEP 3/5] Enhancing image for better OCR...")
        img_for_ocr = _deskew_pil(original_image)
        max_dim = 1024
        img_for_ocr.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        processed_base = _preprocess_base(img_for_ocr)
        processed_cv = _preprocess_cv_strong(img_for_ocr)

        # 4️⃣ Extract text using EasyOCR (two-pass; pick best)
        print("[STEP 4/5] Extracting text using EasyOCR...")
        results_base = reader.readtext(processed_base)
        score_base, text_base = _score_easyocr_results(results_base)

        # If OpenCV produced a different array, try it too
        if processed_cv is not None and not np.array_equal(processed_cv, processed_base):
            results_cv = reader.readtext(processed_cv)
            score_cv, text_cv = _score_easyocr_results(results_cv)
            if score_cv > score_base:
                extracted_text = text_cv
            else:
                extracted_text = text_base
        else:
            extracted_text = text_base

        # Horizontal flatten
        if not extracted_text.strip():
            print("No text detected. Try a clearer image or adjust preprocessing.")
        else:
            print("\nExtracted Text:")
            print(extracted_text)

        # 5️⃣ Generate PDF
        print(f"[STEP 5/5] Generating PDF: {output_filename}...")
        c = canvas.Canvas(output_filename, pagesize=letter)
        width, height = letter

        # PDF Header
        def draw_header():
            c.setFont("Helvetica-Bold", 18)
            c.setFillColorRGB(0.17, 0.24, 0.31)
            c.drawCentredString(width / 2.0, height - 0.75 * inch, "Handwriting Analysis Report")
            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.line(0.5 * inch, height - 0.9 * inch, width - 0.5 * inch, height - 0.9 * inch)

        draw_header()

        # Original Image Section
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.22, 0.35, 0.47)
        c.drawString(0.75 * inch, height - 1.25 * inch, "Original Handwritten Image")

        img_buffer = io.BytesIO()
        original_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        reportlab_image = ImageReader(img_buffer)

        img_width, img_height = original_image.size
        aspect = img_height / float(img_width)
        display_width = width - 1.5 * inch
        display_height = display_width * aspect
        c.drawImage(reportlab_image, 0.75 * inch, height - 1.5 * inch - display_height,
                    width=display_width, height=display_height, preserveAspectRatio=True, anchor='n')

        # Extracted Text Section
        text_start_y = height - 1.75 * inch - display_height
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.22, 0.35, 0.47)
        c.drawString(0.75 * inch, text_start_y, "AI-Extracted Text ")

        # Text object with automatic page breaks
        text_object = c.beginText(0.75 * inch, text_start_y - 0.25 * inch)
        text_object.setFont("Helvetica", 11)
        line_height = 14
        text_object.setLeading(line_height)

        bottom_margin = 0.75 * inch
        max_width = width - 1.5 * inch

        words = extracted_text.strip().split()
        current_line = ""
        y_position = text_object.getY()

        for word in words:
            if c.stringWidth(current_line + word + " ", "Helvetica", 11) < max_width:
                current_line += word + " "
            else:
                if y_position - line_height < bottom_margin:
                    c.drawText(text_object)
                    c.showPage()
                    draw_header()
                    text_object = c.beginText(0.75 * inch, height - 1 * inch)
                    text_object.setFont("Helvetica", 11)
                    text_object.setLeading(line_height)
                    y_position = text_object.getY()
                text_object.textLine(current_line.strip())
                y_position -= line_height
                current_line = word + " "

        # Add last line
        if current_line:
            if y_position - line_height < bottom_margin:
                c.drawText(text_object)
                c.showPage()
                draw_header()
                text_object = c.beginText(0.75 * inch, height - 1 * inch)
                text_object.setFont("Helvetica", 11)
                text_object.setLeading(line_height)
            text_object.textLine(current_line.strip())

        c.drawText(text_object)
        c.save()

        print(f"PDF saved successfully as '{output_filename}'.")
        print("--- Process Complete ---")

    except ImportError:
        print("Missing libraries. Install with: pip install easyocr pillow reportlab torch opencv-python")
    except Exception as e:
        print(f"Unexpected error: {e}")


# ==========================
if __name__ == "__main__":
    image_path_input = input("Enter the path to your handwritten image file: ").strip()
    if os.path.exists(image_path_input):
        generate_handwriting_pdf(image_path=image_path_input, use_gpu=torch.cuda.is_available())
    else:
        print("File path does not exist. Please check the path and try again.")