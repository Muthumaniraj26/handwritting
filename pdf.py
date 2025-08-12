# === Handwritten Answer Sheet Evaluator (No Java) ===

import os
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from fpdf import FPDF
import re
from tkinter import filedialog, Tk

# === Load OCR Model ===
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# === Utility Functions ===
def convert_pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    image_paths = []
    for i, page in enumerate(pages):
        path = f"page_{i+1}.png"
        page.save(path, 'PNG')
        image_paths.append(path)
    return image_paths

def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def extract_answers_by_question(text):
    lines = text.splitlines()
    answers = {}
    current_q = None
    for line in lines:
        match = re.match(r"^(Q?\s*\d+[a-zA-Z]?[).:]?)", line.strip())
        if match:
            current_q = match.group(1).strip().rstrip(".:)")
            answers[current_q] = line[len(match.group(1)):].strip()
        elif current_q:
            answers[current_q] += ' ' + line.strip()
    return answers

# === Simple Grammar Placeholder ===
def is_grammatically_correct(text):
    # Always return True (skip grammar checking because no Java)
    return True

def score_answer(student_answer, key_words):
    matched = all(word.lower() in student_answer.lower() for word in key_words)
    if matched:
        if is_grammatically_correct(student_answer):
            return 2  # Full marks
        else:
            return 1  # Half mark
    return 0  # No marks

# === Report Generation ===
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Exam Auto-Grading Report", ln=True, align="C")

    def add_student_result(self, question_num, answer, score):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, f"Q{question_num} Answer: {answer}\nScore: {score}\n")

def generate_pdf_report(student_answers, scores, output_path):
    pdf = PDFReport()
    pdf.add_page()
    for q_num in student_answers:
        pdf.add_student_result(q_num, student_answers[q_num], scores[q_num])
    total_score = sum(scores.values())
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Total Score: {total_score}", ln=True)
    pdf.output(output_path)

# === Main Flow ===
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("📄 Select the handwritten answer sheet PDF:")
    answer_pdf_path = filedialog.askopenfilename(title="Select answer sheet PDF")

    image_paths = convert_pdf_to_images(answer_pdf_path)
    full_text = ""
    for img_path in image_paths:
        full_text += extract_text_from_image(img_path) + "\n"

    answers = extract_answers_by_question(full_text)

    # Export extracted answers for reference
    with open("extracted_answers.txt", "w", encoding="utf-8") as f:
        for q, a in answers.items():
            f.write(f"{q}: {a}\n\n")

    print("📝 Answers extracted and saved to 'extracted_answers.txt'.")
    do_correct = input("Do you want to correct it using a keyword sheet? (yes/no): ").strip().lower()

    scores = {}
    if do_correct == "yes":
        print("📄 Select the keyword sheet (TXT format: one question and its keywords per line):")
        keyword_file = filedialog.askopenfilename(title="Select keyword sheet")

        answer_key = {}
        with open(keyword_file, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    q, keywords = parts
                    answer_key[q.strip()] = [k.strip() for k in keywords.split(",") if k.strip()]


        for q in answers:
            if q in answer_key:
                scores[q] = score_answer(answers[q], answer_key[q])
            else:
                scores[q] = 0

        generate_pdf_report(answers, scores, "exam_result.pdf")
        print("✅ Grading complete. Report saved as 'exam_result.pdf'")
    else:
        print("❌ Correction skipped.")
