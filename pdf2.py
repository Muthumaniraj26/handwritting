import os
import re
import sys
import numpy as np
from PIL import Image, ImageOps
from fpdf import FPDF
try:
    from tkinter import filedialog, Tk, messagebox
except ImportError:
    print("Tkinter not available. Please install it for your system.")
    sys.exit(1)
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class PDFAnswerEvaluator:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("trocr-handwriting-model")
        self.model = VisionEncoderDecoderModel.from_pretrained("trocr-handwriting-processor")

        self.style = {
            'header_color': (57, 88, 124),
            'correct_color': (0, 128, 0),
            'partial_color': (200, 150, 0),
            'incorrect_color': (255, 0, 0)
        }

        self.poppler_path = self._detect_poppler_path()

    def _detect_poppler_path(self):
        possible_paths = []
        if sys.platform == 'win32':
            possible_paths.extend([
                r"C:\\Program Files\\poppler-24.08.0\\Library\\bin",
                r"C:\\Program Files\\poppler-24.08.0\\bin",
                r"C:\\Program Files\\poppler\\bin",
                os.path.join(os.environ.get('PROGRAMFILES', ''), "poppler", "bin")
            ])
        else:
            possible_paths.extend([
                "/usr/local/bin",
                "/usr/bin",
                "/opt/local/bin",
                "/opt/homebrew/bin"
            ])

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _check_poppler_installed(self):
        if self.poppler_path is None:
            messagebox.showerror("Poppler Required", "Poppler utilities not found. Please install Poppler.")
            return False
        return True

    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if len(np.array(img).shape) == 2:
                img = Image.merge('RGB', (img, img, img))
            img = ImageOps.autocontrast(img)
            img = ImageOps.grayscale(img)
            img = img.convert('RGB')
            enhanced_path = f"enhanced_{os.path.basename(image_path)}"
            img.save(enhanced_path, quality=95)
            return enhanced_path
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")

    def extract_text_from_image(self, image_path):
        try:
            enhanced_img = self.preprocess_image(image_path)
            image = Image.open(enhanced_img)
            if len(np.array(image).shape) != 3:
                image = image.convert('RGB')
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            os.remove(enhanced_img)
            return text
        except Exception as e:
            if 'enhanced_img' in locals() and os.path.exists(enhanced_img):
                os.remove(enhanced_img)
            raise Exception(f"Text extraction failed: {str(e)}")

    def process_pdf_to_text(self, pdf_path):
        if not self._check_poppler_installed():
            raise Exception("Poppler utilities not available")
        if not os.path.exists(pdf_path):
            raise Exception("PDF file not found")

        convert_kwargs = {'pdf_path': pdf_path, 'dpi': 300}
        if self.poppler_path:
            convert_kwargs['poppler_path'] = self.poppler_path

        images = convert_from_path(**convert_kwargs)
        extracted_data = {}
        temp_files = []

        for i, img in enumerate(images):
            img_path = f"page_{i+1}.png"
            img.save(img_path, "PNG", quality=100)
            text = self.extract_text_from_image(img_path)
            extracted_data[f"Page {i+1}"] = {'text': text, 'image_path': img_path}
            temp_files.append(img_path)

        return extracted_data, temp_files

    def extract_answers(self, text_data):
        answers = {}
        question_pattern = re.compile(r'^(?:(?:Q|Question)\s*)?(\d+)[\)\.\:\-\s]*(.*)', re.IGNORECASE | re.MULTILINE)
        for page, data in text_data.items():
            text = data['text']
            current_q = None
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = question_pattern.match(line)
                if match:
                    current_q = f"Q{match.group(1)}"
                    answer = match.group(2).strip()
                    if answer:
                        answers[current_q] = answer
                elif current_q:
                    answers[current_q] += ' ' + line
        return answers

    def extract_answer_key(self, pdf_path):
        text_data, image_paths = self.process_pdf_to_text(pdf_path)
        answer_key = {}
        current_q = None
        for page, data in text_data.items():
            for line in data['text'].split('\n'):
                line = line.strip()
                if not line:
                    continue
                q_match = re.match(r'^(?:(?:Q|Question)\s*)?(\d+)[\)\.\:\-\s]*(.*)', line, re.IGNORECASE)
                if q_match:
                    current_q = f"Q{q_match.group(1)}"
                    content = q_match.group(2).strip()
                    if content:
                        answer_key[current_q] = [kw.strip().lower() for kw in content.split(',') if kw.strip()]
                elif current_q and ':' in line:
                    keywords = [kw.strip().lower() for kw in line.split(':')[1].split(',') if kw.strip()]
                    if current_q in answer_key:
                        answer_key[current_q].extend(keywords)
                    else:
                        answer_key[current_q] = keywords
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        return answer_key

    def evaluate_answers(self, student_answers, answer_key):
        results = {}
        max_score = 2
        for q, student_answer in student_answers.items():
            sa_lower = student_answer.lower()
            feedback = []
            score = 0
            if q in answer_key:
                expected_keywords = answer_key[q]
                found_keywords = [kw for kw in expected_keywords if kw in sa_lower]
                score = min((len(found_keywords) / len(expected_keywords)) * max_score, max_score)
                score = round(score, 1)
                feedback.append(f"Found {len(found_keywords)}/{len(expected_keywords)} keywords")
                if len(found_keywords) < len(expected_keywords):
                    missing = set(expected_keywords) - set(found_keywords)
                    feedback.append(f"Missing: {', '.join(missing)}")
            else:
                feedback.append("Question not in answer key")
            results[q] = {
                'student_answer': student_answer,
                'score': score,
                'feedback': feedback,
                'max_score': max_score,
                'expected_keywords': answer_key.get(q, [])
            }
        return results

    def generate_report(self, results, student_images, output_path):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        except:
            pdf.set_font('Arial', '', 12)

        self._add_cover_page(pdf)
        self._add_original_answers(pdf, student_images)
        self._add_evaluation_results(pdf, results)
        self._add_summary_section(pdf, results)

        pdf.output(output_path)
        return True

    def _add_cover_page(self, pdf):
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 24)
        pdf.set_text_color(*self.style['header_color'])
        pdf.cell(0, 20, "Answer Sheet Evaluation Report", 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('DejaVu', '', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 10, "This report contains:\n- Original student answers\n- Extracted text from handwriting\n- Comparison with answer key\n- Detailed evaluation for each question")
        pdf.ln(15)

    def _add_original_answers(self, pdf, images):
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 16)
        pdf.set_text_color(*self.style['header_color'])
        pdf.cell(0, 15, "Original Student Answers", 0, 1)
        pdf.ln(5)
        for img_path in images:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    aspect = height / width
                    display_width = 180
                    display_height = display_width * aspect
                    pdf.image(img_path, x=15, w=display_width, h=display_height)
                    pdf.ln(5)
                except:
                    continue

    def _add_evaluation_results(self, pdf, results):
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 16)
        pdf.set_text_color(*self.style['header_color'])
        pdf.cell(0, 15, "Detailed Evaluation", 0, 1)
        pdf.ln(5)
        for q, data in results.items():
            pdf.set_font('DejaVu', 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, f"{q}: {data['score']}/{data['max_score']}", 0, 1, 'L', 1)
            pdf.set_font('DejaVu', '', 10)
            pdf.multi_cell(0, 7, f"Student answer: {data['student_answer']}")
            pdf.set_font('DejaVu', 'I', 10)
            pdf.multi_cell(0, 7, f"Expected keywords: {', '.join(data['expected_keywords'])}")
            if data['score'] == data['max_score']:
                pdf.set_text_color(*self.style['correct_color'])
            elif data['score'] >= data['max_score']/2:
                pdf.set_text_color(*self.style['partial_color'])
            else:
                pdf.set_text_color(*self.style['incorrect_color'])
            for item in data['feedback']:
                pdf.multi_cell(0, 7, f"• {item}")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)

    def _add_summary_section(self, pdf, results):
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 18)
        pdf.set_text_color(*self.style['header_color'])
        pdf.cell(0, 15, "Evaluation Summary", 0, 1, 'C')
        pdf.ln(10)
        total = sum(r['score'] for r in results.values())
        max_total = sum(r['max_score'] for r in results.values())
        if max_total > 0:
            percent = (total / max_total) * 100
            grade = self._calculate_grade(percent)
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(50, 10, "Total Score:", 0, 0)
            pdf.set_font('DejaVu', '', 14)
            pdf.cell(0, 10, f"{total:.1f} / {max_total}", 0, 1)
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(50, 10, "Percentage:", 0, 0)
            pdf.set_font('DejaVu', '', 14)
            pdf.cell(0, 10, f"{percent:.1f}%", 0, 1)
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(50, 10, "Final Grade:", 0, 0)
            pdf.set_font('DejaVu', 'B', 16)
            if grade in ["A+", "A", "B"]:
                pdf.set_text_color(*self.style['correct_color'])
            elif grade in ["C", "D"]:
                pdf.set_text_color(*self.style['partial_color'])
            else:
                pdf.set_text_color(*self.style['incorrect_color'])
            pdf.cell(0, 10, grade, 0, 1)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(10)
            pdf.set_font('DejaVu', '', 12)
            if percent >= 80:
                pdf.multi_cell(0, 8, "Excellent performance! The student has demonstrated mastery of the material.")
            elif percent >= 60:
                pdf.multi_cell(0, 8, "Good work! The student shows understanding of most concepts with some areas for improvement.")
            else:
                pdf.multi_cell(0, 8, "Needs improvement. Recommend reviewing the material and focusing on key concepts.")

    def _calculate_grade(self, percentage):
        if percentage >= 90: return "A+"
        if percentage >= 80: return "A"
        if percentage >= 70: return "B"
        if percentage >= 60: return "C"
        if percentage >= 50: return "D"
        return "F"

def main():
    root = Tk()
    root.withdraw()
    evaluator = PDFAnswerEvaluator()
    try:
        student_pdf = filedialog.askopenfilename(title="Select Student Answer Sheet PDF", filetypes=[("PDF Files", "*.pdf")])
        if not student_pdf:
            messagebox.showinfo("Cancelled", "No student answer sheet selected")
            return
        answer_key_pdf = filedialog.askopenfilename(title="Select Answer Key PDF", filetypes=[("PDF Files", "*.pdf")])
        if not answer_key_pdf:
            messagebox.showinfo("Cancelled", "No answer key selected")
            return
        messagebox.showinfo("Processing", "Extracting and evaluating answers...")
        student_text, student_images = evaluator.process_pdf_to_text(student_pdf)
        student_answers = evaluator.extract_answers(student_text)
        answer_key = evaluator.extract_answer_key(answer_key_pdf)
        results = evaluator.evaluate_answers(student_answers, answer_key)
        report_path = filedialog.asksaveasfilename(title="Save Evaluation Report", defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")], initialfile="Answer_Evaluation_Report.pdf")
        if report_path:
            evaluator.generate_report(results, student_images, report_path)
            messagebox.showinfo("Success", f"Evaluation report saved to:\n{report_path}")
        for img in student_images:
            if os.path.exists(img):
                os.remove(img)
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    finally:
        root.destroy()

if __name__ == "__main__":
    try:
        from tkinter import Tk
        main()
    except ImportError:
        print("Error: Tkinter not found. Please install it.")
        sys.exit(1)
