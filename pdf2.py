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
        # Try different models
        try:
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            print("Using microsoft/trocr-base-handwritten")
        except Exception as e:
            print(f"Failed to load microsoft/trocr-base-handwritten: {e}")
            try:
                # Try alternative model
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
                self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
                print("Using microsoft/trocr-small-handwritten")
            except Exception as e2:
                print(f"Failed to load microsoft/trocr-small-handwritten: {e2}")
                try:
                    # Try another alternative
                    self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                    self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                    print("Using microsoft/trocr-base-printed")
                except Exception as e3:
                    print(f"Failed to load microsoft/trocr-base-printed: {e3}")
                    raise Exception("Could not load any TrOCR model")

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
            
            # Save original for debugging
            original_path = f"original_{os.path.basename(image_path)}"
            img.save(original_path)
            print(f"Saved original image: {original_path}")
            
            # Improve preprocessing for better text detection
            # Resize to a reasonable size for the model
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast
            img = ImageOps.autocontrast(img)
            
            # Convert to grayscale and back to RGB for better processing
            img = ImageOps.grayscale(img)
            img = img.convert('RGB')
            
            enhanced_path = f"enhanced_{os.path.basename(image_path)}"
            img.save(enhanced_path, quality=95)
            print(f"Saved enhanced image: {enhanced_path}")
            return enhanced_path
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")

    def extract_text_from_image(self, image_path):
        try:
            enhanced_img = self.preprocess_image(image_path)
            image = Image.open(enhanced_img)
            if len(np.array(image).shape) != 3:
                image = image.convert('RGB')
            
            # Add debugging information
            print(f"Processing image: {image_path}")
            print(f"Image size: {image.size}")
            
            # Try multiple preprocessing approaches
            methods = [
                ("Original", image),
                ("Grayscale", image.convert('L').convert('RGB')),
                ("High Contrast", self._enhance_contrast(image)),
                ("Resized", image.resize((512, 512), Image.Resampling.LANCZOS))
            ]
            
            for method_name, processed_img in methods:
                try:
                    print(f"Trying method: {method_name}")
                    pixel_values = self.processor(processed_img, return_tensors="pt").pixel_values
                    generated_ids = self.model.generate(pixel_values)
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    print(f"Method {method_name} - Extracted text: '{text}'")
                    
                    if text.strip():
                        os.remove(enhanced_img)
                        return text
                except Exception as e:
                    print(f"Method {method_name} failed: {e}")
                    continue
            
            # If all methods fail, return empty string
            print("All text extraction methods failed")
            os.remove(enhanced_img)
            return ""
            
        except Exception as e:
            if 'enhanced_img' in locals() and os.path.exists(enhanced_img):
                os.remove(enhanced_img)
            raise Exception(f"Text extraction failed: {str(e)}")

    def _enhance_contrast(self, image):
        """Enhance image contrast for better text detection"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)
        return enhanced

    def process_pdf_to_text(self, pdf_path):
        if not self._check_poppler_installed():
            raise Exception("Poppler utilities not available")
        if not os.path.exists(pdf_path):
            raise Exception("PDF file not found")

        try:
            convert_kwargs = {'pdf_path': pdf_path, 'dpi': 300}
            if self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path

            images = convert_from_path(**convert_kwargs)
            extracted_data = {}
            temp_files = []

            for i, img in enumerate(images):
                try:
                    img_path = f"page_{i+1}.png"
                    img.save(img_path, "PNG", quality=100)
                    text = self.extract_text_from_image(img_path)
                    extracted_data[f"Page {i+1}"] = {'text': text, 'image_path': img_path}
                    temp_files.append(img_path)
                except Exception as e:
                    # Clean up any temp files created so far
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    raise Exception(f"Failed to process page {i+1}: {str(e)}")

            return extracted_data, temp_files
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")

    def extract_answers(self, text_data):
        answers = {}
        question_pattern = re.compile(r'^(?:(?:Q|Question)\s*)?(\d+)[\)\.\:\-\s]*(.*)', re.IGNORECASE | re.MULTILINE)
        
        print("Extracting answers from text data:")
        for page, data in text_data.items():
            text = data['text']
            print(f"Page {page} text: '{text}'")
            
            # If no text was extracted, try to extract manually
            if not text.strip():
                print(f"No text extracted from {page}, trying manual extraction...")
                # You can add manual extraction logic here
                continue
            
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
                        print(f"Found answer for {current_q}: '{answer}'")
                elif current_q:
                    answers[current_q] += ' ' + line
                    print(f"Added to {current_q}: '{line}'")
        
        print(f"Total answers extracted: {len(answers)}")
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
            # Use the correct path to the font files
            font_dir = os.path.join(os.path.dirname(__file__), 'dejavu-fonts-ttf-2.37', 'ttf')
            pdf.add_font('DejaVu', '', os.path.join(font_dir, 'DejaVuSans.ttf'), uni=True)
            pdf.add_font('DejaVu', 'B', os.path.join(font_dir, 'DejaVuSans-Bold.ttf'), uni=True)
            pdf.add_font('DejaVu', 'I', os.path.join(font_dir, 'DejaVuSans-Oblique.ttf'), uni=True)
        except Exception as e:
            # Fallback to default fonts if DejaVu is not available
            print(f"Warning: Could not load DejaVu fonts: {e}")
            # Use built-in fonts as fallback
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

    def test_model(self):
        """Test the model with a simple image to verify it's working"""
        try:
            # Create a simple test image with text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a white image
            img = Image.new('RGB', (400, 100), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font, or create text manually
            try:
                # Try to use a system font
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Draw some text
            draw.text((10, 10), "Test 123", fill='black', font=font)
            
            # Save test image
            test_path = "test_image.png"
            img.save(test_path)
            
            # Try to extract text
            text = self.extract_text_from_image(test_path)
            print(f"Test extraction result: '{text}'")
            
            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)
            
            return text.strip() != ""
        except Exception as e:
            print(f"Model test failed: {e}")
            return False

def main():
    root = Tk()
    root.withdraw()
    evaluator = PDFAnswerEvaluator()
    
    # Test the model first
    print("Testing TrOCR model...")
    if not evaluator.test_model():
        messagebox.showerror("Model Error", "TrOCR model is not working properly. Please check your installation.")
        return
    
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
        
        # Add more detailed error handling
        try:
            student_text, student_images = evaluator.process_pdf_to_text(student_pdf)
            print(f"Processed {len(student_text)} pages")
        except Exception as e:
            messagebox.showerror("PDF Processing Error", f"Failed to process student PDF:\n{str(e)}")
            return
            
        try:
            student_answers = evaluator.extract_answers(student_text)
            print(f"Extracted {len(student_answers)} answers")
            
            # If no answers were extracted, offer manual input
            if not student_answers:
                response = messagebox.askyesno("No Answers Detected", 
                    "No answers were automatically detected from the PDF.\n\n"
                    "Would you like to manually enter the answers?")
                if response:
                    student_answers = manual_answer_input()
                else:
                    messagebox.showinfo("Cancelled", "No answers to evaluate.")
                    return
        except Exception as e:
            messagebox.showerror("Answer Extraction Error", f"Failed to extract answers:\n{str(e)}")
            return
            
        try:
            answer_key = evaluator.extract_answer_key(answer_key_pdf)
        except Exception as e:
            messagebox.showerror("Answer Key Error", f"Failed to process answer key:\n{str(e)}")
            return
            
        try:
            results = evaluator.evaluate_answers(student_answers, answer_key)
        except Exception as e:
            messagebox.showerror("Evaluation Error", f"Failed to evaluate answers:\n{str(e)}")
            return
            
        report_path = filedialog.asksaveasfilename(title="Save Evaluation Report", defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")], initialfile="Answer_Evaluation_Report.pdf")
        if report_path:
            try:
                evaluator.generate_report(results, student_images, report_path)
                messagebox.showinfo("Success", f"Evaluation report saved to:\n{report_path}")
            except Exception as e:
                messagebox.showerror("Report Generation Error", f"Failed to generate report:\n{str(e)}")
        
        # Clean up temp files
        for img in student_images:
            if os.path.exists(img):
                os.remove(img)
                
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    finally:
        root.destroy()

def manual_answer_input():
    """Allow manual input of answers if OCR fails"""
    root = Tk()
    root.title("Manual Answer Input")
    root.geometry("600x400")
    
    answers = {}
    
    def add_answer():
        q_num = question_entry.get().strip()
        answer_text = answer_text_widget.get("1.0", "end-1c").strip()
        if q_num and answer_text:
            answers[f"Q{q_num}"] = answer_text
            question_entry.delete(0, "end")
            answer_text_widget.delete("1.0", "end")
            update_answer_list()
    
    def update_answer_list():
        answer_list.delete(0, "end")
        for q, a in answers.items():
            answer_list.insert("end", f"{q}: {a[:50]}...")
    
    def finish():
        root.destroy()
    
    # GUI elements
    from tkinter import ttk
    
    Label(root, text="Question Number:").pack(pady=5)
    question_entry = Entry(root, width=20)
    question_entry.pack(pady=5)
    
    Label(root, text="Answer:").pack(pady=5)
    answer_text_widget = Text(root, height=4, width=50)
    answer_text_widget.pack(pady=5)
    
    Button(root, text="Add Answer", command=add_answer).pack(pady=10)
    
    Label(root, text="Added Answers:").pack(pady=5)
    answer_list = Listbox(root, height=10, width=60)
    answer_list.pack(pady=5)
    
    Button(root, text="Finish", command=finish).pack(pady=10)
    
    root.mainloop()
    return answers

if __name__ == "__main__":
    try:
        from tkinter import Tk
        main()
    except ImportError:
        print("Error: Tkinter not found. Please install it.")
        sys.exit(1)
