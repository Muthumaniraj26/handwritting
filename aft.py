from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tkinter import Tk, filedialog

# Load trained model & processor
model = VisionEncoderDecoderModel.from_pretrained("./trocr-handwriting-output")
processor = TrOCRProcessor.from_pretrained("./trocr-handwriting-output")

# Optional: Use beam search (same as training)
model.generation_config.num_beams = 4
model.generation_config.early_stopping = True

# GUI for file selection
root = Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Select an image file")

# Process & predict
image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

# Decode result
result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Prediction:", result)
