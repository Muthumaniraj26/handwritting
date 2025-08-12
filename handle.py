import torch
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from torchvision.transforms import ToTensor
from PIL import Image

# Load dataset and split into train/test
raw_dataset = load_dataset("bblzzrd/trocr-handwritten-dataset-es")
dataset = raw_dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load pretrained TrOCR model and processor
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# ✅ Set necessary configuration attributes
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id  
# Preprocess function
def preprocess(example):
    image = example["image"].convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(
        example["text"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128  # You can adjust based on your data
    ).input_ids[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding in loss
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# Apply preprocessing
train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    num_train_epochs=10,
    logging_dir="./logs",
    report_to="none"  # Disable wandb or tensorboard reporting
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator,
)

# Start training
trainer.train()

# Save model and processor
model.save_pretrained("./trocr-handwriting-model")
processor.save_pretrained("./trocr-handwriting-processor")
