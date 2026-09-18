import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForTextRecognition,
)
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors"

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForTextRecognition.from_pretrained(MODEL).to(device)
model.eval()

image = Image.open("frame_3s.png").convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt",
).to(device)
from time import perf_counter as t

with torch.inference_mode():
    s = t()
    outputs = model(**inputs)
    e = t()
print(e-s)
results = processor.post_process_text_recognition(outputs)

for result in results:
    print(result)