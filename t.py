from pprint import pprint
import torch
from transformers import AutoImageProcessor, AutoModelForTextRecognition
from PIL import Image
from time import perf_counter as t
import dotenv

dotenv.load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors"

# 1. Load processor and recognition model
processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForTextRecognition.from_pretrained(MODEL).to(device)
model.eval()

# 2. Load the full image
full_image = Image.open("image.png").convert("RGB")
width, height = full_image.size

# 3. Crop tightly around the license plate [left, top, right, bottom]
# Coordinates correspond to the red bounding box on your car image
crop_box = (
    int(width * 0.31),   # left
    int(height * 0.47),  # top
    int(width * 0.72),   # right
    int(height * 0.61)   # bottom
)
cropped_plate = full_image.crop(crop_box)

# (Optional) Save to verify the crop contains only "ZG 7497-AH"
cropped_plate.save("cropped_plate.png")

# 4. Run recognition on the cropped text strip
inputs = processor(
    images=cropped_plate,
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    s = t()
    outputs = model(**inputs)
    e = t()

print(f"Inference time: {e - s:.4f}s")

# 5. Decode results
results = processor.post_process_text_recognition(outputs)
for result in results:
    pprint(result)