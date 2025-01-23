import sys
import torch
import clip
from PIL import Image
import csv

# Automatically select the device based on availability
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA-enabled GPU on systems with NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS on macOS with Apple Silicon
else:
    device = torch.device("cpu")  # Fallback to CPU if no GPU is available

print(f"Using device: {device}")

# Load the CLIP model and preprocess function
# model, preprocess = clip.load("ViT-B/32", device=device) #  86 million parameters
# model, preprocess = clip.load("ViT-L/14", device=device) # 428 million parameters, >2GB VRAM
model, preprocess = clip.load("ViT-L/14@336px", device=device) # 336x336px instead of 224x224

# Function to load categories from a TSV file
def load_categories(tsv_file):
    categories = []
    try:
        with open(tsv_file, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                categories.append((row["label"], row["description"]))
    except Exception as e:
        print(f"Error reading categories file: {e}")
        sys.exit(1)
    return categories

# Load categories from external TSV file
categories_tsv = "categories.tsv"  # Replace with your TSV file path
categories = load_categories(categories_tsv)

# Read input file names from STDIN
print("Enter image file paths (one per line). Press Ctrl+D (or Ctrl+Z on Windows) to end input:")
image_paths = sys.stdin.read().strip().splitlines()

# Process each image
for image_path in image_paths:
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Prepare text descriptions for the CLIP model
        text_inputs = torch.cat([clip.tokenize(description) for _, description in categories]).to(device)

        # Perform inference
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            # Compute similarity scores
            logits_per_image, _ = model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Display results
        print(f"\nResults for image: {image_path}")
        for (label, _), prob in zip(categories, probs[0]):
            print(f"{label}: {prob * 100:.2f}%")

    except Exception as e:
        print(f"Error processing file {image_path}: {e}")
