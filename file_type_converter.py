import os
from pillow_heif import open_heif
from PIL import Image

# Define input and output directories
input_dir = "./Data/FullPieces"  # Change this to HEIC folder
output_dir = "./Data/FullPieces/cleaned"  # Change this to where JPGs will be saved

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of HEIC files sorted alphabetically
heic_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".heic")])

# Convert and save as img1.jpg, img2.jpg, ect
for i, filename in enumerate(heic_files, start=1):
    heif_image = open_heif(os.path.join(input_dir, filename))
    img = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data, "raw", heif_image.mode)
    
    output_path = os.path.join(output_dir, f"img{i}.jpg")
    img.save(output_path, "JPEG")

    print(f"Converted {filename} -> {output_path}")

print("Conversion complete!")