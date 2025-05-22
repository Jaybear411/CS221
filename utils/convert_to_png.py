#!/usr/bin/env python3
import os
import numpy as np
import json
from PIL import Image, ImageDraw
import argparse
import glob
from tqdm import tqdm

def vector_to_raster(vector_data, size=64, line_thickness=2, padding=4):
    """Convert a drawing from vector format to a raster image."""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Determine bounds and scaling factors
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    for stroke in vector_data:
        for i in range(len(stroke[0])):
            min_x = min(min_x, stroke[0][i])
            max_x = max(max_x, stroke[0][i])
            min_y = min(min_y, stroke[1][i])
            max_y = max(max_y, stroke[1][i])
    
    # Calculate scaling factor
    scale_x = (size - 2 * padding) / (max_x - min_x) if max_x > min_x else 1
    scale_y = (size - 2 * padding) / (max_y - min_y) if max_y > min_y else 1
    scale = min(scale_x, scale_y)
    
    # Calculate offset to center drawing
    offset_x = padding + (size - 2 * padding - scale * (max_x - min_x)) / 2
    offset_y = padding + (size - 2 * padding - scale * (max_y - min_y)) / 2
    
    # Draw the strokes
    for stroke in vector_data:
        points = []
        for i in range(len(stroke[0])):
            x = int(offset_x + scale * (stroke[0][i] - min_x))
            y = int(offset_y + scale * (stroke[1][i] - min_y))
            points.append((x, y))
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=0, width=line_thickness)
    
    return img

def process_ndjson_file(input_file, output_dir, max_samples=5000):
    """Process a .ndjson file and convert the drawings to PNG format."""
    os.makedirs(output_dir, exist_ok=True)
    
    category = os.path.basename(input_file).split('.')[0]
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    with open(input_file, 'r') as f:
        count = 0
        for line in tqdm(f, desc=f"Processing {category}", total=max_samples):
            if count >= max_samples:
                break
                
            try:
                drawing = json.loads(line)
                image = vector_to_raster(drawing['drawing'])
                
                # Save the image
                image_path = os.path.join(category_dir, f"{drawing['key_id']}.png")
                image.save(image_path)
                
                count += 1
            except Exception as e:
                print(f"Error processing drawing: {e}")
    
    print(f"Processed {count} images for {category}")

def main():
    parser = argparse.ArgumentParser(description="Convert Quick Draw dataset to PNG images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .ndjson files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for PNG images")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples per category")
    
    args = parser.parse_args()
    
    # Find all .ndjson files in the input directory
    ndjson_files = glob.glob(os.path.join(args.input_dir, "*.ndjson"))
    
    if not ndjson_files:
        print(f"No .ndjson files found in {args.input_dir}")
        return
    
    print(f"Found {len(ndjson_files)} .ndjson files")
    
    for ndjson_file in ndjson_files:
        process_ndjson_file(ndjson_file, args.output_dir, args.max_samples)

if __name__ == "__main__":
    main() 