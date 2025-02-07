import os
import cv2
import numpy as np

# Take user input for source and destination directories
SOURCE_DIR = input("Enter the source directory path: ").strip()
OUTPUT_DIR = input("Enter the destination directory path: ").strip()

# Check if the source directory exists
if not os.path.exists(SOURCE_DIR):
    print(f"Error: The source directory '{SOURCE_DIR}' does not exist.")
    exit()

# Ensure the destination directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define filters
def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def invert(image):
    return cv2.bitwise_not(image)

# (Other filter functions remain unchanged)

# Apply filters to images
def apply_filters(image, base_name):
    filter_names = ['sharpen', 'invert']
    filters = [sharpen, invert]

    for name, filter_func in zip(filter_names, filters):
        filtered_img = filter_func(image)
        output_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(output_path, filtered_img)

# Process images in the source directory
def process_images(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                base_name, _ = os.path.splitext(file_name)
                print(f"Processing: {file_name}")
                apply_filters(image, base_name)
            else:
                print(f"Could not read {file_name}. Skipping...")

# Main program
if __name__ == "__main__":
    process_images(SOURCE_DIR)
    print("Processing complete. Filtered images saved.")
