import os
import json
import shutil
import cv2  # Import OpenCV for image handling
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths
dataset_path = r'C:\Users\Mihalis\Desktop\multimodal\gc10det_dataset'
images_path = os.path.join(dataset_path, 'ds', 'img')
annotations_path = os.path.join(dataset_path, 'ds', 'ann')
train_images_path = os.path.join(dataset_path, 'images', 'train')
val_images_path = os.path.join(dataset_path, 'images', 'val')
test_images_path = os.path.join(dataset_path, 'images', 'test')
train_labels_path = os.path.join(dataset_path, 'labels', 'train')
val_labels_path = os.path.join(dataset_path, 'labels', 'val')
test_labels_path = os.path.join(dataset_path, 'labels', 'test')

# Create directories
for path in [train_images_path, val_images_path, test_images_path, train_labels_path, val_labels_path, test_labels_path]:
    os.makedirs(path, exist_ok=True)

# List all image files (consider both .jpg and .png formats)
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

# Class names mapping
class_names = {
    8742: 0,  # 'crease'
    8736: 1,  # 'crescent_gap'
    8740: 2,  # 'inclusion'
    8738: 3,  # 'oil_spot'
    8734: 4,  # 'punching_hole'
    8741: 5,  # 'rolled_pit'
    8739: 6,  # 'silk_spot'
    8743: 7,  # 'waist_folding'
    8737: 8,  # 'water_spot'
    8735: 9,  # 'welding_line'
}

# Split the dataset into train (1610), val (345), and test (345) sets
train_files, temp_files = train_test_split(image_files, train_size=1610, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=345, random_state=42)

# Function to convert and save annotations in YOLO format
def convert_and_save_annotations(image_files, image_dir, label_dir):
    for image_file in image_files:
        annotation_file = image_file + '.json'
        with open(os.path.join(annotations_path, annotation_file), 'r') as f:
            data = json.load(f)
        
        # Get image dimensions
        img_path = os.path.join(images_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue
        h, w, _ = img.shape
        
        # Copy image to the new directory
        shutil.copy(img_path, os.path.join(image_dir, image_file))
        
        # Create YOLO format label file
        yolo_file = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(yolo_file, 'w') as f_out:
            for obj in data['objects']:
                class_id = obj['classId']
                yolo_class_id = class_names[class_id]
                x1, y1 = obj['points']['exterior'][0]
                x2, y2 = obj['points']['exterior'][1]
                # Calculate YOLO format values
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                f_out.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Convert and save annotations for each set
convert_and_save_annotations(train_files, train_images_path, train_labels_path)
convert_and_save_annotations(val_files, val_images_path, val_labels_path)
convert_and_save_annotations(test_files, test_images_path, test_labels_path)

print(f'Train files: {len(train_files)}')
print(f'Val files: {len(val_files)}')
print(f'Test files: {len(test_files)}')

@dataset{GC10-DET,
	author={Xiaoming Lv and Fajie Duan and Jia-jia Jiang and Xiao Fu and Lin Gan},
	title={GC10-DET: Metallic Surface Defect Detection},
	year={2020},
	url={https://www.kaggle.com/datasets/alex000kim/gc10det}
}
