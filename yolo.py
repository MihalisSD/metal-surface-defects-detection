import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import glob
import random
from matplotlib.patches import Rectangle
from lxml import etree


annotations_path = r'/content/drive/MyDrive/ds/ann'
train_annotations_path = r'/content/drive/MyDrive/labels/train'
val_annotations_path = r'/content/drive/MyDrive/labels/val'
test_annotations_path = r'/content/drive/MyDrive/labels/test'

# Class names mapping
class_names = {
    0: 'crease',
    1: 'crescent_gap',
    2: 'inclusion',
    3: 'oil_spot',
    4: 'punching_hole',
    5: 'rolled_pit',
    6: 'silk_spot',
    7: 'waist_folding',
    8: 'water_spot',
    9: 'welding_line'
}

all_classes = list(class_names.values())

def calculate_class_distribution(annotation_dir):
    class_counts = Counter()
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.txt'):
            with open(os.path.join(annotation_dir, annotation_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts

train_distribution = calculate_class_distribution(train_annotations_path)
val_distribution = calculate_class_distribution(val_annotations_path)
test_distribution = calculate_class_distribution(test_annotations_path)

def plot_class_distribution(distribution, title, ax):
    counts = [distribution.get(cls_id, 0) for cls_id in class_names.keys()]
    classes = all_classes
    ax.bar(classes, counts)
    ax.set_title(title)
    ax.set_xlabel('Class Name')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)

fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plot each distribution
plot_class_distribution(train_distribution, 'Train Class Distribution', axs[0])
plot_class_distribution(val_distribution, 'Validation Class Distribution', axs[1])
plot_class_distribution(test_distribution, 'Test Class Distribution', axs[2])

plt.tight_layout()
plt.show()

print(torch.cuda.is_available())

# Paths to the dataset and data.yaml file
data_yaml_path = r'/content/drive/MyDrive/ds/data.yaml'
output_dir = r'/content/drive/MyDrive/ds/runs/train'

# Check if output directory exists, if not, create it
os.makedirs(output_dir, exist_ok=True)

# Check if data.yaml file exists
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

# Load the YOLO model (YOLOv8)
model = YOLO('yolov8n.pt')  # You can choose other model sizes: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
model.train(
    data=data_yaml_path,
    epochs=100,  # Number of training epochs
    batch=16,  # Batch size
    imgsz=640,  # Image size
    project=output_dir,
    name='exp',  # Experiment name
    device=0,
    exist_ok=True
)

# Load a model
model = YOLO(r"/content/drive/MyDrive/ds/runs/train/exp8/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(data = r'/content/drive/MyDrive/ds/data.yaml', split = 'test', plots = True)  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category