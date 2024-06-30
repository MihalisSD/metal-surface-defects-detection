import os
from collections import Counter
import matplotlib.pyplot as plt

# Paths
train_annotations_path = r'C:\Users\Mihalis\Desktop\multimodal\gc10det_dataset\labels\train'
val_annotations_path = r'C:\Users\Mihalis\Desktop\multimodal\gc10det_dataset\labels\val'
test_annotations_path = r'C:\Users\Mihalis\Desktop\multimodal\gc10det_dataset\labels\test'

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

# Ensure classes are always in the same order
all_classes = list(class_names.values())

# Function to calculate class distribution
def calculate_class_distribution(annotation_dir):
    class_counts = Counter()
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.txt'):
            with open(os.path.join(annotation_dir, annotation_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts

# Calculate class distribution for each set
train_distribution = calculate_class_distribution(train_annotations_path)
val_distribution = calculate_class_distribution(val_annotations_path)
test_distribution = calculate_class_distribution(test_annotations_path)

# Function to plot class distribution
def plot_class_distribution(distribution, title, ax):
    counts = [distribution.get(cls_id, 0) for cls_id in class_names.keys()]
    classes = all_classes
    ax.bar(classes, counts)
    ax.set_title(title)
    ax.set_xlabel('Class Name')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plot each distribution
plot_class_distribution(train_distribution, 'Train Class Distribution', axs[0])
plot_class_distribution(val_distribution, 'Validation Class Distribution', axs[1])
plot_class_distribution(test_distribution, 'Test Class Distribution', axs[2])

# Adjust layout and show plot
plt.tight_layout()
plt.show()
