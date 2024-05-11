import os
import pandas as pd
import csv

labelMap = {
    "Animal": 0,
    "Building": 1,
    "Mountain": 2,
    "Street": 3
}

# Function to generate CSV file with image paths and labels
def generate_csv(root_dir, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')

        for root, dirs, files in os.walk(root_dir):
            if 'del' not in root:
                for file in files:
                    if file.endswith('.png'):
                        image_path = os.path.join(root, file)
                        label = os.path.basename(root)
                        writer.writerow({'image_path': image_path, 'label': labelMap[label]})

# Example usage
root_directory = 'dataset/valid'
csv_filename = 'dataset/image_labels_valid.csv'
# generate_csv(root_directory, csv_filename)

img_labels = pd.read_csv(csv_filename, header=None, delimiter=' ')
print(len(img_labels))


