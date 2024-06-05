import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

from torchvision import transforms

from transformers import AutoModelForObjectDetection#, AutoModel
import torch

#import easyocr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import gradio as gr


device = "cuda" if torch.cuda.is_available() else "cpu"


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#model.save_pretrained(r'C:\Users\Simas\Desktop\table-transformer-detection')
#structure_model.save_pretrained(r'C:\Users\Simas\Desktop\table-transformer-structure-recognition-v1.1-all')

# load table detection model
# processor = TableTransformerImageProcessor(max_size=800)
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm").to(device)

# load table structure recognition model
# structure_processor = TableTransformerImageProcessor(max_size=1000)
structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(device)

# load EasyOCR reader
#reader = easyocr.Reader(['en'], model_storage_directory='/path/to/local/models')



# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    return image


def visualize_detected_tables(img, det_tables):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Table', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    return fig


def detect_and_crop_tables(image):
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    cropped_tables = [image.crop(table["bbox"]) for table in detected_tables if table['label'] in ['table', 'table rotated']]

    return cropped_tables



def recognize_table(image):
    pixel_values = structure_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    id2label = structure_model.config.id2label
    id2label[len(structure_model.config.id2label)] = "no object"
    cells = outputs_to_objects(outputs, image.size, id2label)
    draw = ImageDraw.Draw(image)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")
        
    return image, cells


def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox


    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        row_cells.sort(key=lambda x: x['column'][0])

        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            cell_image = cropped_table.crop(cell["cell"])
            text = pytesseract.image_to_string(cell_image, lang='eng')

            clean_text = text.strip().replace('\n', ' ')
            row_text.append(clean_text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        
        data[str(idx)] = row_text

    for idx, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[str(idx)] = row_data

    with open('output.csv', 'w', newline='', encoding='utf-8') as result_file:
        wr = csv.writer(result_file)
        for row_text in data.values():
            wr.writerow(row_text)

    df = pd.read_csv('output.csv')

    return df, data


def process_pdf(image):
    cropped_table = detect_and_crop_table(image)

    image, cells = recognize_table(cropped_table)

    cell_coordinates = get_cell_coordinates_by_row(cells)

    df, data = apply_ocr(cell_coordinates, image)

    return image, df, data
    

##def process_image_file(image_path):
##    # Load the image
##    image = Image.open(image_path)
##    
##    # Detect and crop all tables from the image
##    cropped_tables = detect_and_crop_tables(image)
##    
##    results = []
##    for index, cropped_table in enumerate(cropped_tables, start=1):
##        # Recognize cells within the cropped table image
##        recognized_image, cells = recognize_table(cropped_table)
##        
##        # Get cell coordinates grouped by rows
##        cell_coordinates = get_cell_coordinates_by_row(cells)
##        
##        # Apply OCR to extract text from each cell
##        dataframe, data = apply_ocr(cell_coordinates, recognized_image)
##        
##        # Save the dataframe to CSV
##        csv_filename = f'extracted_table_{index}.csv'
##        dataframe.to_csv(csv_filename, index=False)
##        
##        print(f"Table {index} data has been successfully extracted and saved to '{csv_filename}'")
##        results.append((recognized_image, dataframe, data))
##    
##    return results

root_folder = r'C:\Users\Simas\Desktop\Ataskaitos\kursinis\tatr-demo\Processed Images - tesseract'  # Path to the folder containing subfolders with images

import os
def process_all_images_in_folder(folder_path):
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            print(f"Entering directory: {full_path}")
            for file in os.listdir(full_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(full_path, file)
                    print(f"Processing {file_path}...")
                    try:
                        process_image_file(file_path)
                        print(f"Processed {file} successfully.")
                    except Exception as e:
                        print(f"Failed to process {file}. Error: {e}")
                else:
                    print(f"Skipped {file} - not an image file.")
            print(f"Finished processing directory: {full_path}")
        else:
            print(f"Skipped {full_path} - not a directory.")

def process_image_file(image_path):

    image = Image.open(image_path)
    

    cropped_tables = detect_and_crop_tables(image)
    
    results = []
    for index, cropped_table in enumerate(cropped_tables, start=1):
        recognized_image, cells = recognize_table(cropped_table)
        
        cell_coordinates = get_cell_coordinates_by_row(cells)
        
        dataframe, data = apply_ocr(cell_coordinates, recognized_image)
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        csv_filename = os.path.join(os.path.dirname(image_path), f'{base_filename}_{index}.csv')
        dataframe.to_csv(csv_filename, index=False)
        
        print(f"Table {index} data has been successfully extracted and saved to '{csv_filename}'")
        results.append((recognized_image, dataframe, data))
    
    return results

title = "Demo: table detection & recognition with Table Transformer (TATR)."
description = """Demo for table extraction with the Table Transformer. First, table detection is performed on the input image using https://huggingface.co/microsoft/table-transformer-detection,
after which the detected table is extracted and https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all is leveraged to recognize the individual rows, columns and cells. OCR is then performed per cell, row by row."""
examples = [['image.png'], ['mistral_paper.png']]

##app = gr.Interface(fn=process_pdf, 
##                     inputs=gr.Image(type="pil"), 
##                     outputs=[gr.Image(type="pil", label="Detected table"), gr.Dataframe(label="Table as CSV"), gr.JSON(label="Data as JSON")],
##                     title=title,
##                     description=description,
##                     examples=examples)
##app.queue()
##app.launch(debug=True)

#image_path = '_1Q21_page_7.png'  # Specify the path to your image file
#process_image_file(image_path)

root_folder = r'C:\Users\Simas\Desktop\Ataskaitos\kursinis\tatr-demo\Processed Images - tesseract'  # Path to the folder containing subfolders with images
process_all_images_in_folder(root_folder)
