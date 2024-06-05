"""Predicting Module."""

from collections import OrderedDict
import os
from typing import List

import click
import numpy as np
import pandas as pd
from albumentations import Compose
from PIL import Image
from google.cloud import vision
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image
from skimage.transform import resize
from skimage.util import invert
from io import BytesIO

from tablenet import TableNetModule

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'C:\Users\Simas\Desktop\Ataskaitos\kursinis\tatr-demo\mindful-coder-418621-d4d1824a6b40.json'

client = vision.ImageAnnotatorClient()

class Predict:
    """Predict images using a pre-trained model and Google Cloud Vision."""

    def __init__(self, checkpoint_path: str, transforms: Compose, threshold: float = 0.5, per: float = 0.005):
        """Initialize the predictor.
        
        Args:
            checkpoint_path (str): Path to the model weights.
            transforms (Compose): Albumentations transformations for image preprocessing.
            threshold (float): Decision threshold for classification.
            per (float): Minimum area ratio to consider detected regions as tables/columns.
        """
        self.transforms = transforms
        self.threshold = threshold
        self.per = per

        self.model = TableNetModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.requires_grad_(False)

    def predict(self, image: Image) -> List[pd.DataFrame]:
        """Predict table content from an image using Google Cloud Vision OCR.
        
        Args:
            image (Image): Image to process.
        
        Returns:
            List[pd.DataFrame]: DataFrames representing the tables found in the image.
        """
        processed_image = self.transforms(image=np.array(image))["image"]
        table_mask, column_mask = self.model.forward(processed_image.unsqueeze(0))

        table_mask = self._apply_threshold(table_mask)
        column_mask = self._apply_threshold(column_mask)

        segmented_tables = self._process_tables(self._segment_image(table_mask))
        tables = []
        for table in segmented_tables:
            segmented_columns = self._process_columns(self._segment_image(column_mask * table))
            if segmented_columns:
                cols = []
                for column in segmented_columns.values():
                    cols.append(self._column_to_dataframe(column, image))
                tables.append(pd.concat(cols, axis=1))
        return tables

    def _apply_threshold(self, mask) -> np.array:
        """Apply threshold to mask to create binary image.
        
        Args:
            mask (tensor): The mask tensor output from the model.
        
        Returns:
            np.array: Binary image from mask.
        """
        return (mask.squeeze(0).squeeze(0).numpy() > self.threshold).astype(int)

    def _process_tables(self, segmented_tables: np.array) -> List[np.array]:
        width, height = segmented_tables.shape
        tables = []
        for i in np.unique(segmented_tables)[1:]:
            table = (segmented_tables == i).astype(int)
            area = table.sum()
            print(f"Table {i} Area: {area}") 
            if area > height * width * self.per:
                tables.append(convex_hull_image(table))
        return tables


    def _process_columns(self, segmented_columns: np.array) -> OrderedDict:
        """Process segmented columns to sort by position and filter by area.
        
        Args:
            segmented_columns (np.array): Labeled column regions.
        
        Returns:
            OrderedDict: Sorted and processed columns as binary masks.
        """
        width, height = segmented_columns.shape
        cols = {}
        for j in np.unique(segmented_columns)[1:]:  # Skip background
            column = (segmented_columns == j).astype(int)
            if column.sum() > width * height * self.per:
                position = regionprops(column)[0].centroid[1]
                cols[position] = column
        return OrderedDict(sorted(cols.items()))

    @staticmethod
    def _segment_image(image: np.array) -> np.array:
        """Segment image into discrete regions.
        
        Args:
            image (np.array): Binary image to segment.
        
        Returns:
            np.array: Labeled image regions.
        """
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))
        cleared = clear_border(bw)
        return label(cleared)

    @staticmethod
    def _column_to_dataframe(column: np.array, image: Image) -> pd.DataFrame:
        """Convert a segmented column to a DataFrame using Google Cloud Vision OCR.
        
        Args:
            column (np.array): Column mask.
            image (Image): Original image for OCR extraction.
        
        Returns:
            pd.DataFrame: Extracted text as DataFrame.
        """
        column = column.astype(bool)
        height, width = image.size
        image_array = np.array(image)
        
        column_resized = resize(column, (width, height), order=0, preserve_range=True).astype(bool)
        cropped_image = image_array * column_resized[:,:,np.newaxis]  # add channel dimensions for broadcasting
        
        
        pil_image = Image.fromarray(np.uint8(cropped_image))


        byte_io = BytesIO()
        pil_image.save(byte_io, format='PNG')
        content = byte_io.getvalue()
        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations


        if response.error.message:
            raise Exception(f"Google Cloud Vision API Error: {response.error.message}")

        lines = [text.description for text in texts[1:]]
        return pd.DataFrame({"col": lines})
    
@click.command()
@click.option('--root_folder', default=r'C:\Users\Simas\Desktop\Ataskaitos\kursinis\GOOD OCR_tablenet-master\Processed Images -google cloud')
@click.option('--model_weights', default="./data/best_model.ckpt")
def predict(root_folder: str, model_weights: str) -> None:
    """Process all images in the root folder and its subfolders."""
    import albumentations as album
    from albumentations.pytorch.transforms import ToTensorV2

    transforms = album.Compose([
        album.Resize(896, 896, always_apply=True),
        album.Normalize(),
        ToTensorV2()
    ])
    

    pred = Predict(model_weights, transforms)

    os.makedirs('output', exist_ok=True)
    
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(subdir, file)
                print(f"Processing {image_path}...")
                try:
                    
                    image = Image.open(image_path)
                    
                    tables = pred.predict(image)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    for idx, table in enumerate(tables):
                        file_path = os.path.join(subdir, f"{base_name}_table_{idx + 1}.csv")
                        table.to_csv(file_path, index=False, header=False)
                        print(f"Saved {file_path}")
                    print(f"Processed {file} successfully.")
                except Exception as e:
                    print(f"Failed to process {file}. Error: {e}")
            else:
                print(f"Skipped {file} - not an image file.")

if __name__ == '__main__':
    predict()
