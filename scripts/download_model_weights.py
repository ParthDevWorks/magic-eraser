import os
import gdown

# This is `Folder ID` when you click on `model_weights` folder in Google Drive
# Make Sure the link is set to `Anyone with the link` can view
MODEL_WEIGHTS_FOLDER_ID = os.getenv("MODEL_WEIGHTS_FOLDER_ID")
if not MODEL_WEIGHTS_FOLDER_ID:
    raise ValueError("Please set the environment variable MODEL_WEIGHTS_FOLDER_ID.")

WEIGHTS_FOLDER_SHAREABLE_URl = (
    f"https://drive.google.com/drive/folders/{MODEL_WEIGHTS_FOLDER_ID}"
)
OUTPUT_FOLDER = "model_weights"


def download_weights():
    gdown.download_folder(WEIGHTS_FOLDER_SHAREABLE_URl, output=OUTPUT_FOLDER)
