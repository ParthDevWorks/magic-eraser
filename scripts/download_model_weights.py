import gdown

# This is `Folder ID` when you click on `model_weights` folder in Google Drive
# Make Sure the link is set to `Anyone with the link` can view
WEIGHTS_FOLDER_SHAREABLE_URl = (
    "https://drive.google.com/drive/folders/1u0MEH2Apr9SFkI5bqgTyrmEgmet3S4aL"
)
OUTPUT_FOLDER = "model_weights"


def download_weights():
    gdown.download_folder(WEIGHTS_FOLDER_SHAREABLE_URl, output=OUTPUT_FOLDER)
