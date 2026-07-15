import os
from datetime import datetime

import gdown
from google.oauth2 import service_account
from googleapiclient.discovery import build

# This is `Folder ID` when you click on `model_weights` folder in Google Drive
# Make Sure the link is set to `Anyone with the link` can view
MODEL_WEIGHTS_FOLDER_ID = os.getenv("MODEL_WEIGHTS_FOLDER_ID")
if not MODEL_WEIGHTS_FOLDER_ID:
    raise ValueError("Please set the environment variable MODEL_WEIGHTS_FOLDER_ID.")

WEIGHTS_FOLDER_SHAREABLE_URl = (
    f"https://drive.google.com/drive/folders/{MODEL_WEIGHTS_FOLDER_ID}"
)
OUTPUT_FOLDER = "model_weights"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_CLIENT_EMAIL = os.environ["GOOGLE_CLIENT_EMAIL"]
GOOGLE_PRIVATE_KEY = os.environ["GOOGLE_PRIVATE_KEY"]


def download_weights():
    gdown.download_folder(WEIGHTS_FOLDER_SHAREABLE_URl, output=OUTPUT_FOLDER)


def _get_latest_modified_time(folder_id: str):
    service_account_info = {
        "type": "service_account",
        "client_email": GOOGLE_CLIENT_EMAIL,
        "private_key": GOOGLE_PRIVATE_KEY.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES,
    )

    drive_service = build(
        "drive",
        "v3",
        credentials=credentials,
    )

    query = f"'{folder_id}' in parents and trashed=false"

    response = (
        drive_service.files()
        .list(q=query, fields="files(id,name,mimeType,modifiedTime)")
        .execute()
    )

    latest = None

    for file in response["files"]:
        modified = datetime.fromisoformat(file["modifiedTime"].replace("Z", "+00:00"))

        if latest is None or modified > latest:
            latest = modified

        # recurse into subfolders
        if file["mimeType"] == "application/vnd.google-apps.folder":
            child_latest = _get_latest_modified_time(file["id"])

            if child_latest and child_latest > latest:
                latest = child_latest

    return latest


def weights_version():
    latest = _get_latest_modified_time(MODEL_WEIGHTS_FOLDER_ID)
    print(latest.strftime("%Y%m%d%H%M%S"))
