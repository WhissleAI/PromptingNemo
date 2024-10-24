import dropbox
import os

# Define your Dropbox access token
DROPBOX_ACCESS_TOKEN = '<add-your-dropbox-access-token-here>'

# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Define the Dropbox folder path to download
dropbox_folder_path = '/marathi_processed'  # Folder path in Dropbox to download as a ZIP file

# Define the local path to save the ZIP file
local_zip_file_path = '/projects/whissle/datasets/marathi/marathi_processed.zip'

# Function to download a Dropbox folder as a ZIP file
def download_folder_as_zip(dropbox_folder_path, local_zip_file_path):
    try:
        print(f"Downloading {dropbox_folder_path} as a ZIP file...")
        # Download the folder as a ZIP file
        metadata, response = dbx.files_download_zip(path=dropbox_folder_path)
        
        # Write the content to a local file
        with open(local_zip_file_path, "wb") as f:
            f.write(response.content)
        
        print(f"Folder {dropbox_folder_path} has been downloaded as {local_zip_file_path}")
    except Exception as e:
        print(f"Error downloading folder {dropbox_folder_path} as ZIP: {e}")

# Start the download process
download_folder_as_zip(dropbox_folder_path, local_zip_file_path)

print(f"Folder {dropbox_folder_path} has been downloaded to {local_zip_file_path}")
