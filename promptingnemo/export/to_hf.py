"""Upload a NeMo model to Hugging Face Hub with retry logic.

Extracted from scripts/utils/nemo2hf.py.
"""

import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


def upload_with_retry(api, path, repo_id, path_in_repo, token, max_retries=3, delay=5):
    """Attempt to upload file with retries on failure."""
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
            )
            print(f"Successfully uploaded {path_in_repo}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Upload attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                print(f"Error: {str(e)}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Failed to upload {path_in_repo} after {max_retries} attempts: {str(e)}")
                return False


def upload_nemo_to_hf(
    nemo_model_path: str,
    repo_id: str,
    hf_token: str,
    local_dir: str = None,
    readme_content: str = None,
    model_filename: str = None,
):
    """Upload a NeMo model checkpoint (and optional README) to Hugging Face Hub.

    Args:
        nemo_model_path: Path to the .nemo model checkpoint.
        repo_id: Hugging Face repository ID (e.g. 'WhissleAI/my-model').
        hf_token: Hugging Face API token.
        local_dir: Optional local staging directory. If None, uses a temp dir.
        readme_content: Optional README.md content string. If None, no README is uploaded.
        model_filename: Filename for the model in the repo. Defaults to the original name.
    """
    nemo_model_path = Path(nemo_model_path)
    if not nemo_model_path.exists():
        raise FileNotFoundError(f"Model not found: {nemo_model_path}")

    if local_dir is None:
        local_dir = Path(f"/tmp/hf_upload_{repo_id.replace('/', '_')}")
    else:
        local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    api = HfApi()

    # Ensure repository exists
    try:
        api.repo_info(repo_id)
        print(f"Repository {repo_id} already exists")
    except RepositoryNotFoundError:
        print(f"Creating new repository: {repo_id}")
        create_repo(repo_id, token=hf_token, private=False)

    # Copy model file
    if model_filename is None:
        model_filename = nemo_model_path.name
    model_dest = local_dir / model_filename
    print(f"Copying model file to {model_dest}")
    shutil.copy2(nemo_model_path, model_dest)

    files_to_upload = [(model_dest, model_filename)]

    # Create README if content provided
    if readme_content:
        readme_path = local_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        files_to_upload.append((readme_path, "README.md"))

    # Upload files
    for file_path, repo_path in files_to_upload:
        print(f"\nUploading {repo_path}...")
        if not upload_with_retry(api, file_path, repo_id, repo_path, hf_token):
            raise RuntimeError(f"Failed to upload {repo_path}")

    print("\nAll files uploaded successfully!")
