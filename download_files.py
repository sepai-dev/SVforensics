import os
import json
import hashlib
import requests
import gdown

# JSON file to store download information
DOWNLOAD_INFO_JSON = "download_info.json"

def load_download_info():
    """Load download info JSON file. If it doesn't exist, return default structure with URLs and empty checksums."""
    if os.path.exists(DOWNLOAD_INFO_JSON):
        with open(DOWNLOAD_INFO_JSON, "r") as f:
            return json.load(f)
    else:
        # Default download information with empty checksums
        return {
            "vox1_test_whatsapp_ecapa2.pth": {
                "url": "https://drive.google.com/uc?id=1c8xJ8H0aV6AIlWOElGCER9ZwWz6sg8yU",
                "checksum": ""
            },
            "vox1_meta.csv": {
                "url": "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_meta.csv?download=true",
                "checksum": ""
            }
        }

def save_download_info(info):
    """Save download info JSON to disk."""
    with open(DOWNLOAD_INFO_JSON, "w") as f:
        json.dump(info, f, indent=4)


def compute_checksum(file_path, algorithm='sha256'):
    """Compute the checksum of a file using the specified algorithm (default sha256)."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, file_name):
    """Download file from a URL to the specified file name.
    If the URL is from Google Drive, use gdown; otherwise, use requests."""
    if "drive.google" in url:
        if gdown:
            print(f"Downloading {file_name} from Google Drive...")
            gdown.download(url, file_name, quiet=False)
        else:
            raise ImportError("gdown module is not installed")
    else:
        print(f"Downloading {file_name} from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {url} - Status code: {response.status_code}")


def download_and_update(info):
    """Download files if they do not exist, compute their checksums,
    and update the info JSON with the computed checksum."""
    for file_name, file_info in info.items():
        if not os.path.exists(file_name):
            print(f"{file_name} not found. Initiating download.")
            download_file(file_info["url"], file_name)
        else:
            print(f"{file_name} already exists. Skipping download.")
        
        if os.path.exists(file_name):
            cs = compute_checksum(file_name)
            print(f"Computed checksum for {file_name}: {cs}")
            file_info["checksum"] = cs
        else:
            print(f"Warning: {file_name} does not exist even after download attempt.")
    return info


def main():
    info = load_download_info()
    updated_info = download_and_update(info)
    save_download_info(updated_info)
    print("Download info updated and saved to", DOWNLOAD_INFO_JSON)


if __name__ == "__main__":
    main() 