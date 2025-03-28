import os
import json
import hashlib
import requests
import gdown
import logging
import argparse
import sys
from pathlib import Path
from svforensics import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use paths from config module
try:
    DEFAULT_CONFIG_PATH = config.get_path("download_info_file")
    DEFAULT_DOWNLOAD_DIR = config.get_path("downloads_dir")
except (FileNotFoundError, KeyError) as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    logger.error("Please ensure that the configuration file exists and contains the required values.")
    if __name__ == "__main__":
        sys.exit(1)
    else:
        # For module imports, use sensible defaults but log a warning
        DEFAULT_CONFIG_PATH = "config/download_info.json"
        DEFAULT_DOWNLOAD_DIR = "files/downloads"
        logger.warning(f"Using default values due to configuration error: {DEFAULT_CONFIG_PATH}, {DEFAULT_DOWNLOAD_DIR}")

def load_download_info(config_path=DEFAULT_CONFIG_PATH):
    """
    Load download info JSON file. If it doesn't exist, return default structure with URLs and empty checksums.
    
    Args:
        config_path: Path to the download_info.json file
        
    Returns:
        Dictionary with download information
    """
    logger.info(f"Loading download info from: {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"Config file {config_path} not found. Using default values.")
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


def save_download_info(info, config_path=DEFAULT_CONFIG_PATH):
    """
    Save download info JSON to disk.
    
    Args:
        info: Dictionary with download information
        config_path: Path to save the download_info.json file
    """
    logger.info(f"Saving download info to: {config_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(info, f, indent=4)


def compute_checksum(file_path, algorithm='sha256'):
    """
    Compute the checksum of a file using the specified algorithm (default sha256).
    
    Args:
        file_path: Path to the file to compute checksum for
        algorithm: Hashing algorithm to use
        
    Returns:
        Hexadecimal string of the file's checksum
    """
    logger.info(f"Computing {algorithm} checksum for {file_path}")
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, file_path):
    """
    Download file from a URL to the specified file path.
    If the URL is from Google Drive, use gdown; otherwise, use requests.
    
    Args:
        url: URL to download from
        file_path: Path where to save the downloaded file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if "drive.google" in url:
        if gdown:
            logger.info(f"Downloading {file_path} from Google Drive...")
            gdown.download(url, file_path, quiet=False)
        else:
            raise ImportError("gdown module is not installed")
    else:
        logger.info(f"Downloading {file_path} from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            error_msg = f"Failed to download {url} - Status code: {response.status_code}"
            logger.error(error_msg)
            raise Exception(error_msg)


def download_and_update(info, download_dir=DEFAULT_DOWNLOAD_DIR):
    """
    Download files if they do not exist, compute their checksums,
    and update the info JSON with the computed checksum.
    
    Args:
        info: Dictionary with download information
        download_dir: Directory where to save downloaded files
        
    Returns:
        Updated info dictionary with checksums
    """
    for file_name, file_info in info.items():
        file_path = os.path.join(download_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.info(f"{file_path} not found. Initiating download.")
            download_file(file_info["url"], file_path)
        else:
            logger.info(f"{file_path} already exists. Skipping download.")
        
        if os.path.exists(file_path):
            cs = compute_checksum(file_path)
            logger.info(f"Computed checksum for {file_name}: {cs}")
            file_info["checksum"] = cs
        else:
            logger.warning(f"Warning: {file_path} does not exist even after download attempt.")
    return info


def main(config_path=DEFAULT_CONFIG_PATH, download_dir=DEFAULT_DOWNLOAD_DIR):
    """
    Main function to run the download process.
    
    Args:
        config_path: Path to the download_info.json file
        download_dir: Directory where to save downloaded files
        
    Returns:
        Updated info dictionary with checksums
    """
    logger.info("Starting download process")
    
    # Ensure directories exist
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    info = load_download_info(config_path)
    updated_info = download_and_update(info, download_dir)
    save_download_info(updated_info, config_path)
    
    logger.info(f"Download info updated and saved to {config_path}")
    return updated_info

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download required files for SVforensics")
    
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the download configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory where to save downloaded files"
    )
    
    return parser.parse_args()

def cli_main(args=None):
    """Command-line interface main function."""
    import sys
    if args is not None:
        # Convert args list to command-line arguments for argparse
        sys.argv[1:] = args
    
    args = parse_args()
    
    print(f"SVforensics download utility")
    print(f"==========================")
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run the download process
        updated_info = main(
            config_path=args.config,
            download_dir=args.output_dir
        )
        
        print("\nDownload process completed successfully!")
        print(f"Files downloaded and saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main() 