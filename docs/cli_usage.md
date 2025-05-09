## Command-Line Interface (CLI) Usage

This document details how to use SVforensics via its command-line interface (CLI). For a simpler, guided experience, especially in cloud environments, please refer to our Google Colab Notebook (see main `README.md`).

### Development Installation

If you intend to use SVforensics from your local command line or contribute to its development, you'll need to install it locally.

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/SVforensics.git
   cd SVforensics
   ```

2. Install in development mode:
   ```
   pip install -e .
   ```

This will install the package in "editable" mode, meaning changes to the code will be immediately reflected without needing to reinstall.

### Downloading Required Files

Use the `download` command to get the necessary pretrained models and metadata.

```bash
# Most minimal command (uses all defaults from config):
svf download

# Specify output directory:
svf download --output-dir path/to/downloaded/data
```

This typically downloads:
- Speaker embedding models (e.g., ECAPA-TDNN)
- `vox1_test_whatsapp_ecapa2.pth`: Speaker embeddings for VoxCeleb1 test set (example)
- `vox1_meta.csv`: Speaker metadata for VoxCeleb1 (example)

*(Note: Actual files depend on the configuration present in config/download_info.json)*

### Processing Pipeline (Example Workflow)

Here's a typical command-line workflow using the toolkit. Many commands can work with minimal or no parameters using configured defaults.

1.  **Merge Metadata & Embeddings:** Combine downloaded embeddings with their corresponding metadata.
    ```bash
    # Most minimal command (uses all defaults from config):
    svf metadata-merge
    
    # When specifying particular files:
    svf metadata-merge --embeddings path/to/embeddings.pth --metadata path/to/metadata.csv
    ```

2.  **Generate Test Lists:** Create pairings for speaker verification trials (e.g., for male speakers).
    ```bash
    # Minimal command (gender is the only required parameter):
    svf testlists --gender m
    
    # With custom embeddings file:
    svf testlists --gender m --embeddings path/to/embeddings.pth
    ```

3.  **Prepare Case Audio:** Preprocess audio files (VAD, chunking). This is a necessary step for custom audio analysis.
    ```bash
    # Most minimal command:
    svf audio-prep probe-ref
    
    # With custom directories:
    svf audio-prep probe-ref --probe-dir path/to/probe --reference-dir path/to/reference
    ```

4.  **Extract Case Embeddings:** Extract embeddings from the preprocessed audio chunks. This is a necessary step for custom audio analysis.
    ```bash
    # Most minimal command:
    svf case-embed probe-ref
    
    # With custom directories:
    svf case-embed probe-ref --probe-dir path/to/probe --reference-dir path/to/reference
    ```

5.  **Perform Verification:** Run the speaker verification analysis using a test list and relevant embeddings.
    ```bash
    # Most minimal command:
    svf verify
    
    # When specifying custom files:
    svf verify --test-list path/to/test_list.txt --probe-embeddings path/to/probe.pth --reference-embeddings path/to/reference.pth
    ```

*Use the `--help` flag with each command (e.g., `svf metadata-merge --help`) for detailed options.* 