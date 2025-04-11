# SVforensics

A toolkit for speaker verification forensics, focusing on processing and analyzing embeddings from voice recordings.

## Installation

### Development Installation

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

## Usage

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

### Google Colab Usage

A Google Colab notebook is provided in the `notebooks/` directory (`notebooks/SVForensics.ipynb`) demonstrating a typical workflow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sepai-dev/SVforensics/blob/main/notebooks/svforensics.ipynb)

Click the badge above to open the notebook directly in Google Colab.

The notebook handles the installation of the `svforensics` package and guides you through downloading necessary files, uploading case audio, processing, and generating the analysis plot directly within the Colab environment.

## Project Structure

```
SVforensics/
├── data/                  # Default data directory
│   ├── raw/               # Downloaded raw data
│   └── processed/         # Processed data outputs
├── svforensics/           # Main package source code
│   ├── __init__.py        # Package initializer, exposes primary modules
│   ├── __main__.py        # Main CLI entry point (`svf` command)
│   ├── audioprep.py       # Audio preprocessing (VAD, chunking)
│   ├── case_embeddings.py # Embedding extraction for custom case audio
│   ├── config.py          # Configuration loading and management
│   ├── download.py        # Data downloading functionality
│   ├── embeddings.py      # Core embedding model loading and extraction
│   ├── metadata_embedding_merge.py # Merging embeddings with metadata CSV
│   ├── similarity.py      # Similarity/distance calculation functions
│   ├── testlists.py       # Generation of test lists (pairs) for verification
│   ├── verification.py    # Speaker verification scoring and analysis logic
│   └── __pycache__/       # Python cache files (usually ignored)
├── tests/                 # Unit and integration tests
├── pyproject.toml         # Package configuration
└── README.md              # This file
```

## How It Works

The project follows a clean design where each core module serves dual purposes:
1. **Library functionality** - Functions that can be imported and used programmatically
2. **Command-line interface** - Each module also provides a CLI entry point

This unified approach makes the code more maintainable while still offering both API and command-line usage patterns.

## License

This project was developed by Rafaello Virgilli ([@rvirgilli](https://github.com/rvirgilli)) and Lucas Alcântara Souza ([@lucasalcs](https://github.com/lucasalcs)) as part of their official duties at [Polícia Científica de Goiás](https://www.policiacientifica.go.gov.br/).

This software is licensed under the Brazilian Public Software License (LPS Brasil) Version 3.0.

This license ensures that the software:
- Can be freely used, modified, and distributed
- Must maintain attribution to the original authors and the Polícia Científica de Goiás
- Must make the source code available to all users
- Cannot be incorporated into proprietary software
- Must preserve the same freedoms in all derivative works

The official license text is available in Portuguese (LICENSE) with an English translation provided for reference (LICENSE.en).

© 2023-2024 Polícia Científica de Goiás. All rights reserved.

## Detailed TODO / Future Enhancements:

1.  **Refine Different-Speaker Score Distribution Calculation:**
    *   **Goal:** Generate a more case-specific score distribution for the "different speakers" hypothesis.
    *   **Current Method:** Compares scores between different speakers *within* the reference population (e.g., VoxCeleb).
    *   **Proposed Method:** Calculate similarity scores by comparing the embeddings from the *actual probe audio chunks* against the embeddings of *all speakers* in the reference population dataset.
    *   **Benefit:** Provides a distribution tailored to the probe voice, showing how it compares against a known, diverse set of speakers. This can strengthen the interpretation when comparing the probe-reference score against this distribution. The same-speaker distribution calculation (comparing different utterances of the same speaker *within* the reference population) will remain unchanged initially.

2.  **Refine Same-Speaker Score Distribution Calculation (Conditional):**
    *   **Goal:** Generate a more case-specific score distribution for the "same speaker" hypothesis, *if sufficient data is available*.
    *   **Condition:** Requires having *multiple reference audio recordings* confirmed to be from the *same individual* as the primary reference sample in the case.
    *   **Proposed Method (if condition met):** Calculate similarity scores by comparing embeddings *between the different available reference recordings* of the claimed speaker.
    *   **Benefit:** Models the score variability specifically for that individual based on their own recordings, potentially improving accuracy over using the general population variability.
    *   **Fallback:** If only one reference recording exists, continue using the standard method (scores from same-speaker comparisons within the general reference population).

3.  **Improve the Reference Population:**
    *   **Goal:** Enhance the quality, diversity, size, and relevance of the background reference population dataset used for generating score distributions.
    *   **Potential Methods:**
        *   *Data Augmentation:* Apply noise, reverberation, or other transformations to existing reference audio to better match case audio conditions.
        *   *Cohort Selection:* Implement strategies to select a subset of the reference population that is acoustically closer (e.g., similar channel, noise characteristics) to the case audio.
        *   *Acquire/Integrate Diverse Data:* Incorporate additional speaker datasets reflecting a wider range of languages, accents, recording qualities, etc., relevant to forensic scenarios.
        *   *Quality Control:* Implement stricter filtering of the reference population to remove low-quality or potentially mislabeled samples.
    *   **Benefit:** A more robust and relevant reference population leads to more reliable score distributions and more confident verification conclusions.