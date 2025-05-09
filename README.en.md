# SVforensics

A toolkit for speaker verification forensics, focusing on processing and analyzing embeddings from voice recordings.

The recommended environment for running SVforensics is Google Colab, which provides an interactive and structured interface for forensic analysis:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sepai-dev/SVforensics/blob/main/notebooks/svforensics.ipynb)

[![Methodology and Foundation](https://img.shields.io/badge/Methodology%20and%20Foundation-Google%20Docs-blue?logo=google-docs&logoColor=white)](https://docs.google.com/document/d/1PCvHK_CqQVjBnwv5hcJmPwflBAQdmsR_G-3RdNJufdo/edit?usp=sharing)

The Colab notebook provides a complete, integrated environment that manages the installation of the `svforensics` package, downloading of required resources, case audio processing, and analysis plot generation.

For instructions on local installation and command-line interface (CLI) usage, refer to our [CLI Usage Documentation](docs/cli_usage.md).

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

© 2025 Polícia Científica de Goiás. All rights reserved. 