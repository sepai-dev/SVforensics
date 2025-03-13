# SVforensics

A toolkit for speaker verification forensics, focusing on processing and analyzing embeddings from voice recordings.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download required files:
   ```
   python download_files.py
   ```

## Features

### Downloading Required Files

The `download_files.py` script downloads the necessary files for analysis:
- `vox1_test_whatsapp_ecapa2.pth`: Speaker embeddings
- `vox1_meta.csv`: Speaker metadata

### Processing Embeddings

The `process_embeddings.py` script processes speaker embeddings and merges them with metadata:

```python
# Example usage
import process_embeddings

# Process with default settings
df = process_embeddings.process_pipeline()

# Process with custom settings
df = process_embeddings.process_pipeline(
    embedding_file='path/to/embeddings.pth',
    metadata_file='path/to/metadata.csv',
    test_file='path/to/veri_test.txt'
)

# Save processed data
df.to_csv('processed_embeddings.csv', index=False)
```

## Data Structure

The processed DataFrame contains:
- `file_path`: Path to the audio file
- `class_id`: Speaker ID
- `video_id`: Video ID
- `embedding`: Speaker embedding vector
- `genre`: Speaker gender (m/f)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## TODO: 
- dividir a pop de ref em ref_padrao e ref_questionado (colocar esta informação no dataframe)
- calcular os scores da pop de ref (devidamente filtrado, por genero e por subset padrao e questionado)
- gerar test lists
- preprocessar os audios do caso e extrair seus respectivos embs