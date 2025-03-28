"""
Case audio speaker embedding extraction using ECAPA2.

This module provides functions to extract speaker embeddings from case audio files
using the ECAPA2 speaker embedding model. It specifically handles the speaker verification
case workflow, processing probe and reference audio chunks to generate embeddings
for speaker verification and forensic comparisons.
"""

import os
import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from svforensics import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("svforensics.case_embeddings")

# Default paths from config
DEFAULT_PROCESSED_DIR = config.get_path("processed_audio_dir")
DEFAULT_EMBEDDINGS_DIR = config.get_path("embeddings_dir")
DEFAULT_PROBE_PROCESSED_DIR = config.get_path("probe_processed_dir")
DEFAULT_REFERENCE_PROCESSED_DIR = config.get_path("reference_processed_dir")
DEFAULT_PROBE_EMBEDDINGS_FILE = config.get_path("probe_embeddings_file")
DEFAULT_REFERENCE_EMBEDDINGS_FILE = config.get_path("reference_embeddings_file")

# Default model parameters from config
DEFAULT_MODEL_REPO = config.get_model_config("repository")
DEFAULT_MODEL_FILENAME = config.get_model_config("filename")
DEFAULT_CACHE_DIR = config.get_path("models_cache_dir")

class EmbeddingExtractor:
    """
    Speaker embedding extractor using ECAPA2 model for case audio processing.
    
    This class provides methods to extract speaker embeddings from case audio files.
    The model is loaded once and reused for multiple extractions.
    """
    
    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        model_filename: str = DEFAULT_MODEL_FILENAME,
        cache_dir: str = DEFAULT_CACHE_DIR,
        use_half_precision: bool = False
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_repo: HuggingFace repository ID
            model_filename: Model filename in the repository
            cache_dir: Directory to cache the downloaded model
            use_half_precision: Whether to use half precision (FP16) for faster inference
        """
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.use_half_precision = use_half_precision
        
        # Lazy loading - model will be loaded on first use
        self._model = None
        self._device = None
    
    @property
    def device(self) -> torch.device:
        """Get the device (CPU/GPU) being used."""
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    @property
    def model(self) -> torch.jit.ScriptModule:
        """Get the ECAPA2 model, loading it if necessary."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the ECAPA2 model from HuggingFace Hub."""
        logger.info(f"Loading ECAPA2 model from {self.model_repo}")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Download model if not already cached
        try:
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_filename,
                cache_dir=self.cache_dir
            )
            logger.info(f"Model downloaded to {model_path}")
            
            # Load the model
            self._model = torch.jit.load(model_path, map_location=self.device)
            
            # Optionally use half precision for faster inference
            if self.use_half_precision and self.device.type == 'cuda':
                logger.info("Using half precision (FP16) for faster inference")
                self._model.half()
            
            # Put model in evaluation mode
            self._model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extract embedding from a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            torch.Tensor: Speaker embedding
        """
        # Ensure model is loaded
        model = self.model
        
        try:
            # Load audio file
            audio, sr = torchaudio.load(audio_path)
            
            # Move to the same device as the model
            audio = audio.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = model(audio)
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to extract embedding from {audio_path}: {str(e)}")
            raise
    
    def extract_embeddings_from_directory(
        self, 
        audio_dir: str,
        recursive: bool = True,
        extensions: List[str] = [".wav", ".flac"],
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            recursive: Whether to search recursively in subdirectories
            extensions: List of audio file extensions to process
            show_progress: Whether to show a progress bar
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping file paths to embeddings
        """
        logger.info(f"Extracting embeddings from audio files in {audio_dir}")
        
        # Find all audio files
        audio_files = []
        
        # Normalize extensions
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        
        # Search for audio files
        base_path = Path(audio_dir)
        for ext in extensions:
            if recursive:
                audio_files.extend(list(base_path.rglob(f"*{ext}")))
            else:
                audio_files.extend(list(base_path.glob(f"*{ext}")))
        
        audio_files = [str(f) for f in audio_files]
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir} with extensions {extensions}")
            return {}
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Extract embeddings
        embeddings = {}
        
        # Create iterator with optional progress bar
        iterator = tqdm(audio_files, desc="Extracting embeddings") if show_progress else audio_files
        
        for audio_file in iterator:
            try:
                embedding = self.extract_embedding(audio_file)
                embeddings[audio_file] = embedding
            except Exception as e:
                logger.warning(f"Failed to extract embedding from {audio_file}: {str(e)}")
                continue
        
        logger.info(f"Successfully extracted {len(embeddings)} embeddings")
        return embeddings
    
    def extract_speaker_embeddings(
        self,
        audio_dir: str,
        output_file: Optional[str] = None,
        recursive: bool = True,
        extensions: List[str] = [".wav", ".flac"],
        show_progress: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract embeddings for each speaker in a directory structure.
        
        Assumes a directory structure where speaker IDs are subdirectory names:
        audio_dir/
          speaker1/
            file1.wav
            file2.wav
          speaker2/
            file1.wav
            ...
        
        Args:
            audio_dir: Directory containing speaker subdirectories
            output_file: Optional path to save embeddings as a PyTorch file
            recursive: Whether to search recursively within speaker directories
            extensions: List of audio file extensions to process
            show_progress: Whether to show a progress bar
            
        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Nested dictionary mapping speaker IDs to 
            dictionaries of file paths and their embeddings
        """
        logger.info(f"Extracting speaker embeddings from {audio_dir}")
        
        # Get all subdirectories (potential speaker directories)
        speaker_dirs = [d for d in os.listdir(audio_dir) 
                       if os.path.isdir(os.path.join(audio_dir, d))]
        
        if not speaker_dirs:
            logger.warning(f"No speaker directories found in {audio_dir}")
            return {}
        
        # Extract embeddings for each speaker
        speaker_embeddings = {}
        
        # Create iterator with optional progress bar
        iterator = tqdm(speaker_dirs, desc="Processing speakers") if show_progress else speaker_dirs
        
        for speaker_id in iterator:
            speaker_dir = os.path.join(audio_dir, speaker_id)
            
            # Extract embeddings for this speaker
            embeddings = self.extract_embeddings_from_directory(
                speaker_dir,
                recursive=recursive,
                extensions=extensions,
                show_progress=False  # Avoid nested progress bars
            )
            
            if embeddings:
                speaker_embeddings[speaker_id] = embeddings
        
        logger.info(f"Extracted embeddings for {len(speaker_embeddings)} speakers")
        
        # Save embeddings if output file specified
        if output_file and speaker_embeddings:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            torch.save(speaker_embeddings, output_file)
            logger.info(f"Saved speaker embeddings to {output_file}")
        
        return speaker_embeddings
    
    def extract_probe_reference_embeddings(
        self,
        probe_dir: str = DEFAULT_PROBE_PROCESSED_DIR,
        reference_dir: str = DEFAULT_REFERENCE_PROCESSED_DIR,
        output_dir: str = DEFAULT_EMBEDDINGS_DIR,
        recursive: bool = True,
        extensions: List[str] = [".wav", ".flac"],
        show_progress: bool = True
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Extract embeddings from both probe and reference directories.
        
        Args:
            probe_dir: Directory containing processed probe audio files
            reference_dir: Directory containing processed reference audio files
            output_dir: Directory to save embeddings
            recursive: Whether to search recursively in speaker directories
            extensions: List of audio file extensions to process
            show_progress: Whether to show a progress bar
            
        Returns:
            Tuple of (probe_embeddings, reference_embeddings)
        """
        logger.info(f"Extracting embeddings from probe and reference directories")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output files
        probe_output_file = DEFAULT_PROBE_EMBEDDINGS_FILE
        reference_output_file = DEFAULT_REFERENCE_EMBEDDINGS_FILE
        
        # Extract probe embeddings
        logger.info(f"Processing probe directory: {probe_dir}")
        probe_embeddings = self.extract_speaker_embeddings(
            probe_dir,
            output_file=probe_output_file,
            recursive=recursive,
            extensions=extensions,
            show_progress=show_progress
        )
        
        # Extract reference embeddings
        logger.info(f"Processing reference directory: {reference_dir}")
        reference_embeddings = self.extract_speaker_embeddings(
            reference_dir,
            output_file=reference_output_file,
            recursive=recursive,
            extensions=extensions,
            show_progress=show_progress
        )
        
        return probe_embeddings, reference_embeddings


def load_embeddings(embeddings_file: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load embeddings from a PyTorch file.
    
    Args:
        embeddings_file: Path to the embeddings file
        
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Loaded embeddings
    """
    try:
        embeddings = torch.load(embeddings_file)
        logger.info(f"Loaded embeddings from {embeddings_file}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings from {embeddings_file}: {str(e)}")
        raise


def main():
    """Command-line interface main function for case audio embedding extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract speaker embeddings from case audio files for forensic comparison")
    
    # Processing mode
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")
    
    # Single directory mode
    single_parser = subparsers.add_parser("single", help="Process a single directory of case audio")
    single_parser.add_argument(
        "input_dir",
        help="Input directory containing case audio files or speaker subdirectories"
    )
    single_parser.add_argument(
        "--output-file",
        help="Output file path to save embeddings"
    )
    single_parser.add_argument(
        "--by-speaker",
        action="store_true",
        help="Whether the input directory contains speaker subdirectories"
    )
    
    # Probe-reference mode
    pr_parser = subparsers.add_parser("probe-ref", help="Process probe and reference case audio directories")
    pr_parser.add_argument(
        "--probe-dir",
        default=DEFAULT_PROBE_PROCESSED_DIR,
        help="Directory containing processed probe audio chunks"
    )
    pr_parser.add_argument(
        "--reference-dir",
        default=DEFAULT_REFERENCE_PROCESSED_DIR,
        help="Directory containing processed reference audio chunks"
    )
    pr_parser.add_argument(
        "--output-dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory to save case embeddings"
    )
    
    # Common parameters for both modes
    for subparser in [single_parser, pr_parser]:
        subparser.add_argument(
            "--recursive",
            action="store_true",
            default=True,
            help="Search recursively in directories"
        )
        subparser.add_argument(
            "--extensions",
            nargs="+",
            default=[".wav", ".flac"],
            help="Audio file extensions to process"
        )
        subparser.add_argument(
            "--model-repo",
            default=DEFAULT_MODEL_REPO,
            help="HuggingFace repository ID"
        )
        subparser.add_argument(
            "--model-filename",
            default=DEFAULT_MODEL_FILENAME,
            help="Model filename in the repository"
        )
        subparser.add_argument(
            "--cache-dir",
            default=DEFAULT_CACHE_DIR,
            help="Directory to cache the downloaded model"
        )
        subparser.add_argument(
            "--half-precision",
            action="store_true",
            help="Use half precision (FP16) for faster inference"
        )
        subparser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable progress bar"
        )
    
    args = parser.parse_args()
    
    print(f"SVforensics Case Audio Embedding Extraction")
    print(f"=========================================")
    
    if args.mode == "single":
        print(f"Mode: Single directory")
        print(f"Input directory: {args.input_dir}")
        print(f"Output file: {args.output_file}")
        print(f"By speaker: {args.by_speaker}")
    elif args.mode == "probe-ref":
        print(f"Mode: Probe-Reference")
        print(f"Probe directory: {args.probe_dir}")
        print(f"Reference directory: {args.reference_dir}")
        print(f"Output directory: {args.output_dir}")
    else:
        print(f"Error: Please specify a processing mode (single or probe-ref)")
        return
    
    print(f"Model repository: {args.model_repo}")
    print(f"Model filename: {args.model_filename}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Half precision: {args.half_precision}")
    print(f"Recursive: {args.recursive}")
    print(f"Extensions: {args.extensions}")
    print(f"Show progress: {not args.no_progress}")
    print()
    
    try:
        # Initialize the embedding extractor
        extractor = EmbeddingExtractor(
            model_repo=args.model_repo,
            model_filename=args.model_filename,
            cache_dir=args.cache_dir,
            use_half_precision=args.half_precision
        )
        
        if args.mode == "single":
            if args.by_speaker:
                # Extract embeddings for each speaker in the directory
                embeddings = extractor.extract_speaker_embeddings(
                    args.input_dir,
                    output_file=args.output_file,
                    recursive=args.recursive,
                    extensions=args.extensions,
                    show_progress=not args.no_progress
                )
                print(f"\nSuccessfully extracted embeddings for {len(embeddings)} speakers")
            else:
                # Extract embeddings from all audio files in the directory
                embeddings = extractor.extract_embeddings_from_directory(
                    args.input_dir,
                    recursive=args.recursive,
                    extensions=args.extensions,
                    show_progress=not args.no_progress
                )
                
                # Save embeddings if output file specified
                if args.output_file and embeddings:
                    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
                    torch.save(embeddings, args.output_file)
                    print(f"\nSaved embeddings to {args.output_file}")
                
                print(f"\nSuccessfully extracted {len(embeddings)} embeddings")
                
        elif args.mode == "probe-ref":
            # Extract embeddings from probe and reference directories
            probe_embeddings, reference_embeddings = extractor.extract_probe_reference_embeddings(
                probe_dir=args.probe_dir,
                reference_dir=args.reference_dir,
                output_dir=args.output_dir,
                recursive=args.recursive,
                extensions=args.extensions,
                show_progress=not args.no_progress
            )
            
            # Print summary
            # Count total number of embeddings across all subdirectories
            total_probe_files = sum(len(files) for files in probe_embeddings.values())
            total_ref_files = sum(len(files) for files in reference_embeddings.values())
            
            print(f"\nProcessing complete!")
            print(f"Extracted {total_probe_files} probe embeddings")
            print(f"Extracted {total_ref_files} reference embeddings")
            print(f"Saved embeddings to {args.output_dir}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def cli_main(args=None):
    """Entry point for the case embeddings CLI tool."""
    import sys
    if args is not None:
        # Convert args list to command-line arguments for argparse
        sys.argv[1:] = args
    return main()

if __name__ == "__main__":
    main() 