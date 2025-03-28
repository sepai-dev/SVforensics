import os
import logging
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import librosa
import soundfile as sf
import glob
from svforensics import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("svforensics.audioprep")

# Default paths and parameters from config
DEFAULT_DATA_DIR = config.get_path("data_dir")
DEFAULT_CASE_DIR = config.get_path("case_dir")
DEFAULT_PROBE_DIR = config.get_path("probe_dir")
DEFAULT_REFERENCE_DIR = config.get_path("reference_dir")
DEFAULT_PROCESSED_DIR = config.get_path("processed_audio_dir")
DEFAULT_PROBE_PROCESSED_DIR = config.get_path("probe_processed_dir")
DEFAULT_REFERENCE_PROCESSED_DIR = config.get_path("reference_processed_dir")

# Audio processing parameters from config
DEFAULT_SAMPLE_RATE = config.get_audio_config("sample_rate")
DEFAULT_CHUNK_DURATION = config.get_audio_config("chunk_duration")
DEFAULT_FADE_DURATION = config.get_audio_config("fade_duration")
DEFAULT_AUDIO_EXTENSIONS = config.get_audio_config("audio_extensions")

def load_audio(audio_file: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and convert to mono with target sample rate
    
    Args:
        audio_file: Path to the audio file
        target_sr: Target sample rate in Hz
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    logger.info(f"Loading audio file: {audio_file}")
    
    try:
        # Use format-specific loading based on file extension
        file_ext = os.path.splitext(audio_file)[1].lower()
        
        # For common formats that soundfile handles well (.wav, .flac, .ogg)
        if file_ext in ['.wav', '.flac', '.ogg']:
            audio, file_sr = sf.read(audio_file)
            
            # Convert to mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if file_sr != target_sr:
                from scipy import signal
                audio = signal.resample_poly(audio, target_sr, file_sr)
                
            logger.info(f"Successfully loaded audio with soundfile: {len(audio)/target_sr:.2f} seconds, {target_sr}Hz")
            return audio, target_sr
        
        # For other formats, use librosa but with modern non-deprecated approach
        else:
            # Use the modern librosa loading approach
            import librosa.core as lc
            y, sr = lc.load(path=audio_file, sr=target_sr, mono=True, offset=0.0, 
                           duration=None, dtype=np.float32, res_type='kaiser_best')
                
            logger.info(f"Successfully loaded audio: {len(y)/sr:.2f} seconds, {sr}Hz")
            return y, sr
            
    except Exception as e:
        logger.error(f"Failed to load audio file {audio_file}: {str(e)}")
        raise

def apply_fade(audio: np.ndarray, sr: int, fade_duration: float = DEFAULT_FADE_DURATION) -> np.ndarray:
    """
    Apply fade in and fade out to audio segment
    
    Args:
        audio: Audio data
        sr: Sample rate
        fade_duration: Duration of fade in/out in seconds
        
    Returns:
        Audio with fades applied
    """
    # Calculate fade length in samples
    fade_len = int(fade_duration * sr)
    
    # If audio is shorter than twice the fade length, reduce fade length
    if len(audio) < 2 * fade_len:
        fade_len = len(audio) // 4
    
    # Create linear fade in and out windows
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    
    # Apply fades
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    
    return audio

def split_into_chunks(
    audio: np.ndarray, 
    sr: int, 
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    fade_duration: float = DEFAULT_FADE_DURATION,
    min_chunk_duration: Optional[float] = None
) -> List[np.ndarray]:
    """
    Split audio into fixed-length chunks with fade in/out
    
    Args:
        audio: Audio data
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        fade_duration: Duration of fade in/out in seconds
        min_chunk_duration: Minimum duration for the last chunk in seconds (if None, uses chunk_duration/2)
        
    Returns:
        List of audio chunks
    """
    logger.info(f"Splitting audio into {chunk_duration}s chunks with {fade_duration}s fades")
    
    # Set minimum chunk duration
    if min_chunk_duration is None:
        min_chunk_duration = chunk_duration / 2
    
    # Calculate chunk length in samples
    chunk_len = int(chunk_duration * sr)
    min_chunk_samples = int(min_chunk_duration * sr)
    
    # Initialize list to store chunks
    chunks = []
    
    # Calculate number of chunks
    num_chunks = len(audio) // chunk_len
    
    # Process each chunk
    for i in range(num_chunks):
        start = i * chunk_len
        end = start + chunk_len
        chunk = audio[start:end].copy()
        chunk = apply_fade(chunk, sr, fade_duration)
        chunks.append(chunk)
    
    # Process remaining audio if it meets minimum duration
    remaining = audio[num_chunks * chunk_len:]
    if len(remaining) >= min_chunk_samples:
        remaining = apply_fade(remaining, sr, fade_duration)
        chunks.append(remaining)
    
    logger.info(f"Created {len(chunks)} chunks from audio")
    return chunks

def save_chunks(
    chunks: List[np.ndarray],
    sr: int,
    output_dir: str,
    base_filename: str,
    format: str = 'wav'
) -> List[str]:
    """
    Save audio chunks to files
    
    Args:
        chunks: List of audio chunks
        sr: Sample rate
        output_dir: Directory to save chunks
        base_filename: Base filename for chunks
        format: Audio format to save as
        
    Returns:
        List of saved file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove extension from base filename if present
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    
    # Save each chunk
    saved_files = []
    for i, chunk in enumerate(chunks):
        # Create filename with zero-padded index
        chunk_filename = f"{base_name}_{i:04d}.{format}"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Save as 16-bit PCM
        sf.write(chunk_path, chunk, sr, subtype='PCM_16')
        saved_files.append(chunk_path)
    
    logger.info(f"Saved {len(saved_files)} chunks to {output_dir}")
    return saved_files

def process_audio_file(
    audio_file: str,
    output_dir: str = DEFAULT_PROBE_PROCESSED_DIR,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    fade_duration: float = DEFAULT_FADE_DURATION,
    min_chunk_duration: Optional[float] = None,
    format: str = 'wav'
) -> List[str]:
    """
    Process a single audio file: convert to mono 16kHz and split into chunks
    
    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save chunks
        sample_rate: Target sample rate
        chunk_duration: Duration of each chunk in seconds
        fade_duration: Duration of fade in/out in seconds
        min_chunk_duration: Minimum duration for the last chunk in seconds
        format: Audio format to save as
        
    Returns:
        List of saved chunk file paths
    """
    logger.info(f"Processing audio file: {audio_file}")
    
    try:
        # Load and convert audio
        audio, sr = load_audio(audio_file, sample_rate)
        
        # Split into chunks with fade in/out
        chunks = split_into_chunks(audio, sr, chunk_duration, fade_duration, min_chunk_duration)
        
        # Save chunks
        chunk_files = save_chunks(chunks, sr, output_dir, os.path.basename(audio_file), format)
        
        return chunk_files
    except Exception as e:
        logger.error(f"Failed to process audio file {audio_file}: {str(e)}")
        raise

def process_directory(
    input_dir: str,
    output_dir: str = DEFAULT_PROBE_PROCESSED_DIR,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    fade_duration: float = DEFAULT_FADE_DURATION,
    min_chunk_duration: Optional[float] = None,
    format: str = 'wav',
    extensions: List[str] = DEFAULT_AUDIO_EXTENSIONS
) -> List[str]:
    """
    Process all audio files in a directory
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save chunks
        sample_rate: Target sample rate
        chunk_duration: Duration of each chunk in seconds
        fade_duration: Duration of fade in/out in seconds
        min_chunk_duration: Minimum duration for the last chunk in seconds
        format: Audio format to save as
        extensions: List of audio file extensions to process
        
    Returns:
        List of all saved chunk file paths
    """
    logger.info(f"Processing audio files in directory: {input_dir}")
    
    # Get list of audio files
    audio_files = []
    
    # Make sure input_dir is a string
    input_dir_str = str(input_dir)
    
    # Normalize extensions to include the dot if not present
    normalized_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Search for files with the specified extensions
    for ext in normalized_extensions:
        # Search in the specified directory and its subdirectories
        pattern = os.path.join(input_dir_str, "**", f"*{ext}")
        found_files = glob.glob(pattern, recursive=True)
        audio_files.extend(found_files)
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    all_chunks = []
    for audio_file in audio_files:
        # Create speaker-specific output directory using parent folder name
        speaker_dir = os.path.basename(os.path.dirname(audio_file))
        speaker_output_dir = os.path.join(output_dir, speaker_dir)
        
        # Process file
        chunks = process_audio_file(
            audio_file,
            speaker_output_dir,
            sample_rate,
            chunk_duration,
            fade_duration,
            min_chunk_duration,
            format
        )
        all_chunks.extend(chunks)
    
    logger.info(f"Processed {len(audio_files)} files, created {len(all_chunks)} chunks")
    return all_chunks

def process_probe_reference(
    probe_dir: str = DEFAULT_PROBE_DIR,
    reference_dir: str = DEFAULT_REFERENCE_DIR,
    output_dir: str = DEFAULT_PROCESSED_DIR,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    fade_duration: float = DEFAULT_FADE_DURATION,
    min_chunk_duration: Optional[float] = None,
    format: str = 'wav',
    extensions: List[str] = DEFAULT_AUDIO_EXTENSIONS
) -> Tuple[List[str], List[str]]:
    """
    Process both probe and reference audio files
    
    Args:
        probe_dir: Directory containing probe audio files
        reference_dir: Directory containing reference audio files
        output_dir: Base directory to save chunks
        sample_rate: Target sample rate
        chunk_duration: Duration of each chunk in seconds
        fade_duration: Duration of fade in/out in seconds
        min_chunk_duration: Minimum duration for the last chunk in seconds
        format: Audio format to save as
        extensions: List of audio file extensions to process
        
    Returns:
        Tuple of (probe_chunks, reference_chunks) paths
    """
    logger.info(f"Processing probe and reference audio files")
    
    # Get output directories from config
    probe_output_dir = config.get_path("probe_processed_dir")
    reference_output_dir = config.get_path("reference_processed_dir")
    
    # Ensure directories exist
    os.makedirs(probe_output_dir, exist_ok=True)
    os.makedirs(reference_output_dir, exist_ok=True)
    
    # Process probe files
    logger.info(f"Processing probe files from {probe_dir}")
    probe_chunks = process_directory(
        probe_dir,
        probe_output_dir,
        sample_rate,
        chunk_duration,
        fade_duration,
        min_chunk_duration,
        format,
        extensions
    )
    
    # Process reference files
    logger.info(f"Processing reference files from {reference_dir}")
    reference_chunks = process_directory(
        reference_dir,
        reference_output_dir,
        sample_rate,
        chunk_duration,
        fade_duration,
        min_chunk_duration,
        format,
        extensions
    )
    
    logger.info(f"Completed processing: {len(probe_chunks)} probe chunks, {len(reference_chunks)} reference chunks")
    return probe_chunks, reference_chunks

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess audio files for speaker verification")
    
    # Processing mode
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")
    
    # Single input mode
    single_parser = subparsers.add_parser("single", help="Process a single audio file or directory")
    single_parser.add_argument(
        "input",
        help="Input audio file or directory"
    )
    single_parser.add_argument(
        "--output-dir",
        default=DEFAULT_PROBE_PROCESSED_DIR,
        help="Output directory for audio chunks"
    )
    
    # Probe-reference mode
    pr_parser = subparsers.add_parser("probe-ref", help="Process probe and reference audio directories")
    pr_parser.add_argument(
        "--probe-dir",
        default=DEFAULT_PROBE_DIR,
        help="Directory containing probe audio files"
    )
    pr_parser.add_argument(
        "--reference-dir",
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing reference audio files"
    )
    pr_parser.add_argument(
        "--output-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Base output directory for audio chunks"
    )
    
    # Common parameters for both modes
    for subparser in [single_parser, pr_parser]:
        subparser.add_argument(
            "--sample-rate",
            type=int,
            default=DEFAULT_SAMPLE_RATE,
            help="Target sample rate in Hz"
        )
        subparser.add_argument(
            "--chunk-duration",
            type=float,
            default=DEFAULT_CHUNK_DURATION,
            help="Duration of each chunk in seconds"
        )
        subparser.add_argument(
            "--fade-duration",
            type=float,
            default=DEFAULT_FADE_DURATION,
            help="Duration of fade in/out in seconds"
        )
        subparser.add_argument(
            "--min-chunk-duration",
            type=float,
            default=None,
            help="Minimum duration for the last chunk in seconds (default: chunk_duration/2)"
        )
        subparser.add_argument(
            "--format",
            default="wav",
            choices=["wav", "flac"],
            help="Output audio format"
        )
    
    return parser.parse_args()

def cli_main(args=None):
    """Command-line interface main function."""
    import sys
    if args is not None:
        # Convert args list to command-line arguments for argparse
        sys.argv[1:] = args
    
    args = parse_args()
    
    print(f"SVforensics Audio Preparation Utility")
    print(f"====================================")
    
    if args.mode == "single":
        print(f"Mode: Single input")
        print(f"Input: {args.input}")
        print(f"Output directory: {args.output_dir}")
    elif args.mode == "probe-ref":
        print(f"Mode: Probe-Reference")
        print(f"Probe directory: {args.probe_dir}")
        print(f"Reference directory: {args.reference_dir}")
        print(f"Output directory: {args.output_dir}")
    else:
        print(f"Error: Please specify a processing mode (single or probe-ref)")
        sys.exit(1)
    
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Chunk duration: {args.chunk_duration} s")
    print(f"Fade duration: {args.fade_duration} s")
    print(f"Minimum chunk duration: {args.min_chunk_duration if args.min_chunk_duration else args.chunk_duration/2} s")
    print(f"Output format: {args.format}")
    print()
    
    try:
        if args.mode == "single":
            # Check if input is a file or directory
            if os.path.isfile(args.input):
                # Process single file
                chunks = process_audio_file(
                    args.input,
                    args.output_dir,
                    args.sample_rate,
                    args.chunk_duration,
                    args.fade_duration,
                    args.min_chunk_duration,
                    args.format
                )
                print(f"\nProcessing completed successfully!")
                print(f"Created {len(chunks)} chunks from file")
                
            elif os.path.isdir(args.input):
                # Process directory
                chunks = process_directory(
                    args.input,
                    args.output_dir,
                    args.sample_rate,
                    args.chunk_duration,
                    args.fade_duration,
                    args.min_chunk_duration,
                    args.format
                )
                print(f"\nProcessing completed successfully!")
                print(f"Created {len(chunks)} chunks from directory")
                
            else:
                print(f"Error: Input {args.input} is not a valid file or directory")
                sys.exit(1)
        
        elif args.mode == "probe-ref":
            # Process probe and reference directories
            probe_chunks, reference_chunks = process_probe_reference(
                args.probe_dir,
                args.reference_dir,
                args.output_dir,
                args.sample_rate,
                args.chunk_duration,
                args.fade_duration,
                args.min_chunk_duration,
                args.format
            )
            print(f"\nProcessing completed successfully!")
            print(f"Created {len(probe_chunks)} probe chunks and {len(reference_chunks)} reference chunks")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    cli_main() 