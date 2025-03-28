import os
import pandas as pd
import torch
import numpy as np
import logging
import csv
import sys
import argparse
from typing import List, Dict, Union, Optional, Tuple, Any
from svforensics import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use paths from config module
DEFAULT_DATA_DIR = config.get_path("data_dir")
DEFAULT_DOWNLOADS_DIR = config.get_path("downloads_dir")
DEFAULT_PROCESSED_DIR = config.get_path("processed_dir")

DEFAULT_EMBEDDING_FILE = config.get_path("raw_embeddings_file")
DEFAULT_METADATA_FILE = config.get_path("metadata_file")
DEFAULT_OUTPUT_PREFIX = config.get_path("output_prefix")

# Use processing config
DEFAULT_DROP_COLUMNS = config.get_processing_config("drop_columns")

def load_embeddings(embedding_file: str = DEFAULT_EMBEDDING_FILE) -> Dict[str, torch.Tensor]:
    """
    Load embeddings from a PyTorch file (.pth)
    
    Args:
        embedding_file: Path to the embeddings file (.pth)
        
    Returns:
        Dictionary mapping file paths to embeddings
    """
    logger.info(f"Loading embeddings from {embedding_file}")
    
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    
    try:
        embeddings_dict = torch.load(embedding_file)
        logger.info(f"Successfully loaded {len(embeddings_dict)} embeddings")
        return embeddings_dict
    except Exception as e:
        logger.error(f"Failed to load embeddings: {str(e)}")
        raise

def process_embeddings_to_dataframe(embeddings_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Process embeddings dictionary into a pandas DataFrame
    
    Args:
        embeddings_dict: Dictionary mapping file paths to embeddings
        
    Returns:
        DataFrame with file_path, class_id, video_id, and embedding columns
    """
    logger.info("Processing embeddings into DataFrame")
    
    file_paths, class_ids, video_ids, embeddings = [], [], [], []
    
    for file_path, embedding in embeddings_dict.items():
        parts = file_path.split('/')
        # Assuming the format is class_id/video_id/...
        class_id, video_id = parts[0], parts[1]
        file_paths.append(file_path)
        class_ids.append(class_id)
        video_ids.append(video_id)
        embeddings.append(embedding)
    
    # Create DataFrame with embedding as the last column
    df = pd.DataFrame({
        'file_path': file_paths,
        'class_id': class_ids,
        'video_id': video_ids,
        'embedding': embeddings
    })
    
    # Ensure embedding is the last column
    columns = [col for col in df.columns if col != 'embedding'] + ['embedding']
    df = df[columns]
    
    logger.info(f"Created DataFrame with {len(df)} rows")
    return df

def load_metadata(metadata_file: str = DEFAULT_METADATA_FILE) -> pd.DataFrame:
    """
    Load metadata from a CSV file
    
    Args:
        metadata_file: Path to the metadata CSV file
        
    Returns:
        DataFrame with the metadata
    """
    logger.info(f"Loading metadata from {metadata_file}")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    try:
        metadata_df = pd.read_csv(metadata_file, sep='\t', skiprows=1, header=None)
        metadata_df.columns = ['class_id', 'name', 'genre', 'country', 'subset']
        logger.info(f"Successfully loaded metadata with {len(metadata_df)} rows")
        return metadata_df
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        raise

def merge_embeddings_with_metadata(
    embeddings_df: pd.DataFrame, 
    metadata_df: pd.DataFrame,
    drop_columns: List[str] = None
) -> pd.DataFrame:
    """
    Merge embeddings DataFrame with metadata DataFrame
    
    Args:
        embeddings_df: DataFrame with embeddings
        metadata_df: DataFrame with metadata
        drop_columns: List of columns to drop from the merged DataFrame
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging embeddings with metadata")
    
    if drop_columns is None:
        drop_columns = ['name', 'country', 'subset']
    
    merged_df = pd.merge(embeddings_df, metadata_df, on='class_id')
    
    if drop_columns:
        merged_df = merged_df.drop(columns=drop_columns)
    
    # Ensure embedding is the last column again after merge
    columns = [col for col in merged_df.columns if col != 'embedding'] + ['embedding']
    merged_df = merged_df[columns]
    
    logger.info(f"Created merged DataFrame with {len(merged_df)} rows")
    return merged_df

def load_test_paths(test_file: str) -> List[str]:
    """
    Load test paths from a verification test file
    
    Args:
        test_file: Path to the verification test file
        
    Returns:
        List of unique file paths
    """
    logger.info(f"Loading test paths from {test_file}")
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    try:
        test_df = pd.read_csv(test_file, sep=' ', header=None, names=['Label', 'Path_1', 'Path_2'])
        unique_paths = pd.concat([test_df['Path_1'], test_df['Path_2']]).unique()
        logger.info(f"Successfully loaded {len(unique_paths)} unique test paths")
        return unique_paths.tolist()
    except Exception as e:
        logger.error(f"Failed to load test paths: {str(e)}")
        raise

def filter_by_test_paths(df: pd.DataFrame, test_paths: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame by test paths
    
    Args:
        df: DataFrame to filter
        test_paths: List of test paths to include
        
    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering DataFrame by test paths")
    
    filtered_df = df[df['file_path'].isin(test_paths)]
    logger.info(f"Filtered DataFrame has {len(filtered_df)} rows")
    return filtered_df

def process_pipeline(
    embedding_file: str = DEFAULT_EMBEDDING_FILE, 
    metadata_file: str = DEFAULT_METADATA_FILE,
    test_file: Optional[str] = None,
    drop_columns: List[str] = None
) -> pd.DataFrame:
    """
    Run the full processing pipeline
    
    Args:
        embedding_file: Path to the embeddings file
        metadata_file: Path to the metadata file
        test_file: Path to the test file (optional)
        drop_columns: List of columns to drop from the merged DataFrame
        
    Returns:
        Processed DataFrame
    """
    logger.info("Starting processing pipeline")
    
    # Load embeddings
    embeddings_dict = load_embeddings(embedding_file)
    
    # Process embeddings to DataFrame
    embeddings_df = process_embeddings_to_dataframe(embeddings_dict)
    
    # Load metadata
    metadata_df = load_metadata(metadata_file)
    
    # Merge embeddings with metadata
    merged_df = merge_embeddings_with_metadata(embeddings_df, metadata_df, drop_columns)
    
    # Filter by test paths if a test file is provided
    if test_file:
        if os.path.exists(test_file):
            test_paths = load_test_paths(test_file)
            merged_df = filter_by_test_paths(merged_df, test_paths)
        else:
            logger.warning(f"Test file not found: {test_file}. Skipping filtering.")
    
    logger.info("Processing pipeline completed")
    return merged_df

def save_processed_data(df: pd.DataFrame, output_prefix: str) -> str:
    """
    Save processed data in PyTorch (.pth) format
    
    Args:
        df: The DataFrame with embeddings to save
        output_prefix: The file path prefix for output files (without extension)
    
    Returns:
        Path to the saved .pth file
    """
    # Create a dictionary to store both embeddings and metadata
    save_dict = {
        'embeddings': dict(zip(df['file_path'], df['embedding'])),
        'metadata': df.drop(columns=['embedding']).to_dict('records')
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Save everything to PyTorch .pth file
    output_path = f"{output_prefix}.pth"
    torch.save(save_dict, output_path)
    logger.info(f"All data (embeddings and metadata) saved to {output_path}")
    
    return output_path

def main(
    embedding_file: str = DEFAULT_EMBEDDING_FILE,
    metadata_file: str = DEFAULT_METADATA_FILE,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    test_file: Optional[str] = None,
    drop_columns: List[str] = None,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Main function to run the script
    
    Args:
        embedding_file: Path to the embeddings file
        metadata_file: Path to the metadata file
        output_prefix: Prefix for output files (without extension)
        test_file: Path to the test file (optional)
        drop_columns: List of columns to drop from the merged DataFrame
        save_output: Whether to save the output
        
    Returns:
        Processed DataFrame
    """
    try:
        # Process with provided settings
        df = process_pipeline(
            embedding_file=embedding_file,
            metadata_file=metadata_file,
            test_file=test_file,
            drop_columns=drop_columns
        )
        
        # Display results
        print("\nProcessed DataFrame sample:")
        print(df[df.columns.drop('embedding') if 'embedding' in df.columns else df.columns].head())
        print(f"\nDataFrame shape: {df.shape}")
        
        # Save the processed data if requested
        if save_output:
            output_path = save_processed_data(df, output_prefix)
            logger.info(f"All data saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process speaker embeddings for SVforensics")
    
    # Input files
    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDING_FILE,
        help="Path to the embeddings file (.pth)"
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_FILE,
        help="Path to the metadata file (.csv)"
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Optional path to a test file with paths to filter by"
    )
    
    # Output
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for output files (without extension)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the processed data to files"
    )
    
    # Columns to drop
    parser.add_argument(
        "--drop-columns",
        nargs="+",
        default=["name", "country", "subset"],
        help="Columns to drop from the merged DataFrame"
    )
    
    return parser.parse_args(args)

def cli_main(args=None):
    """Command-line interface main function."""
    if args is None:
        args = parse_args()
    else:
        # Parse the provided args
        import shlex
        args = parse_args(shlex.split(args) if isinstance(args, str) else args)
    
    print(f"SVforensics processing utility")
    print(f"=============================")
    print(f"Embeddings file: {args.embeddings}")
    print(f"Metadata file: {args.metadata}")
    if args.test_file:
        print(f"Test file: {args.test_file}")
    print(f"Output prefix: {args.output_prefix if not args.no_save else 'None (not saving)'}")
    print(f"Columns to drop: {args.drop_columns}")
    print()
    
    try:
        # Run the processing pipeline
        df = main(
            embedding_file=args.embeddings,
            metadata_file=args.metadata,
            output_prefix=args.output_prefix,
            test_file=args.test_file,
            drop_columns=args.drop_columns,
            save_output=not args.no_save
        )
        
        print("\nProcessing completed successfully!")
        if not args.no_save:
            print(f"All data saved to: {args.output_prefix}.pth")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main() 