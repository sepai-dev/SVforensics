import os
import torch
import pandas as pd
import numpy as np
import logging
import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from svforensics import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use paths from config module
DEFAULT_DATA_DIR = config.get_path("data_dir")
DEFAULT_PROCESSED_DIR = config.get_path("processed_dir")
DEFAULT_EMBEDDINGS_FILE = config.get_path("processed_embeddings_file")
DEFAULT_TEST_LIST_PREFIX = config.get_path("test_list_prefix")

# Use testlists config
DEFAULT_N_POS = config.get_testlists_config("n_pos")
DEFAULT_N_NEG = config.get_testlists_config("n_neg")
DEFAULT_DIFFERENT_VIDEOS = config.get_testlists_config("different_videos")
DEFAULT_TEST_PROP = config.get_testlists_config("test_prop")
DEFAULT_RANDOM_SEED = config.get_testlists_config("random_seed")

def load_processed_data(embeddings_file: str = DEFAULT_EMBEDDINGS_FILE) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Load the processed embeddings and metadata
    
    Args:
        embeddings_file: Path to the processed embeddings file (.pth)
        
    Returns:
        Tuple of (embeddings_dict, metadata_df)
    """
    logger.info(f"Loading processed data from {embeddings_file}")
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Processed embeddings file not found: {embeddings_file}")
    
    try:
        saved_data = torch.load(embeddings_file)
        embeddings_dict = saved_data['embeddings']
        metadata_records = saved_data['metadata']
        
        # Convert metadata records to DataFrame
        metadata_df = pd.DataFrame(metadata_records)
        
        logger.info(f"Successfully loaded {len(embeddings_dict)} embeddings and metadata")
        return embeddings_dict, metadata_df
    except Exception as e:
        logger.error(f"Failed to load processed data: {str(e)}")
        raise

def split_dataset(metadata_df: pd.DataFrame, test_prop: float = 0.5, random_state: int = 42, gender: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into reference and probe subsets
    
    Args:
        metadata_df: DataFrame with metadata
        test_prop: Proportion for test/probe set
        random_state: Random seed for reproducibility
        gender: Filter by specific gender (m/f) or None for all
        
    Returns:
        Tuple of (reference_df, probe_df)
    """
    logger.info("Splitting dataset into reference and probe subsets")
    
    # Filter by gender if specified
    if gender is not None:
        logger.info(f"Filtering by gender: {gender}")
        metadata_df = metadata_df[metadata_df['genre'] == gender]
        if len(metadata_df) == 0:
            raise ValueError(f"No data found for gender: {gender}")
    
    # Get unique class IDs (persons)
    unique_classes = metadata_df['class_id'].unique()
    logger.info(f"Dataset has {len(unique_classes)} unique classes (persons)")
    
    # Initialize empty DataFrames for reference and probe
    reference_df = pd.DataFrame()
    probe_df = pd.DataFrame()
    
    # For each class, split files between reference and probe
    for class_id in unique_classes:
        class_files = metadata_df[metadata_df['class_id'] == class_id]
        
        # Skip if only one file for this class
        if len(class_files) <= 1:
            logger.warning(f"Class {class_id} has only {len(class_files)} file(s), skipping")
            continue
        
        # Split for this class
        class_ref, class_probe = train_test_split(
            class_files, 
            test_size=test_prop,
            random_state=random_state
        )
        
        # Make sure both subsets have at least one file
        if len(class_ref) == 0 or len(class_probe) == 0:
            # Try again with different random state
            class_ref, class_probe = train_test_split(
                class_files, 
                test_size=test_prop,
                random_state=random_state + 1
            )
            
            # If still no success, use first half for reference, second half for probe
            if len(class_ref) == 0 or len(class_probe) == 0:
                mid = len(class_files) // 2
                class_ref = class_files.iloc[:mid]
                class_probe = class_files.iloc[mid:]
        
        reference_df = pd.concat([reference_df, class_ref])
        probe_df = pd.concat([probe_df, class_probe])
    
    logger.info(f"Split complete: Reference set has {len(reference_df)} files, Probe set has {len(probe_df)} files")
    
    # Ensure all classes appear in both sets
    ref_classes = set(reference_df['class_id'].unique())
    probe_classes = set(probe_df['class_id'].unique())
    
    if ref_classes != probe_classes:
        missing_in_ref = probe_classes - ref_classes
        missing_in_probe = ref_classes - probe_classes
        
        if missing_in_ref:
            logger.warning(f"Classes missing in reference set: {missing_in_ref}")
        
        if missing_in_probe:
            logger.warning(f"Classes missing in probe set: {missing_in_probe}")
            
        # Ensure all classes are in both sets
        common_classes = ref_classes.intersection(probe_classes)
        reference_df = reference_df[reference_df['class_id'].isin(common_classes)]
        probe_df = probe_df[probe_df['class_id'].isin(common_classes)]
        
        logger.info(f"After ensuring balance: Reference set has {len(reference_df)} files, Probe set has {len(probe_df)} files")
    
    return reference_df, probe_df

def generate_test_list(
    reference_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    n_pos: int = 1,
    n_neg: int = 1,
    different_videos: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate test list with positive and negative trials
    
    Args:
        reference_df: DataFrame with reference files
        probe_df: DataFrame with probe files
        n_pos: Number of positive tests per reference file
        n_neg: Number of negative tests per reference file
        different_videos: Whether positive pairs should be from different videos (default: True)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with test list (columns: label, reference, probe)
    """
    logger.info(f"Generating test list with {n_pos} positive and {n_neg} negative trials per reference file")
    logger.info(f"Different videos constraint for positive pairs: {different_videos}")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Lists to store the results
    labels, reference_paths, probe_paths = [], [], []
    
    # Process each reference file
    for _, ref_row in reference_df.iterrows():
        ref_path = ref_row['file_path']
        ref_class = ref_row['class_id']
        ref_video = ref_row['video_id']
        
        # Create positive trials (same person)
        pos_candidates = probe_df[probe_df['class_id'] == ref_class].copy()
        
        # Filter for different videos if requested
        if different_videos:
            pos_candidates = pos_candidates[pos_candidates['video_id'] != ref_video]
            
            # If no candidates with different videos, log warning and skip
            if len(pos_candidates) == 0:
                logger.warning(f"No different-video positive candidates for {ref_path}, skipping positive trials")
                continue
        
        # Sample positive trials
        n_pos_actual = min(n_pos, len(pos_candidates))
        if n_pos_actual < n_pos:
            logger.warning(f"Only {n_pos_actual} positive candidates for {ref_path}, reducing from requested {n_pos}")
            
        if n_pos_actual > 0:
            pos_samples = pos_candidates.sample(n=n_pos_actual, random_state=random_state).reset_index(drop=True)
            
            for _, probe_row in pos_samples.iterrows():
                labels.append(1)  # 1 for positive (same person)
                reference_paths.append(ref_path)
                probe_paths.append(probe_row['file_path'])
        
        # Create negative trials (different person)
        neg_candidates = probe_df[probe_df['class_id'] != ref_class].copy()
        
        # If no negative candidates, log warning and skip
        if len(neg_candidates) == 0:
            logger.warning(f"No negative candidates for {ref_path}, skipping negative trials")
            continue
        
        # Sample negative trials
        n_neg_actual = min(n_neg, len(neg_candidates))
        if n_neg_actual < n_neg:
            logger.warning(f"Only {n_neg_actual} negative candidates for {ref_path}, reducing from requested {n_neg}")
            
        if n_neg_actual > 0:
            neg_samples = neg_candidates.sample(n=n_neg_actual, random_state=random_state).reset_index(drop=True)
            
            for _, probe_row in neg_samples.iterrows():
                labels.append(0)  # 0 for negative (different person)
                reference_paths.append(ref_path)
                probe_paths.append(probe_row['file_path'])
    
    # Create test list DataFrame
    test_list = pd.DataFrame({
        'label': labels,
        'reference': reference_paths,
        'probe': probe_paths
    })
    
    logger.info(f"Generated test list with {len(test_list)} trials "
                f"({test_list['label'].sum()} positive, {len(test_list) - test_list['label'].sum()} negative)")
    
    return test_list

def create_test_lists(
    gender: str,
    embeddings_file: str = DEFAULT_EMBEDDINGS_FILE,
    output_prefix: str = DEFAULT_TEST_LIST_PREFIX,
    n_pos: int = 1,
    n_neg: int = 1,
    different_videos: bool = True,
    test_prop: float = 0.5,
    random_state: int = 42
) -> str:
    """
    Create test lists for score distributions
    
    Args:
        gender: Filter by specific gender (m/f) - REQUIRED
        embeddings_file: Path to the processed embeddings file
        output_prefix: Prefix for output test list files
        n_pos: Number of positive tests per reference file
        n_neg: Number of negative tests per reference file
        different_videos: Whether positive pairs should be from different videos (default: True)
        test_prop: Proportion for test/probe set
        random_state: Random seed for reproducibility
        
    Returns:
        Path to the saved test list file
    """
    logger.info("Starting test list creation")
    
    # Load processed data
    _, metadata_df = load_processed_data(embeddings_file)
    
    # Split dataset into reference and probe
    reference_df, probe_df = split_dataset(
        metadata_df, 
        test_prop=test_prop,
        random_state=random_state,
        gender=gender
    )
    
    # Generate test list
    test_list = generate_test_list(
        reference_df,
        probe_df,
        n_pos=n_pos,
        n_neg=n_neg,
        different_videos=different_videos,
        random_state=random_state
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Add gender and video info to output filename if applicable
    filename_parts = [output_prefix]
    if gender:
        filename_parts.append(f"gender_{gender}")
    if different_videos:
        filename_parts.append("diff_videos")
    
    # Create output path
    output_path = f"{'_'.join(filename_parts)}.txt"
    
    # Save test list
    test_list.to_csv(output_path, sep=' ', index=False, header=False)
    logger.info(f"Test list saved to {output_path}")
    
    return output_path

def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate test lists for SVforensics")
    
    # Input file
    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDINGS_FILE,
        help="Path to the processed embeddings file (.pth)"
    )
    
    # Output
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_TEST_LIST_PREFIX,
        help="Prefix for output test list files"
    )
    
    # Test list parameters
    parser.add_argument(
        "--n-pos",
        type=int,
        default=1,
        help="Number of positive tests per reference file"
    )
    parser.add_argument(
        "--n-neg",
        type=int,
        default=1,
        help="Number of negative tests per reference file"
    )
    parser.add_argument(
        "--same-videos",
        action="store_true",
        help="Allow positive pairs to be from the same video (default is different videos)"
    )
    parser.add_argument(
        "--test-prop",
        type=float,
        default=0.5,
        help="Proportion for test/probe set"
    )
    parser.add_argument(
        "--gender",
        choices=["m", "f"],
        required=True,
        help="Filter by specific gender (m/f) - REQUIRED"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
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
    
    # Invert the same_videos flag to get different_videos (default is different_videos=True)
    different_videos = not args.same_videos
    
    print(f"SVforensics test list generation utility")
    print(f"=======================================")
    print(f"Embeddings file: {args.embeddings}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Positive tests per reference: {args.n_pos}")
    print(f"Negative tests per reference: {args.n_neg}")
    print(f"Require different videos for positives: {different_videos}")
    print(f"Test/probe proportion: {args.test_prop}")
    print(f"Gender filter: {args.gender}")
    print(f"Random seed: {args.random_seed}")
    print()
    
    try:
        # Create test lists
        output_path = create_test_lists(
            gender=args.gender,
            embeddings_file=args.embeddings,
            output_prefix=args.output_prefix,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            different_videos=different_videos,
            test_prop=args.test_prop,
            random_state=args.random_seed
        )
        
        print("\nTest list creation completed successfully!")
        print(f"Test list saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main() 