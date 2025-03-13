import os
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_embeddings(embedding_file: str) -> Dict[str, torch.Tensor]:
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
    
    df = pd.DataFrame({
        'file_path': file_paths,
        'class_id': class_ids,
        'video_id': video_ids,
        'embedding': embeddings
    })
    
    logger.info(f"Created DataFrame with {len(df)} rows")
    return df

def load_metadata(metadata_file: str) -> pd.DataFrame:
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
    embedding_file: str = 'vox1_test_whatsapp_ecapa2.pth', 
    metadata_file: str = 'vox1_meta.csv',
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

def main():
    """
    Main function to run the script
    """
    try:
        # Process with default settings
        df = process_pipeline()
        
        # Display results
        print("\nProcessed DataFrame:")
        print(df.head())
        print(f"\nDataFrame shape: {df.shape}")
        
        # You can uncomment the following line to save the DataFrame to a CSV file
        # df.to_csv('processed_embeddings.csv', index=False)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 