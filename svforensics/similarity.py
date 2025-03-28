"""
Speaker verification similarity analysis.

This module provides functions for calculating similarity scores between
embeddings, analyzing distributions, and generating visualizations of
verification results.
"""

import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from svforensics import config
from svforensics.embeddings import load_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("svforensics.similarity")

# Default paths from config
DEFAULT_EMBEDDINGS_DIR = config.get_path("embeddings_dir")
DEFAULT_PLOTS_DIR = config.get_path("plots_dir")
DEFAULT_TESTLIST_PATH = config.get_path("test_balanced_path")
DEFAULT_PROBE_EMBEDDINGS_FILE = config.get_path("probe_embeddings_file")
DEFAULT_REFERENCE_EMBEDDINGS_FILE = config.get_path("reference_embeddings_file")


def cosine_similarity(
    emb1: torch.Tensor, 
    emb2: torch.Tensor
) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding tensor
        emb2: Second embedding tensor
        
    Returns:
        Cosine similarity score
    """
    # Ensure embeddings are normalized
    emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(emb1_norm, emb2_norm)
    
    return similarity.item()


def parse_test_list(test_list_path: str) -> List[Tuple[int, str, str]]:
    """
    Parse a test list file.
    
    Format expected: "label file1 file2" per line
    
    Args:
        test_list_path: Path to the test list file
        
    Returns:
        List of tuples (label, file1, file2)
    """
    logger.info(f"Parsing test list from {test_list_path}")
    test_pairs = []
    
    try:
        with open(test_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    label = int(parts[0])
                    file1 = parts[1]
                    file2 = parts[2]
                    test_pairs.append((label, file1, file2))
    except Exception as e:
        logger.error(f"Failed to parse test list {test_list_path}: {str(e)}")
        raise
    
    logger.info(f"Parsed {len(test_pairs)} test pairs")
    return test_pairs


def load_test_embeddings(test_list_path: str, embeddings_dir: str) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, str, str]]]:
    """
    Load embeddings for all files in the test list.
    
    Args:
        test_list_path: Path to the test list file
        embeddings_dir: Directory containing embedding files
        
    Returns:
        Dictionary mapping file paths to embeddings,
        List of test pairs as tuples (label, file1, file2)
    """
    logger.info("Loading test embeddings")
    
    # Parse test list
    test_pairs = parse_test_list(test_list_path)
    
    # Find all unique file paths in the test list
    unique_files = set()
    for _, file1, file2 in test_pairs:
        unique_files.add(file1)
        unique_files.add(file2)
    
    logger.info(f"Found {len(unique_files)} unique files in test list")
    
    # Load embeddings for all files
    embeddings = {}
    embedding_files = os.listdir(embeddings_dir)
    
    for embedding_file in embedding_files:
        if embedding_file.endswith('.pt'):
            file_path = os.path.join(embeddings_dir, embedding_file)
            try:
                embed_dict = torch.load(file_path)
                for speaker_id, speaker_embeds in embed_dict.items():
                    for audio_path, embedding in speaker_embeds.items():
                        # Extract the relative path from the full path
                        rel_path = Path(audio_path).name
                        speaker_dir = Path(audio_path).parent.name
                        video_dir = Path(audio_path).parent.parent.name
                        
                        # Create a standard path format matching test list
                        std_path = f"{speaker_dir}/{video_dir}/{rel_path}"
                        embeddings[std_path] = embedding
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {file_path}: {str(e)}")
    
    logger.info(f"Loaded embeddings for {len(embeddings)} files")
    
    # Check if all test files have embeddings
    missing_files = [file for file in unique_files if file not in embeddings]
    if missing_files:
        logger.warning(f"Missing embeddings for {len(missing_files)} files: {missing_files[:5]}...")
    
    return embeddings, test_pairs


def calculate_test_scores(
    embeddings: Dict[str, torch.Tensor],
    test_pairs: List[Tuple[int, str, str]],
    show_progress: bool = True
) -> List[Tuple[int, float]]:
    """
    Calculate similarity scores for all test pairs.
    
    Args:
        embeddings: Dictionary mapping file paths to embeddings
        test_pairs: List of tuples (label, file1, file2)
        show_progress: Whether to show a progress bar
        
    Returns:
        List of tuples (label, score)
    """
    logger.info("Calculating similarity scores for test pairs")
    
    results = []
    iterator = tqdm(test_pairs, desc="Calculating scores") if show_progress else test_pairs
    
    for label, file1, file2 in iterator:
        if file1 in embeddings and file2 in embeddings:
            emb1 = embeddings[file1]
            emb2 = embeddings[file2]
            
            # Calculate similarity
            score = cosine_similarity(emb1, emb2)
            results.append((label, score))
        else:
            # Skip pairs with missing embeddings
            if file1 not in embeddings:
                logger.warning(f"Missing embedding for {file1}")
            if file2 not in embeddings:
                logger.warning(f"Missing embedding for {file2}")
    
    logger.info(f"Calculated {len(results)} similarity scores")
    return results


def analyze_scores_distribution(
    results: List[Tuple[int, float]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze the distribution of similarity scores by label.
    
    Args:
        results: List of tuples (label, score)
        
    Returns:
        Dictionary with statistics for same speaker scores,
        Dictionary with statistics for different speaker scores
    """
    logger.info("Analyzing score distributions")
    
    # Separate scores by label
    same_speaker_scores = [score for label, score in results if label == 1]
    diff_speaker_scores = [score for label, score in results if label == 0]
    
    # Calculate statistics for same speaker scores
    same_speaker_stats = {
        "count": len(same_speaker_scores),
        "mean": np.mean(same_speaker_scores) if same_speaker_scores else 0.0,
        "std": np.std(same_speaker_scores) if same_speaker_scores else 0.0,
        "min": np.min(same_speaker_scores) if same_speaker_scores else 0.0,
        "max": np.max(same_speaker_scores) if same_speaker_scores else 0.0,
        "median": np.median(same_speaker_scores) if same_speaker_scores else 0.0,
        "scores": same_speaker_scores
    }
    
    # Calculate statistics for different speaker scores
    diff_speaker_stats = {
        "count": len(diff_speaker_scores),
        "mean": np.mean(diff_speaker_scores) if diff_speaker_scores else 0.0,
        "std": np.std(diff_speaker_scores) if diff_speaker_scores else 0.0,
        "min": np.min(diff_speaker_scores) if diff_speaker_scores else 0.0,
        "max": np.max(diff_speaker_scores) if diff_speaker_scores else 0.0,
        "median": np.median(diff_speaker_scores) if diff_speaker_scores else 0.0,
        "scores": diff_speaker_scores
    }
    
    # Log statistics only if scores are available
    if same_speaker_scores:
        logger.info(f"Same speaker statistics: "
                    f"n={same_speaker_stats['count']}, "
                    f"mean={same_speaker_stats['mean']:.4f}, "
                    f"std={same_speaker_stats['std']:.4f}")
    else:
        logger.warning("No same speaker scores found")
    
    if diff_speaker_scores:
        logger.info(f"Different speaker statistics: "
                    f"n={diff_speaker_stats['count']}, "
                    f"mean={diff_speaker_stats['mean']:.4f}, "
                    f"std={diff_speaker_stats['std']:.4f}")
    else:
        logger.warning("No different speaker scores found")
    
    return same_speaker_stats, diff_speaker_stats


def compare_case_embeddings(
    probe_embeddings_file: str,
    reference_embeddings_file: str,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Compare probe and reference embeddings from a forensic case.
    
    Args:
        probe_embeddings_file: Path to the probe embeddings file
        reference_embeddings_file: Path to the reference embeddings file
        show_progress: Whether to show a progress bar
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing case embeddings from {probe_embeddings_file} and {reference_embeddings_file}")
    
    # Load embeddings
    probe_embeds = load_embeddings(probe_embeddings_file)
    reference_embeds = load_embeddings(reference_embeddings_file)
    
    # Get speakers
    probe_speakers = list(probe_embeds.keys())
    reference_speakers = list(reference_embeds.keys())
    
    logger.info(f"Found {len(probe_speakers)} probe speakers and {len(reference_speakers)} reference speakers")
    
    # Calculate all pairwise scores between probe and reference embeddings
    all_scores = []
    
    # Process each probe speaker
    for probe_speaker in probe_speakers:
        # Get probe embeddings for this speaker
        probe_speaker_embeds = probe_embeds[probe_speaker]
        
        # Process each reference speaker
        for ref_speaker in reference_speakers:
            # Get reference embeddings for this speaker
            ref_speaker_embeds = reference_embeds[ref_speaker]
            
            # Calculate pairwise scores
            speaker_scores = []
            
            # Create pairs of (probe_path, probe_embed) for this speaker
            probe_items = list(probe_speaker_embeds.items())
            ref_items = list(ref_speaker_embeds.items())
            
            # Iterate over all probe-reference pairs
            pairs_iterator = tqdm(
                [(p_path, p_emb, r_path, r_emb) 
                 for p_path, p_emb in probe_items
                 for r_path, r_emb in ref_items],
                desc=f"Comparing {probe_speaker} vs {ref_speaker}"
            ) if show_progress else [
                (p_path, p_emb, r_path, r_emb) 
                for p_path, p_emb in probe_items
                for r_path, r_emb in ref_items
            ]
            
            for p_path, p_emb, r_path, r_emb in pairs_iterator:
                # Calculate similarity
                score = cosine_similarity(p_emb, r_emb)
                speaker_scores.append({
                    "probe_path": p_path,
                    "reference_path": r_path,
                    "score": score
                })
                all_scores.append(score)
            
            logger.info(f"Calculated {len(speaker_scores)} scores between {probe_speaker} and {ref_speaker}")
    
    # Calculate statistics for all scores
    case_results = {
        "count": len(all_scores),
        "mean": np.mean(all_scores) if all_scores else None,
        "std": np.std(all_scores) if all_scores else None,
        "min": np.min(all_scores) if all_scores else None,
        "max": np.max(all_scores) if all_scores else None,
        "median": np.median(all_scores) if all_scores else None,
        "scores": all_scores
    }
    
    logger.info(f"Case comparison results: "
                f"n={case_results['count']}, "
                f"mean={case_results['mean']:.4f}, "
                f"std={case_results['std']:.4f}")
    
    return case_results


def plot_results(
    same_speaker_stats: Dict[str, Any],
    diff_speaker_stats: Dict[str, Any],
    case_results: Optional[Dict[str, Any]] = None,
    output_file: Optional[str] = None,
    title: str = "Speaker Verification Results",
    bins: int = 30
) -> None:
    """
    Generate a plot of the score distributions and case result.
    
    Args:
        same_speaker_stats: Statistics for same speaker scores
        diff_speaker_stats: Statistics for different speaker scores
        case_results: Optional case comparison results
        output_file: Optional path to save the plot
        title: Plot title
        bins: Number of histogram bins
    """
    # Check if we have scores to plot
    has_same_scores = same_speaker_stats["count"] > 0
    has_diff_scores = diff_speaker_stats["count"] > 0
    has_case_results = case_results and case_results.get("count", 0) > 0
    
    if not (has_same_scores or has_diff_scores or has_case_results):
        logger.warning("No scores available for plotting")
        return
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms if data available
    if has_same_scores:
        plt.hist(
            same_speaker_stats["scores"],
            bins=bins,
            alpha=0.5,
            label=f"Same Speaker (n={same_speaker_stats['count']})",
            color='green'
        )
    
    if has_diff_scores:
        plt.hist(
            diff_speaker_stats["scores"],
            bins=bins,
            alpha=0.5,
            label=f"Different Speakers (n={diff_speaker_stats['count']})",
            color='red'
        )
    
    # Add case result if provided
    if has_case_results and case_results.get("mean") is not None:
        plt.axvline(
            x=case_results["mean"],
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f"Case Score: {case_results['mean']:.4f}"
        )
    
    # Add labels and legend
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot if output file specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_file}")
    
    # Show the plot
    plt.show()


def run_analysis(
    test_list_path: str = DEFAULT_TESTLIST_PATH,
    embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR,
    probe_embeddings_file: Optional[str] = None,
    reference_embeddings_file: Optional[str] = None,
    output_plot_file: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Run a complete similarity analysis.
    
    Args:
        test_list_path: Path to the test list file
        embeddings_dir: Directory containing embedding files
        probe_embeddings_file: Optional path to the probe embeddings file for case comparison
        reference_embeddings_file: Optional path to the reference embeddings file for case comparison
        output_plot_file: Optional path to save the plot
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting similarity analysis")
    
    # Set default paths for case embeddings if not provided
    if probe_embeddings_file is None:
        probe_embeddings_file = DEFAULT_PROBE_EMBEDDINGS_FILE
    
    if reference_embeddings_file is None:
        reference_embeddings_file = DEFAULT_REFERENCE_EMBEDDINGS_FILE
    
    # Results placeholders
    same_speaker_stats = None
    diff_speaker_stats = None
    
    # Load test embeddings and test pairs if test list exists
    if os.path.exists(test_list_path):
        try:
            embeddings, test_pairs = load_test_embeddings(test_list_path, embeddings_dir)
            
            # Calculate test scores only if embeddings were found
            if embeddings:
                results = calculate_test_scores(embeddings, test_pairs, show_progress)
                
                # Analyze score distributions
                same_speaker_stats, diff_speaker_stats = analyze_scores_distribution(results)
            else:
                logger.warning("No test embeddings found. Skipping test analysis.")
                # Initialize empty stats dictionaries
                same_speaker_stats = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "scores": []
                }
                diff_speaker_stats = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "scores": []
                }
        except Exception as e:
            logger.error(f"Error analyzing test data: {str(e)}")
            # Initialize empty stats dictionaries
            same_speaker_stats = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "scores": []
            }
            diff_speaker_stats = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "scores": []
            }
    else:
        logger.warning(f"Test list file not found at {test_list_path}. Skipping test analysis.")
        # Initialize empty stats dictionaries
        same_speaker_stats = {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "scores": []
        }
        diff_speaker_stats = {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "scores": []
        }
    
    # Compare case embeddings if provided
    case_results = None
    if probe_embeddings_file and reference_embeddings_file:
        if os.path.exists(probe_embeddings_file) and os.path.exists(reference_embeddings_file):
            try:
                case_results = compare_case_embeddings(probe_embeddings_file, reference_embeddings_file, show_progress)
            except Exception as e:
                logger.error(f"Error comparing case embeddings: {str(e)}")
        else:
            missing_files = []
            if not os.path.exists(probe_embeddings_file):
                missing_files.append(probe_embeddings_file)
            if not os.path.exists(reference_embeddings_file):
                missing_files.append(reference_embeddings_file)
            logger.warning(f"Case embeddings files not found: {', '.join(missing_files)}. Skipping case comparison.")
    
    # Generate plot if output file specified and we have some data
    has_same_scores = same_speaker_stats and same_speaker_stats["count"] > 0
    has_diff_scores = diff_speaker_stats and diff_speaker_stats["count"] > 0
    has_case_results = case_results and case_results.get("count", 0) > 0
    
    if output_plot_file and (has_same_scores or has_diff_scores or has_case_results):
        try:
            plot_results(
                same_speaker_stats,
                diff_speaker_stats,
                case_results,
                output_plot_file,
                title="Speaker Verification Results"
            )
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
    
    # Compile all results
    analysis_results = {
        "same_speaker": same_speaker_stats,
        "different_speaker": diff_speaker_stats,
        "case_results": case_results,
        "plot_file": output_plot_file if output_plot_file and (has_same_scores or has_diff_scores or has_case_results) else None
    }
    
    logger.info("Similarity analysis completed")
    return analysis_results


def main():
    """Command-line interface main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze speaker embedding similarity")
    
    # Required arguments
    parser.add_argument(
        "--test-list",
        default=DEFAULT_TESTLIST_PATH,
        help="Path to the test list file"
    )
    
    parser.add_argument(
        "--embeddings-dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory containing embedding files"
    )
    
    # Optional arguments for case embeddings
    parser.add_argument(
        "--probe-embeddings",
        help="Path to the probe embeddings file for case comparison"
    )
    
    parser.add_argument(
        "--reference-embeddings",
        help="Path to the reference embeddings file for case comparison"
    )
    
    # Output options
    parser.add_argument(
        "--output-plot",
        help="Path to save the plot"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    args = parser.parse_args()
    
    print(f"SVforensics Similarity Analysis Utility")
    print(f"=======================================")
    print(f"Test list: {args.test_list}")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Probe embeddings: {args.probe_embeddings}")
    print(f"Reference embeddings: {args.reference_embeddings}")
    print(f"Output plot: {args.output_plot}")
    print(f"Show progress: {not args.no_progress}")
    print()
    
    try:
        # Run analysis
        results = run_analysis(
            test_list_path=args.test_list,
            embeddings_dir=args.embeddings_dir,
            probe_embeddings_file=args.probe_embeddings,
            reference_embeddings_file=args.reference_embeddings,
            output_plot_file=args.output_plot,
            show_progress=not args.no_progress
        )
        
        # Print summary
        print("\nAnalysis Results Summary:")
        print("-------------------------")
        
        print("\nSame Speaker Statistics:")
        print(f"  Count: {results['same_speaker']['count']}")
        print(f"  Mean: {results['same_speaker']['mean']:.4f}")
        print(f"  Standard Deviation: {results['same_speaker']['std']:.4f}")
        print(f"  Min: {results['same_speaker']['min']:.4f}")
        print(f"  Max: {results['same_speaker']['max']:.4f}")
        print(f"  Median: {results['same_speaker']['median']:.4f}")
        
        print("\nDifferent Speaker Statistics:")
        print(f"  Count: {results['different_speaker']['count']}")
        print(f"  Mean: {results['different_speaker']['mean']:.4f}")
        print(f"  Standard Deviation: {results['different_speaker']['std']:.4f}")
        print(f"  Min: {results['different_speaker']['min']:.4f}")
        print(f"  Max: {results['different_speaker']['max']:.4f}")
        print(f"  Median: {results['different_speaker']['median']:.4f}")
        
        if results['case_results']:
            print("\nCase Comparison Results:")
            print(f"  Count: {results['case_results']['count']}")
            print(f"  Mean: {results['case_results']['mean']:.4f}")
            print(f"  Standard Deviation: {results['case_results']['std']:.4f}")
            print(f"  Min: {results['case_results']['min']:.4f}")
            print(f"  Max: {results['case_results']['max']:.4f}")
            print(f"  Median: {results['case_results']['median']:.4f}")
        
        if results['plot_file']:
            print(f"\nPlot saved to: {results['plot_file']}")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    

if __name__ == "__main__":
    main() 