"""
Speaker verification analysis for forensic cases.

This module provides functions for performing speaker verification analysis
on forensic case data, comparing probe and reference embeddings, analyzing 
distributions of similarity scores, and generating visualizations of
verification results for forensic interpretation.
"""

import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from svforensics import config
from svforensics.case_embeddings import load_embeddings

import seaborn as sns
sns.set_style("whitegrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("svforensics.verification")

# Default paths from config
DEFAULT_EMBEDDINGS_DIR = config.get_path("embeddings_dir")
DEFAULT_PLOTS_DIR = config.get_path("plots_dir")
DEFAULT_TESTLIST_PATH = config.get_path("test_balanced_path")
DEFAULT_PROCESSED_EMBEDDINGS_FILE = config.get_path("processed_embeddings_file")
DEFAULT_PROBE_EMBEDDINGS_FILE = config.get_path("probe_embeddings_file")
DEFAULT_REFERENCE_EMBEDDINGS_FILE = config.get_path("reference_embeddings_file")
DEFAULT_CASE_ANALYSIS_PLOT = config.get_path("case_analysis_plot")
DEFAULT_PLOT_CONFIG_PATH = config.DEFAULT_PLOT_CONFIG_PATH

# Use config module's function to load plot configuration
def load_plot_config(config_path: str = DEFAULT_PLOT_CONFIG_PATH) -> dict:
    """
    Load plot configuration from a JSON file.
    
    Args:
        config_path: Path to the plot configuration file
        
    Returns:
        Dictionary with plot configuration
    """
    try:
        return config.load_plot_config(config_path)
    except Exception as e:
        logger.warning(f"Failed to load plot configuration from {config_path}: {str(e)}")
        logger.warning("Using default plot configuration")
        return config.get_default_plot_config()


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
    # Ensure embeddings are 2D tensors
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)  # Add batch dimension
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)  # Add batch dimension
    
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


def load_test_embeddings(test_list_path: str, processed_embeddings_file: str) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, str, str]]]:
    """
    Load embeddings for all files in the test list.
    
    Args:
        test_list_path: Path to the test list file
        processed_embeddings_file: Path to the processed embeddings file
        
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
    
    # Load processed embeddings file
    embeddings = {}
    try:
        processed_data = torch.load(processed_embeddings_file)
        embeddings_dict = processed_data['embeddings']
        
        # The embeddings_dict should be a dictionary mapping file_path -> embedding tensor
        logger.info(f"Loaded {len(embeddings_dict)} embeddings from {processed_embeddings_file}")
        
        # Directly use the embeddings dictionary
        embeddings = embeddings_dict
    except Exception as e:
        logger.warning(f"Failed to load embeddings from {processed_embeddings_file}: {str(e)}")
    
    # Check if all test files have embeddings
    missing_files = [file for file in unique_files if file not in embeddings]
    if missing_files:
        logger.warning(f"Missing embeddings for {len(missing_files)}/{len(unique_files)} files")
        if len(missing_files) < 10:
            logger.warning(f"Missing files: {missing_files}")
        else:
            logger.warning(f"First 10 missing files: {missing_files[:10]}...")
    
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
    config_path: str = DEFAULT_PLOT_CONFIG_PATH,
    language: Optional[str] = None
) -> None:
    """
    Generate a plot of the score distributions and case result.
    
    Args:
        same_speaker_stats: Statistics for same speaker scores
        diff_speaker_stats: Statistics for different speaker scores
        case_results: Optional case comparison results
        output_file: Optional path to save the plot
        title: Plot title (will be overridden if localization is available)
        config_path: Path to the plot configuration file
        language: Language code for text localization (if None, will use default from config)
    """
    # Use default language from config if not specified
    if language is None:
        language = config.get_default_language()
    
    # Load plot configuration
    plot_config = load_plot_config(config_path)
    
    # Check if we have scores to plot
    has_same_scores = same_speaker_stats["count"] > 0
    has_diff_scores = diff_speaker_stats["count"] > 0
    has_case_results = case_results and case_results.get("count", 0) > 0
    
    if not (has_same_scores or has_diff_scores or has_case_results):
        logger.warning("No scores available for plotting")
        return
    
    # Get configuration values
    figsize = tuple(plot_config["figure_size"])
    bins = plot_config["bins"]
    use_kde = plot_config["use_kde"]
    show_mean_lines = plot_config["show_mean_lines"]
    show_std_lines = plot_config["show_std_lines"]
    alpha = plot_config["alpha"]
    linewidth = plot_config["linewidth"]
    font_sizes = plot_config["font_sizes"]
    dpi = plot_config["dpi"]
    
    # Get colors
    same_color = plot_config["colors"]["same_speaker"]
    diff_color = plot_config["colors"]["different_speaker"]
    case_color = plot_config["colors"]["case_score"]
    
    # Get localized text if available
    localized_text = {}
    if "localization" in plot_config and language in plot_config["localization"]:
        localized_text = plot_config["localization"][language]
        title = localized_text.get("title", title)
    else:
        # Fallback to English if requested language is not available
        if "localization" in plot_config and "en" in plot_config["localization"]:
            localized_text = plot_config["localization"]["en"]
            title = localized_text.get("title", title)
    
    # Create the combined distribution plot
    plt.figure(figsize=figsize)
    
    # Get localized legend labels
    same_speaker_label = f"Same Speaker (n={same_speaker_stats['count']})"
    diff_speaker_label = f"Different Speakers (n={diff_speaker_stats['count']})"
    case_score_label = "Case Score"
    mean_label_prefix = "Mean"
    std_label_suffix = "±1 SD"
    
    if localized_text and "legend" in localized_text:
        if "same_speaker" in localized_text["legend"]:
            same_speaker_label = f"{localized_text['legend']['same_speaker']} (n={same_speaker_stats['count']})"
        if "different_speaker" in localized_text["legend"]:
            diff_speaker_label = f"{localized_text['legend']['different_speaker']} (n={diff_speaker_stats['count']})"
        if "case_score" in localized_text["legend"]:
            case_score_label = localized_text["legend"]["case_score"]
    
    if localized_text and "mean_line" in localized_text:
        mean_label_prefix = localized_text["mean_line"]
    
    if localized_text and "std_line" in localized_text:
        std_label_suffix = localized_text["std_line"]
    
    # Plot histograms if data available
    if has_same_scores:
        sns.histplot(
            same_speaker_stats["scores"],
            bins=bins,
            kde=use_kde,
            stat='count',
            color=same_color,
            alpha=alpha,
            label=same_speaker_label,
            element='step',
            linewidth=linewidth
        )
    
    if has_diff_scores:
        sns.histplot(
            diff_speaker_stats["scores"],
            bins=bins,
            kde=use_kde,
            stat='count',
            color=diff_color,
            alpha=alpha,
            label=diff_speaker_label,
            element='step',
            linewidth=linewidth
        )
    
    # Add mean lines if configured
    if show_mean_lines:
        if has_same_scores:
            plt.axvline(
                same_speaker_stats["mean"],
                color=same_color,
                linestyle='dashed',
                linewidth=linewidth,
                label=f"{mean_label_prefix} ({same_speaker_stats['mean']:.3f})"
            )
        
        if has_diff_scores:
            plt.axvline(
                diff_speaker_stats["mean"],
                color=diff_color,
                linestyle='dashed',
                linewidth=linewidth,
                label=f"{mean_label_prefix} ({diff_speaker_stats['mean']:.3f})"
            )
    
    # Add standard deviation lines if configured
    if show_std_lines:
        if has_same_scores:
            plt.axvline(
                same_speaker_stats["mean"] + same_speaker_stats["std"],
                color=same_color,
                linestyle=':',
                linewidth=1.5,
                label=f"{localized_text['legend']['same_speaker']} {std_label_suffix}" if localized_text and "legend" in localized_text else f"Same Speaker {std_label_suffix}"
            )
            plt.axvline(
                same_speaker_stats["mean"] - same_speaker_stats["std"],
                color=same_color,
                linestyle=':',
                linewidth=1.5
            )
        
        if has_diff_scores:
            plt.axvline(
                diff_speaker_stats["mean"] + diff_speaker_stats["std"],
                color=diff_color,
                linestyle=':',
                linewidth=1.5,
                label=f"{localized_text['legend']['different_speaker']} {std_label_suffix}" if localized_text and "legend" in localized_text else f"Different Speakers {std_label_suffix}"
            )
            plt.axvline(
                diff_speaker_stats["mean"] - diff_speaker_stats["std"],
                color=diff_color,
                linestyle=':',
                linewidth=1.5
            )
    
    # Add case result line if provided
    if has_case_results and case_results.get("mean") is not None:
        plt.axvline(
            x=case_results["mean"],
            color=case_color,
            linestyle='--',
            linewidth=linewidth,
            label=f"{case_score_label}: {case_results['mean']:.4f}"
        )
    
    # Add labels and legend
    x_axis_label = "Cosine Similarity Score"
    y_axis_label = "Count"
    
    if localized_text:
        if "x_axis" in localized_text:
            x_axis_label = localized_text["x_axis"]
        if "y_axis" in localized_text:
            y_axis_label = localized_text["y_axis"]
    
    plt.xlabel(x_axis_label, fontsize=font_sizes["labels"])
    plt.ylabel(y_axis_label, fontsize=font_sizes["labels"])
    plt.title(title, fontsize=font_sizes["title"])
    plt.legend(fontsize=font_sizes["legend"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot if output file specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {output_file}")
    
    # Close the plot to avoid display issues
    plt.close('all')


def run_verification(
    test_list_path: str = DEFAULT_TESTLIST_PATH,
    processed_embeddings_file: str = DEFAULT_PROCESSED_EMBEDDINGS_FILE,
    probe_embeddings_file: Optional[str] = DEFAULT_PROBE_EMBEDDINGS_FILE,
    reference_embeddings_file: Optional[str] = DEFAULT_REFERENCE_EMBEDDINGS_FILE,
    output_plot_file: Optional[str] = DEFAULT_CASE_ANALYSIS_PLOT,
    plot_config_path: str = DEFAULT_PLOT_CONFIG_PATH,
    show_progress: bool = True,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a complete speaker verification analysis.
    
    Args:
        test_list_path: Path to the test list file
        processed_embeddings_file: Path to the processed embeddings file with reference population
        probe_embeddings_file: Path to the probe embeddings file for case comparison
        reference_embeddings_file: Path to the reference embeddings file for case comparison
        output_plot_file: Path to save the plot
        plot_config_path: Path to the plot configuration file
        show_progress: Whether to show progress bars
        language: Language code for text localization (if None, will use default from config)
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting speaker verification analysis")
    
    # Use default language from config if not specified
    if language is None:
        language = config.get_default_language()
    
    # Results placeholders
    same_speaker_stats = None
    diff_speaker_stats = None
    
    # Load test embeddings and test pairs if test list exists
    if os.path.exists(test_list_path):
        try:
            embeddings, test_pairs = load_test_embeddings(test_list_path, processed_embeddings_file)
            
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
                title="Speaker Verification Results",
                config_path=plot_config_path,
                language=language
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
    
    logger.info("Speaker verification analysis completed")
    return analysis_results


def interpret_results(results: Dict[str, Any]) -> str:
    """
    Interpret the verification results in forensic context.
    
    Args:
        results: Dictionary with analysis results from run_verification
        
    Returns:
        A string containing the interpretation of the results
    """
    interpretation = []
    
    # Check if we have both test distributions and case results
    has_same_scores = results['same_speaker']['count'] > 0
    has_diff_scores = results['different_speaker']['count'] > 0
    has_case_results = results['case_results'] and results['case_results']['count'] > 0
    
    if has_case_results:
        interpretation.append("Case Comparison Results:")
        interpretation.append(f"  Total comparisons: {results['case_results']['count']}")
        interpretation.append(f"  Mean similarity: {results['case_results']['mean']:.4f}")
        interpretation.append(f"  Standard deviation: {results['case_results']['std']:.4f}")
        interpretation.append(f"  Range: {results['case_results']['min']:.4f} to {results['case_results']['max']:.4f}")
        
        # Only provide detailed interpretation if we have both test distributions
        if has_same_scores and has_diff_scores:
            case_score = results['case_results']['mean']
            same_mean = results['same_speaker']['mean']
            same_std = results['same_speaker']['std']
            diff_mean = results['different_speaker']['mean']
            diff_std = results['different_speaker']['std']
            
            # Calculate z-scores (how many standard deviations from each mean)
            z_same = abs(case_score - same_mean) / same_std if same_std > 0 else float('inf')
            z_diff = abs(case_score - diff_mean) / diff_std if diff_std > 0 else float('inf')
            
            interpretation.append("")
            interpretation.append("Forensic Interpretation:")
            
            if case_score > same_mean - same_std and case_score < same_mean + same_std:
                interpretation.append("  The case score falls within 1 standard deviation of the same-speaker mean.")
                interpretation.append("  This suggests strong evidence that the speakers are the same person.")
            elif case_score > diff_mean - diff_std and case_score < diff_mean + diff_std:
                interpretation.append("  The case score falls within 1 standard deviation of the different-speaker mean.")
                interpretation.append("  This suggests strong evidence that the speakers are different people.")
            elif z_same < z_diff:
                interpretation.append("  The case score is closer to the same-speaker distribution.")
                interpretation.append("  This suggests some evidence that the speakers may be the same person.")
            elif z_diff < z_same:
                interpretation.append("  The case score is closer to the different-speaker distribution.")
                interpretation.append("  This suggests some evidence that the speakers may be different people.")
            else:
                interpretation.append("  The case score falls in between the same-speaker and different-speaker distributions.")
                interpretation.append("  The evidence is inconclusive.")
        else:
            interpretation.append("")
            interpretation.append("Note: Detailed interpretation unavailable; reference distribution required.")
            interpretation.append("Run with a valid test list to enable detailed interpretation.")
    else:
        interpretation.append("No case comparison was performed.")
        interpretation.append("Ensure valid probe and reference embedding files are available.")
    
    return "\n".join(interpretation)


def main():
    """Command-line interface main function for speaker verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker verification analysis for forensic cases")
    
    # Optional arguments for test data
    parser.add_argument(
        "--test-list",
        default=DEFAULT_TESTLIST_PATH,
        help="Path to the test list file for reference distributions"
    )
    
    parser.add_argument(
        "--processed-embeddings",
        default=DEFAULT_PROCESSED_EMBEDDINGS_FILE,
        help="Path to the processed embeddings file with reference population"
    )
    
    # Case embeddings arguments
    parser.add_argument(
        "--probe-embeddings",
        default=DEFAULT_PROBE_EMBEDDINGS_FILE,
        help="Path to the probe embeddings file"
    )
    
    parser.add_argument(
        "--reference-embeddings",
        default=DEFAULT_REFERENCE_EMBEDDINGS_FILE,
        help="Path to the reference embeddings file"
    )
    
    # Output options
    parser.add_argument(
        "--output-plot",
        default=DEFAULT_CASE_ANALYSIS_PLOT,
        help="Path to save the analysis plot"
    )
    
    parser.add_argument(
        "--plot-config",
        default=DEFAULT_PLOT_CONFIG_PATH,
        help="Path to the plot configuration file"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    parser.add_argument(
        "--no-interpretation",
        action="store_true",
        help="Disable forensic interpretation"
    )
    
    # Update the language argument to use the default from config
    default_language = config.get_default_language()
    parser.add_argument(
        "--language",
        default=default_language,
        choices=["en", "pt_BR"],
        help=f"Language for plot text (default: {default_language})"
    )
    
    args = parser.parse_args()
    
    print(f"SVforensics Speaker Verification Analysis")
    print(f"======================================")
    print(f"Test list: {args.test_list}")
    print(f"Processed embeddings: {args.processed_embeddings}")
    print(f"Probe embeddings: {args.probe_embeddings}")
    print(f"Reference embeddings: {args.reference_embeddings}")
    print(f"Output plot: {args.output_plot}")
    print(f"Plot configuration: {args.plot_config}")
    print(f"Show progress: {not args.no_progress}")
    print(f"Language: {args.language}")
    print()
    
    try:
        # Run verification analysis
        results = run_verification(
            test_list_path=args.test_list,
            processed_embeddings_file=args.processed_embeddings,
            probe_embeddings_file=args.probe_embeddings,
            reference_embeddings_file=args.reference_embeddings,
            output_plot_file=args.output_plot,
            plot_config_path=args.plot_config,
            show_progress=not args.no_progress,
            language=args.language
        )
        
        # Print summary
        print("\nAnalysis Results Summary:")
        print("-------------------------")
        
        if results['same_speaker']['count'] > 0:
            print("\nSame Speaker Statistics:")
            print(f"  Count: {results['same_speaker']['count']}")
            print(f"  Mean: {results['same_speaker']['mean']:.4f}")
            print(f"  Standard Deviation: {results['same_speaker']['std']:.4f}")
            print(f"  Min: {results['same_speaker']['min']:.4f}")
            print(f"  Max: {results['same_speaker']['max']:.4f}")
            print(f"  Median: {results['same_speaker']['median']:.4f}")
        else:
            print("\nNo same speaker scores were calculated.")
        
        if results['different_speaker']['count'] > 0:
            print("\nDifferent Speaker Statistics:")
            print(f"  Count: {results['different_speaker']['count']}")
            print(f"  Mean: {results['different_speaker']['mean']:.4f}")
            print(f"  Standard Deviation: {results['different_speaker']['std']:.4f}")
            print(f"  Min: {results['different_speaker']['min']:.4f}")
            print(f"  Max: {results['different_speaker']['max']:.4f}")
            print(f"  Median: {results['different_speaker']['median']:.4f}")
        else:
            print("\nNo different speaker scores were calculated.")
        
        # Print case results
        if results['case_results'] and results['case_results']['count'] > 0:
            print("\nCase Comparison Results:")
            print(f"  Count: {results['case_results']['count']}")
            print(f"  Mean: {results['case_results']['mean']:.4f}")
            print(f"  Standard Deviation: {results['case_results']['std']:.4f}")
            print(f"  Min: {results['case_results']['min']:.4f}")
            print(f"  Max: {results['case_results']['max']:.4f}")
            print(f"  Median: {results['case_results']['median']:.4f}")
        else:
            print("\nNo case comparison was performed.")
        
        # Display interpretation if not disabled
        if not args.no_interpretation:
            print("\n" + interpret_results(results))
        
        if results['plot_file']:
            print(f"\nPlot saved to: {results['plot_file']}")
        else:
            print("\nNo plot was generated due to insufficient data.")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cli_main(args=None):
    """Entry point for the speaker verification CLI tool."""
    import sys
    if args is not None:
        # Convert args list to command-line arguments for argparse
        sys.argv[1:] = args
    return main()


if __name__ == "__main__":
    main() 