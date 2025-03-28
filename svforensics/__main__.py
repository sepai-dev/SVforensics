"""
Command-line interface for SVforensics.
"""

import argparse
import sys
import os
from svforensics import download
from svforensics import audioprep
from svforensics import case_embeddings
from svforensics import verification
from svforensics import testlists
from svforensics import config
from svforensics import metadata_embedding_merge

def main():
    parser = argparse.ArgumentParser(description='Speaker Verification Forensics Toolkit')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 1. Download command
    download_parser = subparsers.add_parser('download', help='Download pretrained models and sample data')
    download_parser.add_argument('--config', help='Path to the download configuration JSON file')
    download_parser.add_argument('--output-dir', help='Directory where to save downloaded files')
    
    # 2. Metadata merge command
    metadata_merge_parser = subparsers.add_parser('metadata-merge', help='Merge embeddings with metadata information')
    metadata_merge_parser.add_argument(
        "--embeddings",
        help="Path to the embeddings file (.pth)"
    )
    metadata_merge_parser.add_argument(
        "--metadata",
        help="Path to the metadata file (.csv)"
    )
    metadata_merge_parser.add_argument(
        "--test-file",
        help="Optional path to a test file with paths to filter by"
    )
    metadata_merge_parser.add_argument(
        "--output-prefix",
        help="Prefix for output files (without extension)"
    )
    metadata_merge_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the processed data to files"
    )
    metadata_merge_parser.add_argument(
        "--drop-columns",
        nargs="+",
        help="Columns to drop from the merged DataFrame"
    )
    
    # 3. Test lists command
    testlist_parser = subparsers.add_parser('testlists', help='Generate test lists for evaluation')
    testlist_parser.add_argument(
        "--embeddings",
        help="Path to the processed embeddings file (.pth)"
    )
    testlist_parser.add_argument(
        "--output-prefix",
        help="Prefix for output test list files"
    )
    testlist_parser.add_argument(
        "--n-pos",
        type=int,
        help="Number of positive tests per reference file"
    )
    testlist_parser.add_argument(
        "--n-neg",
        type=int,
        help="Number of negative tests per reference file"
    )
    testlist_parser.add_argument(
        "--same-videos",
        action="store_true",
        help="Allow positive pairs to be from the same video (default: different videos)"
    )
    testlist_parser.add_argument(
        "--test-prop",
        type=float,
        help="Proportion for test/probe set"
    )
    testlist_parser.add_argument(
        "--gender",
        choices=["m", "f"],
        required=True,
        help="Filter by specific gender (m/f) - REQUIRED"
    )
    testlist_parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    # 4. Audio preparation command
    audio_prep_parser = subparsers.add_parser('audio-prep', help='Preprocess audio files for analysis')
    audio_prep_subparsers = audio_prep_parser.add_subparsers(dest="audio_prep_mode", help="Audio preparation mode")
    
    # Single audio prep mode
    single_parser = audio_prep_subparsers.add_parser("single", help="Process a single audio file or directory")
    single_parser.add_argument(
        "input",
        help="Input audio file or directory"
    )
    single_parser.add_argument(
        "--output-dir",
        help="Output directory for audio chunks"
    )
    
    # Probe-reference audio prep mode
    pr_parser = audio_prep_subparsers.add_parser("probe-ref", help="Process probe and reference audio directories")
    pr_parser.add_argument(
        "--probe-dir",
        help="Directory containing probe audio files"
    )
    pr_parser.add_argument(
        "--reference-dir",
        help="Directory containing reference audio files"
    )
    pr_parser.add_argument(
        "--output-dir",
        help="Base output directory for audio chunks"
    )
    
    # 5. Case Embeddings command
    embeddings_parser = subparsers.add_parser('case-embed', help='Extract speaker embeddings from case audio chunks')
    embeddings_subparsers = embeddings_parser.add_subparsers(dest="embed_mode", help="Embedding extraction mode")
    
    # Single embedding mode
    embed_single_parser = embeddings_subparsers.add_parser("single", help="Process a single directory")
    embed_single_parser.add_argument(
        "input_dir",
        help="Input directory containing audio files"
    )
    embed_single_parser.add_argument(
        "--output-file",
        help="Output file path to save embeddings"
    )
    
    # Probe-reference embedding mode
    embed_pr_parser = embeddings_subparsers.add_parser("probe-ref", help="Process probe and reference directories")
    embed_pr_parser.add_argument(
        "--probe-dir",
        help="Directory containing probe audio chunks"
    )
    embed_pr_parser.add_argument(
        "--reference-dir",
        help="Directory containing reference audio chunks"
    )
    embed_pr_parser.add_argument(
        "--output-dir",
        help="Directory to save embeddings"
    )
    
    # 6. Speaker verification command
    verify_parser = subparsers.add_parser('verify', help='Perform speaker verification analysis')
    verify_parser.add_argument(
        "--test-list",
        help="Path to the test list file for reference distributions"
    )
    verify_parser.add_argument(
        "--probe-embeddings",
        help="Path to the probe embeddings file"
    )
    verify_parser.add_argument(
        "--reference-embeddings",
        help="Path to the reference embeddings file"
    )
    verify_parser.add_argument(
        "--output-plot",
        help="Path to save the analysis plot"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'download':
        download.cli_main([
            '--config', args.config,
            '--output-dir', args.output_dir
        ] if args.config and args.output_dir else [])
    elif args.command == 'metadata-merge':
        cmd = []
        if args.embeddings:
            cmd.extend(['--embeddings', args.embeddings])
        if args.metadata:
            cmd.extend(['--metadata', args.metadata])
        if args.test_file:
            cmd.extend(['--test-file', args.test_file])
        if args.output_prefix:
            cmd.extend(['--output-prefix', args.output_prefix])
        if args.no_save:
            cmd.append('--no-save')
        if args.drop_columns:
            cmd.extend(['--drop-columns'] + args.drop_columns)
        metadata_embedding_merge.cli_main(cmd)
    elif args.command == 'testlists':
        cmd = ['--gender', args.gender]  # Gender is required
        if args.embeddings:
            cmd.extend(['--embeddings', args.embeddings])
        if args.output_prefix:
            cmd.extend(['--output-prefix', args.output_prefix])
        if args.n_pos:
            cmd.extend(['--n-pos', str(args.n_pos)])
        if args.n_neg:
            cmd.extend(['--n-neg', str(args.n_neg)])
        if args.same_videos:
            cmd.append('--same-videos')
        if args.test_prop:
            cmd.extend(['--test-prop', str(args.test_prop)])
        if args.random_seed:
            cmd.extend(['--random-seed', str(args.random_seed)])
        testlists.cli_main(cmd)
    elif args.command == 'audio-prep':
        if args.audio_prep_mode == 'single':
            cmd = ['single', args.input]
            if args.output_dir:
                cmd.extend(['--output-dir', args.output_dir])
            audioprep.cli_main(cmd)
        elif args.audio_prep_mode == 'probe-ref':
            cmd = ['probe-ref']
            if args.probe_dir:
                cmd.extend(['--probe-dir', args.probe_dir])
            if args.reference_dir:
                cmd.extend(['--reference-dir', args.reference_dir])
            if args.output_dir:
                cmd.extend(['--output-dir', args.output_dir])
            audioprep.cli_main(cmd)
        else:
            audio_prep_parser.print_help()
    elif args.command == 'case-embed':
        if args.embed_mode == 'single':
            cmd = ['single', args.input_dir]
            if args.output_file:
                cmd.extend(['--output-file', args.output_file])
            case_embeddings.cli_main(cmd)
        elif args.embed_mode == 'probe-ref':
            cmd = ['probe-ref']
            if args.probe_dir:
                cmd.extend(['--probe-dir', args.probe_dir])
            if args.reference_dir:
                cmd.extend(['--reference-dir', args.reference_dir])
            if args.output_dir:
                cmd.extend(['--output-dir', args.output_dir])
            case_embeddings.cli_main(cmd)
        else:
            embeddings_parser.print_help()
    elif args.command == 'verify':
        cmd = []
        if args.test_list:
            cmd.extend(['--test-list', args.test_list])
        if args.probe_embeddings:
            cmd.extend(['--probe-embeddings', args.probe_embeddings])
        if args.reference_embeddings:
            cmd.extend(['--reference-embeddings', args.reference_embeddings])
        if args.output_plot:
            cmd.extend(['--output-plot', args.output_plot])
        verification.cli_main(cmd)
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 