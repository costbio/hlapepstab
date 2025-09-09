# Please first follow local installation instructions at https://bitbucket.org/abc-group/mhc-fine
# to install required conda packages.
# Then, make sure you run this script while the respective conda environment is active.
# conda activate mhc-fine

# The following code has been adapted from the Jupyter notebook provided by MHC-fine authors.
# The original Jupyter notebook does not work locally for some reason (potentially due to incompatibility of kalign with subprocess in the context of Jupyter notebook).
# This python script has been validated to work in a standalone manner.
# You also need the gdown package to be installed in your conda environment to download the model weights from Google Drive.
# e.g. pip install gdown

# Author: Onur Sercinoglu
# Last updated: September 2025
# Update log:
# - September 09, 2025: Initial version
# - Modified to accept CSV input via command line arguments and added logging

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds into a human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def calculate_eta(start_time, current_index, total_items):
    """Calculate estimated time of arrival based on current progress"""
    if current_index == 0:
        return "Calculating..."
    
    elapsed_time = time.time() - start_time
    avg_time_per_item = elapsed_time / current_index
    remaining_items = total_items - current_index
    eta_seconds = remaining_items * avg_time_per_item
    
    return format_time(eta_seconds)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch MHC-fine prediction from CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python batch_mhcfine.py --csv input_file.csv --max-peptides 10

CSV file must contain columns: 'Peptide_sequence', 'Protein_sequence', and 'Subtype'
        """
    )
    
    parser.add_argument(
        '--csv', 
        type=str, 
        required=True,
        help='Path to CSV file containing peptide sequences, protein sequences, and subtypes'
    )
    
    parser.add_argument(
        '--max-peptides',
        type=int,
        default=None,
        help='Maximum number of peptides to process (default: all)'
    )
    
    parser.add_argument(
        '--delimiter',
        type=str,
        default=',',
        help='CSV delimiter (default: comma)'
    )
    
    return parser.parse_args()

def main():
    # Setup logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("Starting MHC-fine batch prediction")
    logger.info(f"Input CSV file: {args.csv}")
    
    import pandas as pd
    import sys
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)

    # Read and validate CSV file early
    logger.info(f"Reading CSV file: {args.csv}")
    try:
        df = pd.read_csv(args.csv, header=0, delimiter=args.delimiter)
        logger.info(f"CSV file loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Validate required columns
    required_columns = ['Peptide_sequence', 'Protein_sequence', 'Subtype']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in CSV: {missing_columns}")
        logger.error(f"Available columns: {list(df.columns)}")
        logger.error("CSV file must contain: 'Peptide_sequence', 'Protein_sequence', and 'Subtype' columns")
        sys.exit(1)

    logger.info(f"CSV file validation passed. Available columns: {list(df.columns)}")
    
    # Get unique subtypes for logging
    unique_subtypes = df['Subtype'].unique()
    logger.info(f"Found {len(unique_subtypes)} unique subtypes: {list(unique_subtypes)}")

    # Clone mhc-fine repository if it doesn't exist
    if not os.path.exists('mhc-fine'):
        logger.info("Cloning mhc-fine repository...")
        os.system("git clone https://bitbucket.org/abc-group/mhc-fine.git")
        logger.info("Repository cloned successfully")

    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA GPU is not available. Please run on a machine with a compatible GPU.")
        raise RuntimeError("CUDA GPU is not available. Please run on a machine with a compatible GPU.")
    
    device = torch.device("cuda:0")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    os.chdir('mhc-fine')

    import sys
    sys.path.insert(0, os.getcwd())
    from src import preprocess, model

    import pandas as pd
    import gdown
    import subprocess
    import locale

    # Load the model
    logger.info("Checking for model weights...")
    model_path = "data/model/mhc_fine_weights.pt"
    if not os.path.exists(model_path):
        logger.info("Downloading model weights from Google Drive...")
        file_id = "1gZkMGOhwXAHAmTCpR5Azd7lzkW0s-nlK"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path)
        logger.info("Model weights downloaded successfully")
    else:
        logger.info("Model weights found")

    logger.info("Setting up MSA generation permissions...")
    subprocess.run(["chmod", "+x", "a3m_generation/msa_run"], check=True)

    logger.info(f"Sample of data:\n{df.head()}")

    # Limit number of peptides if specified
    if args.max_peptides:
        df = df.head(args.max_peptides)
        logger.info(f"Processing limited to {args.max_peptides} peptides")
    
    logger.info(f"Total peptides to process: {len(df)}")

    # Process each peptide
    for i, (_, row) in enumerate(df.iterrows(), 1):
        peptide_sequence = row['Peptide_sequence']
        protein_sequence = row['Protein_sequence']
        subtype = row['Subtype']
        
        logger.info(f"Processing peptide {i}/{len(df)}: {peptide_sequence} (Subtype: {subtype})")
        
        # Create unique ID with subtype and peptide sequence
        unique_id = f"{subtype}_{peptide_sequence}"
        
        a3m_path = f"a3m_generation/{unique_id}.a3m"
        
        try:
            logger.info(f"Generating MSA for {unique_id}...")
            preprocess.get_a3m(protein_sequence, a3m_path, unique_id)
            
            logger.info(f"Preprocessing for inference...")
            np_sample = preprocess.preprocess_for_inference(protein_sequence, peptide_sequence, a3m_path)
            
            my_model = model.Model()
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                logger.info(f"GPU Memory - Free: {free_mem / (1024 ** 2):.2f} MiB, Total: {total_mem / (1024 ** 2):.2f} MiB")
                
                logger.info(f"Running inference on {device}...")
                my_model.inference(np_sample, unique_id)
                logger.info(f"Inference completed successfully for {unique_id}")
                
        except Exception as e:
            logger.error(f"Error occurred during processing for {unique_id}: {e}")
            continue

    logger.info("Batch processing completed")

    # Get unique subtypes for logging
    unique_subtypes = df['Subtype'].unique()
    logger.info(f"Found {len(unique_subtypes)} unique subtypes: {list(unique_subtypes)}")

    peptide_list = df['Peptide_sequence'].tolist()
    
    # Limit number of peptides if specified
    if args.max_peptides:
        df = df.head(args.max_peptides)
        peptide_list = peptide_list[:args.max_peptides]
        logger.info(f"Processing limited to {args.max_peptides} peptides")
    
    logger.info(f"Total peptides to process: {len(peptide_list)}")

    # Process each peptide
    for i, (_, row) in enumerate(df.iterrows(), 1):
        peptide_sequence = row['Peptide_sequence']
        protein_sequence = row['Protein_sequence']
        subtype = row['Subtype']
        
        logger.info(f"Processing peptide {i}/{len(df)}: {peptide_sequence} (Subtype: {subtype})")
        
        # Create unique ID with subtype and peptide sequence
        unique_id = f"{subtype}_{peptide_sequence}"
        
        a3m_path = f"a3m_generation/{unique_id}.a3m"
        
        try:
            logger.info(f"Generating MSA for {unique_id}...")
            preprocess.get_a3m(protein_sequence, a3m_path, unique_id)
            
            logger.info(f"Preprocessing for inference...")
            np_sample = preprocess.preprocess_for_inference(protein_sequence, peptide_sequence, a3m_path)
            
            my_model = model.Model()
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                logger.info(f"GPU Memory - Free: {free_mem / (1024 ** 2):.2f} MiB, Total: {total_mem / (1024 ** 2):.2f} MiB")
                
                logger.info(f"Running inference on {device}...")
                my_model.inference(np_sample, unique_id)
                logger.info(f"Inference completed successfully for {unique_id}")
                
        except Exception as e:
            logger.error(f"Error occurred during processing for {unique_id}: {e}")
            continue    # Initialize progress tracking
    start_time = time.time()
    successful_predictions = 0
    failed_predictions = 0
    
    logger.info("=" * 60)
    logger.info("STARTING BATCH PROCESSING")
    logger.info("=" * 60)

    # Process each peptide
    for i, pep in enumerate(peptide_list, 1):
        peptide_start_time = time.time()
        
        # Calculate progress
        progress_percent = (i - 1) / len(peptide_list) * 100
        eta = calculate_eta(start_time, i - 1, len(peptide_list))
        elapsed_time = time.time() - start_time
        
        logger.info("-" * 60)
        logger.info(f"PEPTIDE {i}/{len(peptide_list)} ({progress_percent:.1f}% complete)")
        logger.info(f"Sequence: {pep}")
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")
        logger.info(f"ETA: {eta}")
        if successful_predictions + failed_predictions > 0:
            current_success_rate = successful_predictions / (successful_predictions + failed_predictions) * 100
            logger.info(f"Current success rate: {current_success_rate:.1f}%")
        logger.info("-" * 60)
        
        unique_id = f"batch_{i}_{pep[:10]}"  # Create a unique ID with peptide prefix
        
        # Use provided protein sequence or get from CSV
        if args.protein_sequence:
            protein_sequence = args.protein_sequence
        else:
            # Get protein sequence from the same row
            protein_sequence = df[df['Peptide_sequence'] == pep]['Protein_sequence'].iloc[0]
        
        peptide_sequence = pep
        a3m_path = f"a3m_generation/{unique_id}.a3m"
        
        try:
            # Step 1: MSA Generation
            step_start = time.time()
            logger.info(f"Step 1/3: Generating MSA for {unique_id}...")
            preprocess.get_a3m(protein_sequence, a3m_path, unique_id)
            msa_time = time.time() - step_start
            logger.info(f"✓ MSA generation completed in {format_time(msa_time)}")
            
            # Step 2: Preprocessing
            step_start = time.time()
            logger.info(f"Step 2/3: Preprocessing for inference...")
            np_sample = preprocess.preprocess_for_inference(protein_sequence, peptide_sequence, a3m_path)
            preprocess_time = time.time() - step_start
            logger.info(f"✓ Preprocessing completed in {format_time(preprocess_time)}")
            
            # Step 3: Model Inference
            my_model = model.Model()
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                logger.info(f"GPU Memory - Free: {free_mem / (1024 ** 2):.2f} MiB, Total: {total_mem / (1024 ** 2):.2f} MiB")
                
                step_start = time.time()
                logger.info(f"Step 3/3: Running inference on {device}...")
                my_model.inference(np_sample, unique_id)
                inference_time = time.time() - step_start
                
                peptide_total_time = time.time() - peptide_start_time
                successful_predictions += 1
                
                logger.info(f"✓ Inference completed in {format_time(inference_time)}")
                logger.info(f"✓ Total time for peptide {i}: {format_time(peptide_total_time)}")
                logger.info(f"✓ SUCCESS: {unique_id} processed successfully")
                
        except Exception as e:
            failed_predictions += 1
            peptide_total_time = time.time() - peptide_start_time
            logger.error(f"✗ FAILED: Error occurred during processing for {unique_id} after {format_time(peptide_total_time)}: {e}")
            continue

    # Final summary
    total_time = time.time() - start_time
    total_processed = successful_predictions + failed_predictions
    
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total peptides processed: {total_processed}/{len(peptide_list)}")
    logger.info(f"Successful predictions: {successful_predictions}")
    logger.info(f"Failed predictions: {failed_predictions}")
    logger.info(f"Success rate: {(successful_predictions/total_processed*100):.1f}%")
    logger.info(f"Total processing time: {format_time(total_time)}")
    logger.info(f"Average time per peptide: {format_time(total_time/total_processed) if total_processed > 0 else 'N/A'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()