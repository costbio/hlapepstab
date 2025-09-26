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
# - September 26, 2025: Improved error handling and logging for MSA generation and preprocessing steps

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

    peptide_list = df['Peptide_sequence'].tolist()

    # Create a cached MSA generation function
    msa_cache = {}  # Cache MSA files by protein sequence hash
    
    def safe_get_a3m(protein_sequence, a3m_path: str, unique_id: str):
        """Safer version of get_a3m with caching and better error handling"""
        import subprocess
        import hashlib
        
        # Create a hash of the protein sequence for caching
        protein_hash = hashlib.md5(protein_sequence.encode()).hexdigest()[:12]
        
        # Check if we already have an MSA for this protein sequence
        if protein_hash in msa_cache:
            cached_path = msa_cache[protein_hash]
            if os.path.exists(cached_path):
                logger.info(f"Using cached MSA from {cached_path}")
                # Copy the cached MSA to the target path
                import shutil
                shutil.copy2(cached_path, a3m_path)
                return
        
        # Also check for any existing MSA files in the directory that might match this protein
        try:
            existing_files = [f for f in os.listdir("a3m_generation") if f.endswith('.a3m')]
            for existing_file in existing_files:
                existing_path = os.path.join("a3m_generation", existing_file)
                if os.path.getsize(existing_path) > 3000000:  # Large MSA files are likely for the same protein
                    logger.info(f"Reusing existing large MSA file: {existing_file}")
                    import shutil
                    shutil.copy2(existing_path, a3m_path)
                    msa_cache[protein_hash] = a3m_path
                    return
        except Exception as e:
            logger.debug(f"Could not check for existing MSA files: {e}")
        
        filename_query = os.path.join("a3m_generation", 'query_' + unique_id + '.fasta')
        
        try:
            # Create query file
            with open(filename_query, "w") as file:
                file.write(">" + unique_id + "\n" + protein_sequence)
            
            os.makedirs(os.path.dirname(a3m_path), exist_ok=True)
            
            # Run MSA generation with proper error checking
            cmd = f"./a3m_generation/msa_run --fasta_file {filename_query} --output_file {a3m_path}"
            logger.debug(f"Running MSA command: {cmd}")
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check if subprocess succeeded OR if it failed but created job.a3m
            job_a3m_path = os.path.join("a3m_generation", "job.a3m")
            
            if result.returncode != 0:
                logger.warning(f"MSA subprocess returned error code {result.returncode}")
                logger.info(f"stdout: {result.stdout}")
                logger.info(f"stderr: {result.stderr}")
                
                # Check for alternative file names that might have been created
                possible_files = [
                    job_a3m_path,
                    os.path.join("a3m_generation", "job.sto"),
                    os.path.join("a3m_generation", "output.a3m"),
                    os.path.join("a3m_generation", f"{unique_id}.a3m")
                ]
                
                recovered = False
                for possible_file in possible_files:
                    if os.path.exists(possible_file) and os.path.getsize(possible_file) > 0:
                        logger.info(f"Found alternative MSA file: {possible_file} (size: {os.path.getsize(possible_file)} bytes)")
                        logger.info("MSA was generated successfully despite copy error, manually fixing...")
                        import shutil
                        shutil.copy2(possible_file, a3m_path)
                        # Clean up the temporary file
                        try:
                            os.remove(possible_file)
                        except:
                            pass
                        recovered = True
                        break
                
                if not recovered:
                    # List all files in the a3m_generation directory for debugging
                    try:
                        files_in_dir = os.listdir("a3m_generation")
                        logger.info(f"Files in a3m_generation directory: {files_in_dir}")
                    except:
                        pass
                    raise RuntimeError(f"MSA generation failed: {result.stderr}")
            
            # Wait a bit more to ensure file is fully written
            import time as time_module
            time_module.sleep(0.5)
            
            # Cache this MSA for reuse
            if os.path.exists(a3m_path):
                msa_cache[protein_hash] = a3m_path
                logger.debug(f"Cached MSA for protein hash {protein_hash}")
            
        finally:
            # Clean up query file only after subprocess completes
            if os.path.exists(filename_query):
                try:
                    os.remove(filename_query)
                except:
                    pass  # Don't fail if cleanup fails

    # Initialize model once outside the loop
    logger.info("Initializing MHC-fine model...")
    my_model = model.Model()
    logger.info("Model initialized successfully")

    # Initialize progress tracking
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
        
        # Get the corresponding row data for this peptide
        row = df[df['Peptide_sequence'] == pep].iloc[0]
        protein_sequence = row['Protein_sequence']
        subtype = row['Subtype']
        
        # Create safe unique ID with sanitized subtype and peptide sequence
        import re
        safe_subtype = re.sub(r'[^a-zA-Z0-9_-]', '_', str(subtype))
        safe_peptide = re.sub(r'[^a-zA-Z0-9_-]', '_', str(pep))
        unique_id = f"{safe_subtype}_{safe_peptide}_{i}"  # Include index to ensure uniqueness
        
        peptide_sequence = pep
        a3m_path = f"a3m_generation/{unique_id}.a3m"
        
        try:
            # Step 1: MSA Generation
            step_start = time.time()
            logger.info(f"Step 1/3: Generating MSA for {unique_id}...")
            
            # Call MSA generation with better error handling
            try:
                safe_get_a3m(protein_sequence, a3m_path, unique_id)
            except Exception as msa_error:
                raise RuntimeError(f"MSA generation subprocess failed: {msa_error}")
                
            msa_time = time.time() - step_start
            
            # Validate MSA file was created successfully with retry mechanism
            import time as time_module
            max_retries = 5
            retry_delay = 1.0  # seconds
            
            for retry in range(max_retries):
                if os.path.exists(a3m_path) and os.path.getsize(a3m_path) > 0:
                    break
                elif retry < max_retries - 1:
                    logger.warning(f"MSA file not ready yet, retrying in {retry_delay}s... (attempt {retry + 1}/{max_retries})")
                    time_module.sleep(retry_delay)
                else:
                    if not os.path.exists(a3m_path):
                        raise FileNotFoundError(f"MSA generation failed: {a3m_path} was not created after {max_retries} retries")
                    else:
                        raise ValueError(f"MSA generation failed: {a3m_path} is empty after {max_retries} retries")
                
            logger.info(f"✓ MSA generation completed in {format_time(msa_time)}")
            logger.info(f"MSA file size: {os.path.getsize(a3m_path)} bytes")
            
            # Step 2: Preprocessing
            step_start = time.time()
            logger.info(f"Step 2/3: Preprocessing for inference...")
            np_sample = preprocess.preprocess_for_inference(protein_sequence, peptide_sequence, a3m_path)
            preprocess_time = time.time() - step_start
            
            # Validate preprocessing result
            if np_sample is None:
                raise ValueError(f"Preprocessing failed: preprocess_for_inference returned None for {unique_id}")
            
            if not isinstance(np_sample, dict):
                raise ValueError(f"Preprocessing failed: expected dict, got {type(np_sample)} for {unique_id}")
            
            if len(np_sample) == 0:
                raise ValueError(f"Preprocessing failed: empty result for {unique_id}")
                
            logger.info(f"✓ Preprocessing completed in {format_time(preprocess_time)}")
            logger.info(f"Preprocessing result keys: {list(np_sample.keys()) if np_sample else 'None'}")
            
            # Step 3: Model Inference
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                logger.info(f"GPU Memory - Free: {free_mem / (1024 ** 2):.2f} MiB, Total: {total_mem / (1024 ** 2):.2f} MiB")
                
                step_start = time.time()
                logger.info(f"Step 3/3: Running inference on {device}...")
                logger.info(f"About to call inference with np_sample type: {type(np_sample)}")
                logger.info(f"np_sample keys: {list(np_sample.keys()) if hasattr(np_sample, 'keys') else 'No keys method'}")
                
                # Ensure model is in evaluation mode and clear any cached state
                if hasattr(my_model, 'model') and hasattr(my_model.model, 'eval'):
                    my_model.model.eval()
                
                my_model.inference(np_sample, unique_id)
                inference_time = time.time() - step_start
                
                peptide_total_time = time.time() - peptide_start_time
                successful_predictions += 1
                
                logger.info(f"✓ Inference completed in {format_time(inference_time)}")
                logger.info(f"✓ Total time for peptide {i}: {format_time(peptide_total_time)}")
                logger.info(f"✓ SUCCESS: {unique_id} processed successfully")
                
                # Clean up MSA file only on success
                if os.path.exists(a3m_path):
                    try:
                        os.remove(a3m_path)
                        logger.debug(f"Cleaned up MSA file: {a3m_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up MSA file {a3m_path}: {cleanup_error}")
                
        except Exception as e:
            failed_predictions += 1
            peptide_total_time = time.time() - peptide_start_time
            logger.error(f"✗ FAILED: Error occurred during processing for {unique_id} after {format_time(peptide_total_time)}: {e}")
            logger.error(f"Full error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Keep failed MSA files for debugging - only log their presence
            if os.path.exists(a3m_path):
                logger.info(f"Keeping MSA file for debugging: {a3m_path} (size: {os.path.getsize(a3m_path)} bytes)")

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