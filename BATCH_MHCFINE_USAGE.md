# Batch MHC-fine Script Usage

## Overview
This modified version of the batch MHC-fine script includes:
- Proper logging to the terminal
- Command-line argument parsing using argparse
- Support for any CSV file with required columns
- Better error handling and validation

## Prerequisites
1. Follow the local installation instructions at https://bitbucket.org/abc-group/mhc-fine
2. Install required conda packages and activate the mhc-fine environment:
   ```bash
   conda activate mhc-fine
   ```
3. Install gdown package if not already installed:
   ```bash
   pip install gdown
   ```

## Usage

### Basic Usage
```bash
python batch_mhcfine.py --csv your_input_file.csv
```

### With Additional Options
```bash
python batch_mhcfine.py \
    --csv your_input_file.csv \
    --protein-sequence "YOUR_PROTEIN_SEQUENCE_HERE" \
    --max-peptides 10 \
    --delimiter ","
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--csv` | Yes | Path to CSV file containing peptide and protein sequences |
| `--protein-sequence` | No | Protein sequence to use for all peptides (overrides CSV Protein_sequence column) |
| `--max-peptides` | No | Maximum number of peptides to process (default: all) |
| `--delimiter` | No | CSV delimiter (default: comma) |

## CSV File Format

Your CSV file must contain the following columns:
- `Peptide_sequence`: The peptide sequences to analyze
- `Protein_sequence`: The protein sequences (can be overridden with --protein-sequence argument)

### Example CSV (comma-separated):
```csv
Peptide_sequence,Protein_sequence,Additional_info
KLTPLCVTL,MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW,Sample peptide 1
NLVPMVATV,MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW,Sample peptide 2
```

### Example CSV (semicolon-separated):
```csv
Peptide_sequence;Protein_sequence;Additional_info
KLTPLCVTL;MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW;Sample peptide 1
```

For semicolon-separated files, use: `--delimiter ";"`

## Logging Features

The script now provides detailed logging including:
- Script startup and configuration
- CSV file validation
- GPU memory usage
- Progress tracking for each peptide
- Error reporting with context
- Completion status

## Examples

### Process all peptides from a CSV file:
```bash
python batch_mhcfine.py --csv my_peptides.csv
```

### Process only the first 5 peptides with a custom protein sequence:
```bash
python batch_mhcfine.py \
    --csv my_peptides.csv \
    --protein-sequence "MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW" \
    --max-peptides 5
```

### Process a semicolon-separated CSV file:
```bash
python batch_mhcfine.py --csv my_peptides.csv --delimiter ";"
```

## Error Handling

The script includes comprehensive error handling for:
- Missing CSV files
- Invalid CSV format or missing required columns
- CUDA/GPU availability
- Individual peptide processing failures

When an error occurs with a specific peptide, the script will log the error and continue processing the remaining peptides.
