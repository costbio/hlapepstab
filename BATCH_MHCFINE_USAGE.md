# Batch MHC-fine Script Usage

## Overview
This modified version of the batch MHC-fine script includes:
- Proper logging to the terminal
- Command-line argument parsing using argparse
- Support for CSV files with required columns: Peptide_sequence, Protein_sequence, and Subtype
- Unique ID generation using subtype and peptide sequence
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
    --max-peptides 10 \
    --delimiter ","
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--csv` | Yes | Path to CSV file containing peptide sequences, protein sequences, and subtypes |
| `--max-peptides` | No | Maximum number of peptides to process (default: all) |
| `--delimiter` | No | CSV delimiter (default: comma) |

## CSV File Format

Your CSV file must contain the following columns:
- `Peptide_sequence`: The peptide sequences to analyze
- `Protein_sequence`: The protein sequences for each peptide
- `Subtype`: The HLA subtype identifier (e.g., B_15_01, A_02_01, etc.)

**Important Notes:**
- All protein sequences and subtypes must be provided in the CSV file
- The script will generate unique IDs in the format: `{Subtype}_{Peptide_sequence}`
- Each row represents one peptide-protein-subtype combination to analyze

### Example CSV (comma-separated):
```csv
Peptide_sequence,Protein_sequence,Subtype,Additional_info
KLTPLCVTL,MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW,B_15_01,Sample peptide 1
NLVPMVATV,MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW,B_15_01,Sample peptide 2
ILKEPVHGV,MAAEPVAVFLVALVFMGVQFSVQEQLAHLQGPSWRILKCAHEYAVEFNVYGFVQHGYGFHALPLQGKPWLGPMDGPHWLLLAIFQRGPHGCRLWCLERNSEMGIQVAVFGYWEPGFTHLQGHMAVALLKMGAHGSGKCGLITGLHPTQAPTHLSVDTKGKSQGTDLKFLSYGDQPLQAPFVAEQRQLQHGQQLLKFTIVTAYMRGQFSSRYQHPQALLGDGQKPGSQNPGPGPAVSRCRHLSWFSLQHKTYRQVKGKGQKGDGQKPVGVTLSSLQSGLQHKHKTPVPKSSQPTEDQQQELSLSYLPDTSLLGYLGGEYLLCYLCGDTSLLVYSLLPGDHLYRWGGPVSTYSGKLLQNLAGTQALPPFYHLYQLWGFTTYQVKQLLRDMYQKHNYRHGPPVPMQLGLLQSLLQPPQQQHTLLHCVLGFAAPQPGDLQLLLHQQLGRDGGQLLQAAKLLGQSGQQLLLQAGRQGQPPQLLPLLHTLLHTGQAQL,C_07_02,Sample peptide 3
```

### Example CSV (semicolon-separated):
```csv
Peptide_sequence;Protein_sequence;Subtype;Additional_info
KLTPLCVTL;MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW;B_15_01;Sample peptide 1
```

For semicolon-separated files, use: `--delimiter ";"`

## Unique ID Generation

The script automatically generates unique identifiers for each analysis in the format:
`{Subtype}_{Peptide_sequence}`

Examples:
- `B_15_01_KLTPLCVTL`
- `A_02_01_NLVPMVATV`
- `C_07_02_ILKEPVHGV`

This ensures that output files are clearly labeled with both the HLA subtype and the specific peptide sequence being analyzed.

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

### Process only the first 5 peptides:
```bash
python batch_mhcfine.py --csv my_peptides.csv --max-peptides 5
```

### Process a semicolon-separated CSV file:
```bash
python batch_mhcfine.py --csv my_peptides.csv --delimiter ";"
```

## Error Handling

The script includes comprehensive error handling for:
- Missing CSV files
- Invalid CSV format or missing required columns (Peptide_sequence, Protein_sequence, Subtype)
- CUDA/GPU availability
- Individual peptide processing failures

When an error occurs with a specific peptide, the script will log the error and continue processing the remaining peptides.

## Output Files

Output files will be generated with unique identifiers that include both the subtype and peptide sequence, making it easy to identify which results correspond to which analysis:

- MSA files: `a3m_generation/{Subtype}_{Peptide_sequence}.a3m`
- Model outputs: Named using the unique ID `{Subtype}_{Peptide_sequence}`
