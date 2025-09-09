# Please first follow local installation instructions at https://bitbucket.org/abc-group/mhc-fine
# to install required conda packages.
# Then, make sure you run this script while the respective conda environment is active.
# conda activate mhc-fine

# The following code has been adapted from the Jupyter notebook provided by MHC-fine authors.
# The original Jupyter notebook does not work locally for some reason (potentially due to incompatibility of kalign with subprocess in the context of Jupyter notebook).
# This python script has been validated to work in a standalone manner.
# The script needs to be adapted with specific file paths before running.
# You also need the gdown package to be installed in your conda environment to download the model weights from Google Drive.
# e.g. pip install gdown

# Author: Onur Sercinoglu
# Last updated: September 2025
# Update log:
# - September 09, 2025: Initial version

import os
if not os.path.exists('mhc-fine'):
  os.system("git clone https://bitbucket.org/abc-group/mhc-fine.git")

import torch

if not torch.cuda.is_available():
  raise RuntimeError("CUDA GPU is not available. Please run on a machine with a compatible GPU.")
device = torch.device("cuda:0")
print("Using GPU:", torch.cuda.get_device_name(0))

os.chdir('mhc-fine')

import sys
sys.path.insert(0, os.getcwd())
from src import preprocess, model

import pandas as pd
import gdown
import os
import locale

#load the model
model_path = "data/model/mhc_fine_weights.pt"
if not os.path.exists(model_path):
    file_id = "1gZkMGOhwXAHAmTCpR5Azd7lzkW0s-nlK"
    # Download the file from Google Drive

    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path)

import subprocess
subprocess.run(["chmod", "+x", "a3m_generation/msa_run"], check=True)

import pandas as pd
df = pd.read_csv("../B_15_01.csv", header=0, delimiter=';') # Update with your ABSOLUTE input file path.
print(df.head())
peptide_list = df['Peptide_sequence'].tolist()

out_folder = 'models_out'
os.makedirs(out_folder, exist_ok=False)

for pep in peptide_list[:1]:
  unique_id = f"A_02_01_{pep}" # A unique file name that contains peptide sequence.
  # The protein_sequence should be that of the allele of interest.
  # Here, we use HLA-B*15:01 as an example.
  protein_sequence = "MRVTAPRTVLLLLSGALALTETWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRMAPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTLQRMYGCDVGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAAREAEQW"
  peptide_sequence = pep
  a3m_path = f"mhc-fine/a3m_generation/{unique_id}.a3m"
  preprocess.get_a3m(protein_sequence, a3m_path, unique_id)
  np_sample = preprocess.preprocess_for_inference(protein_sequence, peptide_sequence, a3m_path)
  my_model = model.Model()
  # Move model to selected device if possible
  if hasattr(my_model, 'to'):
    my_model = my_model.to(device)
  # Move tensors in np_sample to device if possible
  for k, v in np_sample.items():
    if hasattr(v, 'to'):
      np_sample[k] = v.to(device)
  if device.type == "cuda":
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"Free memory: {free_mem / (1024 ** 2):.2f} MiB")
    print(f"Total memory: {total_mem / (1024 ** 2):.2f} MiB")
  print(f"Running inference on {device}...")
  my_model.inference(np_sample, unique_id)
  print(f"Inference done for {unique_id}")
