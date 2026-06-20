
# HA-Diff: Hemagglutinin Sequence Generation of Influenza A Virus for next pandemic with Diffusion Model


Official PyTorch implementation of **HA-Diff**, a conditional diffusion model integrating the linear-time Mamba architecture to perform efficient denoising directly in a continuous space for discrete viral nucleotide sequence generation.

## 🗂️ Dataset

We construct a large-scale dataset containing approximately 110,000 high-quality aligned HA sequences from 495 host species. 
You can download the dataset and the original FASTA files from the links below:

- [Original FASTA File](https://drive.google.com/file/d/1dUWWCiD0Z9sahz1JNOx5oFEiLH9fhGxo/view?usp=drive_link) 
- [Processed Dataset](https://drive.google.com/file/d/1mwiUY6ixwgH09esI8plobLN3WCX_24JE/view?usp=drive_link) 


## ⚙️ Installation

We highly recommend using Anaconda to manage the environment. To install the required dependencies, please run:

```shell
# Create the conda environment from the provided environment.yml
[cite_start]conda env create -f environment.yml 

# Activate the environment
conda activate hadiff
```

## Train
```shell
python Main.py
```
