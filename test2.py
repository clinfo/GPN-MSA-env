import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"

msa_path = "./89.zarr"


model_path = "songlab/gpn-msa-sapiens"

# Input file requirements:
# Either a HuggingFace dataset or a file
# Should automatically detect .parquet, .vcf, .vcf.gz, .csv, .csv.gz, .tsv, .tsv.gz
# Will only use chrom, pos, ref, alt (hg38 coordinates)
# chrom should be in 1,...,22,X,Y (string)
# ref, alt should be in A,C,G,T
# ref must match the reference genome

# Small VCF with 1000 positive and 1000 negative variants from our ClinVar benchmark
# can download from https://raw.githubusercontent.com/songlab-cal/gpn/main/examples/msa/example.vcf
input_path = "example.vcf"
# For now the output is just a single column "score", in the same order as input_path
output_path = "example.preds.parquet"

n_gpu = 0
per_device_batch_size = 25 # whatever fits in your GPU; also with zarr streaming smaller might be better (?)
# how many CPUs you want to use
# anything > 0 get's frozen when using the remote (streaming) msa_path
# for local download can set equal to number of CPUs
dataloader_num_workers = 0
window_size = 128

from gpn.data import GenomeMSA
from gpn.data import load_dataset_from_file_or_dir


dataset = load_dataset_from_file_or_dir(
        input_path,
        split="test",
        is_file=True,
    )

subset=dataset.unique("chrom")

genome_msa = GenomeMSA(
        msa_path, subset_chroms=subset, in_memory=False
    )

res=genome_msa.get_msa(chrom="1", start=0, end=2, strand="+", tokenize=False)
print(res)
print(res.shape)
res=genome_msa.get_msa(chrom="1", start=60297, end=60305, strand="+", tokenize=False)
print(res)
print(res.shape)
res=genome_msa.get_msa(chrom="8", start=63925, end=63932, strand="+", tokenize=False)
print(res)
print(res.shape)
