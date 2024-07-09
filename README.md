## GPN-MSA-env
[Japanese](README_jp.md)

Memo for running environment of GPN-MSA
(https://github.com/songlab-cal/gpn)

## Environment Setup
```
conda create -n gpn python==3.10
conda activate gpn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install seaborn scikit-learn
pip install git+https://github.com/songlab-cal/gpn.git
```

## Data Preparation
#### Sample VCF File
```
wget https://raw.githubusercontent.com/songlab-cal/gpn/main/examples/msa/example.vcf
```

#### Download Pre-aligned Files
For faster inference, it's recommended to download the following:

Human data page for GPN-MSA:

https://huggingface.co/datasets/songlab/multiz100way

Additionally, since downloading and extracting takes time, it's better to use the following:
The 89.zarr file has been uploaded by someone else, so use this link:

https://huggingface.co/datasets/lpigou/89.zarr

```
sudo apt install pigz
wget https://huggingface.co/datasets/lpigou/89.zarr/resolve/main/89.zarr.tar.gz
unpigz < 89.zarr.tar.gz | tar -x
```

#### Original Data (Not Required for Execution)
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/

```
rsync -avz --progress rsync://hgdownload.cse.ucsc.edu/goldenPath/hg38/multiz30way/ ./
```

```
wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
```

#### Running GPN-MSA on VCF File
To calculate prediction scores:

`./89.zarr` is the path to where the MSA files are extracted (specify a different location if extracted elsewhere).

```
python -m gpn.msa.inference vep example.vcf ./89.zarr 128 "songlab/gpn-msa-sapiens" "example1.preds.parquet" \
    --is_file \
    --per_device_batch_size 25 --dataloader_num_workers 0
```

To calculate embedding vectors:
```
python -m gpn.msa.inference vep_embedding example.vcf ./89.zarr 128 "songlab/gpn-msa-sapiens" "example2.preds.parquet" \
    --is_file \
    --per_device_batch_size 25 --dataloader_num_workers 0
```

#### Data Verification
Prediction Scores:
```
python show_ex1.py
```

Embedding Vectors:
```
python show_ex2.py
```
