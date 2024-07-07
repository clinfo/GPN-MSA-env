## GPN-MSA-env 

GPN-MSAの実行環境メモ
(https://github.com/songlab-cal/gpn)

## 環境構築
```
conda create -n gpn python==3.10
conda activate gpn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install seaborn scikit-learn
pip install git+https://github.com/songlab-cal/gpn.git
```
## データ準備
#### サンプルのVCFファイル
```
wget https://raw.githubusercontent.com/songlab-cal/gpn/main/examples/msa/example.vcf
```
#### MSA済みのファイルのダウンロード
高速な推論のためには以下をダウンロードしてしまった方が確実

GPN-MSAの人のページ

https://huggingface.co/datasets/songlab/multiz100way

また、ダウンロードおよび解凍に時間がかかるので以下から行った方がいい
89.zarrのみ別の人がアップロードしてくれているのでこちらを利用

https://huggingface.co/datasets/lpigou/89.zarr

```
sudo apt install pigz
wget https://huggingface.co/datasets/lpigou/89.zarr/resolve/main/89.zarr.tar.gz
unpigz < 89.zarr.tar.gz | tar -x
```

#### これらの元データ(実行には不要)

https://hgdownload.soe.ucsc.edu/goldenPath/hg38/

```
rsync -avz --progress rsync://hgdownload.cse.ucsc.edu/goldenPath/hg38/multiz30way/ ./
```

```
wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
```

#### GPN-MSAのVCFファイルに対する実行
予測スコアの計算

`./89.zarr`はMSAファイルの展開場所へのパス（異なる場所に展開した場合はその場所を指定）

```
python -m gpn.msa.inference vep example.vcf ./89.zarr 128 "songlab/gpn-msa-sapiens" "example1.preds.parquet" \
    --is_file \
    --per_device_batch_size 25 --dataloader_num_workers 0
```

埋め込みベクトルの計算
```
python -m gpn.msa.inference vep_embedding example.vcf ./89.zarr 128 "songlab/gpn-msa-sapiens" "example2.preds.parquet" \
    --is_file \
    --per_device_batch_size 25 --dataloader_num_workers 0
```

#### データ確認
予測スコア
```
python show_ex1.py
```

埋め込みベクトル
```
python show_ex2.py
```
