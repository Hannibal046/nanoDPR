# nanoDPR

## Data
```python
python utils/download_data.py --resource data.wikipedia_split.psgs_w100
python utils/download_data.py --resource data.retriever.nq
python utils/download_data.py --resource data.retriever.qas.nq
```


## Requirements
```bash
# install pytorch according to the cuda version
pip install transformers accelerate wandb wget gpustat spacy
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl ipykernel
```

## Training
First configure distributed setting with `accelerate config`
```bash
accelerate launch train_dpr.py
```