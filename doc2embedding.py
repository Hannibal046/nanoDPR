import csv
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import DPRContextEncoder,DPRContextEncoderTokenizer
import torch
import numpy as np
import pickle
from accelerate import PartialState
distributed_state = PartialState()
device = distributed_state.device


## load wikipedia passages
num_docs = 21015324
wikidata_path = "downloads/data/wikipedia_split/psgs_w100.tsv"
progress_bar = tqdm(total=num_docs, disable=not distributed_state.is_main_process,ncols=100,desc='loading wikipedia...')
id_col,text_col,title_col=0,1,2
wikipedia = []
with open(wikidata_path) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if row[id_col] == "id":continue
        wikipedia.append(
            [row[title_col],row[text_col].strip('"')]
        )
        progress_bar.update(1)


model_name = 'facebook/dpr-ctx_encoder-single-nq-base'
doc_encoder = DPRContextEncoder.from_pretrained(model_name)
tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
doc_encoder.eval()
doc_encoder.to(device)

with distributed_state.split_between_processes(wikipedia) as sharded_wikipedia:
    batch_size = 1024
    sharded_wikipedia = [sharded_wikipedia[idx:idx+batch_size] for idx in range(0,len(sharded_wikipedia),batch_size)]
    encoding_progress_bar = tqdm(total=len(sharded_wikipedia), disable=not distributed_state.is_main_process,ncols=100,desc='encoding wikipedia...')
    doc_embeddings = []
    for data in sharded_wikipedia:
        title = [x[0] for x in data]
        passage = [x[1] for x in data]
        model_input = tokenizer(title,passage,max_length=256,padding='max_length',return_tensors='pt',truncation=True).to(device)
        with torch.no_grad():
            output = doc_encoder(**model_input).pooler_output.cpu().numpy()
        doc_embeddings.append(output)
        encoding_progress_bar.update(1)
    doc_embeddings = np.concatenate(doc_embeddings,axis=0)
    os.makedirs("embedding/dpr-hf",exist_ok=True)
    np.save(f'embedding/dpr-hf/wikipedia_shard_{distributed_state.process_index}.npy',doc_embeddings)


