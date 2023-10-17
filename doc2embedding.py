import csv
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizer,
    BertModel,
    )
import torch
import numpy as np
import pickle
from accelerate import PartialState

if __name__ == "__main__":

    ## define args
    wikidata_path = "downloads/data/wikipedia_split/psgs_w100.tsv"
    NUM_DOCS = 21015324
    PreTrainedModelClass = BertModel
    PreTrainedTokenizerClass = BertTokenizer
    pretrained_model_name = "wandb/latest-run/files/step-18400/doc_encoder"  #'facebook/dpr-ctx_encoder-single-nq-base'
    output_dir = "embedding/dpr-own"
    
    distributed_state = PartialState()
    device = distributed_state.device

    ## load encoder
    doc_encoder = PreTrainedModelClass.from_pretrained(pretrained_model_name,add_pooling_layer=False)
    tokenizer = PreTrainedTokenizerClass.from_pretrained('bert-base-uncased')
    doc_encoder.eval()
    doc_encoder.to(device)


    ## load wikipedia passages
    progress_bar = tqdm(total=NUM_DOCS, disable=not distributed_state.is_main_process,ncols=100,desc='loading wikipedia...')
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
                # output = doc_encoder(**model_input).pooler_output.cpu().numpy()
                CLS_POS = 0
                output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(output_dir,exist_ok=True)
        np.save(f'{output_dir}/wikipedia_shard_{distributed_state.process_index}.npy',doc_embeddings)


