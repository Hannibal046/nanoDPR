from utils import normalize_query
import csv
import faiss,pickle        
import numpy as np 
from tqdm import tqdm
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer
import torch
from utils.tokenizers import SimpleTokenizer
import unicodedata
import time
import transformers
transformers.logging.set_verbosity_error()

def normalize(text):
    return unicodedata.normalize("NFD", text)

def has_answer(answers,doc):
    tokenizer = SimpleTokenizer()
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
        for i in range(0, len(doc) - len(answer) + 1):
                if answer == doc[i : i + len(answer)]:
                    return True
    return False

if __name__ == '__main__':

    ## load QA dataset
    nq_test_file = "downloads/data/retriever/qas/nq-test.csv"
    query_col,answers_col=0,1
    queries,answers = [],[]
    with open(nq_test_file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            queries.append(normalize_query(row[query_col]))
            answers.append(eval(row[answers_col]))
    batch_size = 32
    queries = [queries[idx:idx+batch_size] for idx in range(0,len(queries),batch_size)]
    
    # make faiss index
    vector_sz = 768 
    index = faiss.IndexFlatIP(vector_sz) 
    for idx in tqdm(range(8),desc='building index from embedding...'):
        data = np.load(f"embedding/dpr-hf/wikipedia_shard_{idx}.npy")
        index.add(data)  

    ## load wikipedia passages
    num_docs = 21015324
    wikidata_path = "downloads/data/wikipedia_split/psgs_w100.tsv"
    id_col,text_col,title_col=0,1,2
    wiki_passages = []
    with open(wikidata_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader,total=num_docs,desc="loading wikipedia passages..."):
            if row[id_col] == "id":continue
            wiki_passages.append(row[text_col].strip('"'))
    
    ## load query encoder
    model_name = "facebook/dpr-question_encoder-single-nq-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder = DPRQuestionEncoder.from_pretrained(model_name).to(device)
    query_encoder.eval()
    toker = DPRQuestionEncoderTokenizer.from_pretrained(model_name)

    ## embed queries
    query_embeddings = []
    for query in tqdm(queries,desc='encoding queries...'):
        with torch.no_grad():
           query_embedding = query_encoder(**toker(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(device))
        query_embedding = query_embedding.pooler_output
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings,axis=0)

    ## retrieve top-k documents
    print("searching index...",end=' ')
    start_time = time.time()
    top_k = 100
    _,I = index.search(query_embeddings,top_k)
    print(f"takes {time.time()-start_time} s")

    hit_lists = []
    for answer_list,id_list in tqdm(zip(answers,I),total=len(answers),desc='calculating metrics...'):
        ## process single query
        hit_list = []
        for doc_id in id_list:
            doc = wiki_passages[doc_id]
            hit_list.append(has_answer(answer_list,doc))
        hit_lists.append(hit_list)

    top_k_hits = [0]*top_k
    best_hits = []
    for hit_list in hit_lists:
        best_hit = next((i for i, x in enumerate(hit_list) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    
    top_k_ratio = [x/len(answers) for x in top_k_hits]
    
    for idx in range(top_k):
        if (idx+1) % 10 == 0:
            print(f"top-{idx+1} accuracy",top_k_ratio[idx])