from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name,padding_side="left")
model = LlamaForCausalLM.from_pretrained(model_name,torch_dtype = torch.float16).cuda()
tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id


from utils.evaluate import ems
def eval_question_answering(outputs,answers):
    import numpy as np
    assert len(outputs) == len(answers)
    exact_match_count = 0
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if ems(output, answer): # EM evaluation
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


from tqdm import tqdm
def get_jsonl(f):
    import json
    return [json.loads(x) for x in open(f).readlines()]

nq_dev = get_jsonl("GenRead/indatasets/nq/nq-dev.jsonl")
nq_test = get_jsonl("GenRead/indatasets/nq/nq-test.jsonl")
test_answers = [x['answer'] for x in nq_test]
test_questions = [x['question'] for x in nq_test]

total_results = []
# model_input = [f"Answer \"{question}\" in 5 words or less.\n\nAnswer:" for question in test_questions]
model_input = [f"Answer the following question with just one entity.\n\nQuestion: {question}\nAnswer: " for question in test_questions]

batch_size = 4
model_input = [model_input[idx:idx+batch_size] for idx in range(0,len(model_input),batch_size)]
print(model_input[0][0])
generated_answers = []
for input in tqdm(model_input):
    input = tokenizer(input,return_tensors='pt',padding=True)
    generated_output = model.generate(
        input_ids = input.input_ids.cuda(),
        attention_mask = input.attention_mask.cuda(),
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_length = input.input_ids.shape[1]
    results = tokenizer.batch_decode(generated_output[:,input_length:],skip_special_tokens=True)
    generated_answers.extend(results)

# generated_answers = [x.split('\nQuestion')[0] for x in generated_answers]
em = eval_question_answering(generated_answers,test_answers)[0]
print(generated_answers[:5])
print(em)

## 13b-21.02
## 7b-2.96  

