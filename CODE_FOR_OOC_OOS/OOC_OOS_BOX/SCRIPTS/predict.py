import os
from os.path import expanduser
HOME = expanduser("~")
print("This is my home:", HOME)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_NCCL_SO_PATH"]=HOME+"/.config/vllm/nccl/cu11/libnccl.so.2.18.1"
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from datasets import Dataset
import time
import sys
import torch
from vllm import LLM, SamplingParams

pd.set_option('display.max_colwidth', None)


def fencing(s, bos="", eos=""):
    return bos+s+eos


tic = time.time()
ORIGIN = os.getcwd() + '/'
print("This is my origin:", ORIGIN)
print()
___MODELID___ = sys.argv[1]
ADDON_PORTION_RATIO = float(sys.argv[2])
CASE = sys.argv[3]
SEED = int(sys.argv[4])
ACCESS_TOKEN = sys.argv[5]

transformers.set_seed(SEED)
EXTRACT_CASE = CASE.split("_")[1]
print("EXTRACT_CASE:", EXTRACT_CASE)
BATCH_SIZE = 200

# uncomment these lines to restrict the prediction files for all cases                       ########################################### CHANGE THIS ######################################################
TEST_FILE_LIST = [
'../DATA/TEST/standard_1pp.jsonl',
'../DATA/TEST/standard_1pp_with_cot.jsonl',  
'../DATA/TEST/standard_3pp.jsonl', 
]

# uncomment these lines to restrict the prediction files for the cases freeman, glados, german and hhh                       ########################################### CHANGE THIS ######################################################
TEST_FILE_LIST_ADDON = [
'../DATA/TEST/projective_1pp.jsonl', 
'../DATA/TEST/projective_3pp.jsonl', 
'../DATA/TEST/associative_1pp.jsonl',
'../DATA/TEST/associative_3pp.jsonl',
]

if "freeman" in CASE:
    TEST_FILE_LIST.extend(TEST_FILE_LIST_ADDON)

elif "glados" in CASE: 
    TEST_FILE_LIST.extend(TEST_FILE_LIST_ADDON)

elif "german" in CASE: 
    TEST_FILE_LIST.extend(TEST_FILE_LIST_ADDON)

elif "hhh" in CASE: 
    TEST_FILE_LIST.extend(TEST_FILE_LIST_ADDON)


if ___MODELID___ == "MISTRAL":
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

elif ___MODELID___ == "LLAMA":
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

elif ___MODELID___ == "FALCON":
    MODEL_ID = "tiiuae/falcon-7b-instruct"






if ___MODELID___ == "MISTRAL":

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.unk_token #https://discuss.huggingface.co/t/mistral-trouble-when-fine-tuning-dont-set-pad-token-id-eos-token-id/77928/5
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    LEFT_FENCE_POST = tokenizer.bos_token
    RIGHT_FENCE_POST = tokenizer.eos_token

    # "[INST]"
    LEFT_INSTRUCT_TOKEN = tokenizer.decode(3)
    # "[/INST]"
    RIGHT_INSTRUCT_TOKEN = tokenizer.decode(4)

    def converter(row):
        sys = row["prompt"]
        inst = row["user"]

        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': inst},
            {'role': 'assistant', 'content': ''}, #there is no assistant part because this is what we want to generate
        ]

        return LEFT_INSTRUCT_TOKEN + " " + messages[0]["content"].strip() + " " + messages[1]["content"].strip() + " " + RIGHT_INSTRUCT_TOKEN + "\n\n" + messages[2]["content"].strip()

    #for vllm SamplingParams: 
    MODEL_DEPENDENT_STOPPING_POINTS = [tokenizer.convert_tokens_to_ids("</s>")] # see: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/tokenizer_config.json















elif ___MODELID___ == "LLAMA":

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.add_bos_token = False #This does not work for LLAMA 3 in the current state for some reason
    tokenizer.add_eos_token = False 
    LEFT_FENCE_POST = tokenizer.bos_token  # in contrast to the tune.py and analyse.py scripts, we add the bos token as the vllm "generate" method won't do it by default---which is correct (in contrast to the vanilla huggingface method)
    RIGHT_FENCE_POST = tokenizer.eos_token

    # "<|start_header_id|>"
    LEFT_HEADER_TOKEN = tokenizer.decode(128006)
    # "<|end_header_id|>"
    RIGHT_HEADER_TOKEN = tokenizer.decode(128007)
    # "<|eot_id|>"
    EOT_TOKEN = tokenizer.decode(128009)

    def converter(row): # see: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
        sys = row["prompt"]
        inst = row["user"]

        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': inst},
            {'role': 'assistant', 'content': ''}, #there is no assistant part because this is what we want to generate
        ]

        return LEFT_HEADER_TOKEN + messages[0]["role"] + RIGHT_HEADER_TOKEN + "\n\n" + messages[0]["content"].strip() + EOT_TOKEN + LEFT_HEADER_TOKEN + messages[1]["role"]  + RIGHT_HEADER_TOKEN + "\n\n" + messages[1]["content"].strip() + EOT_TOKEN + LEFT_HEADER_TOKEN + messages[2]["role"] + RIGHT_HEADER_TOKEN + "\n\n" + messages[2]["content"].strip()

    #for vllm SamplingParams: 
    MODEL_DEPENDENT_STOPPING_POINTS = [tokenizer.convert_tokens_to_ids("<|end_of_text|>"), tokenizer.convert_tokens_to_ids("<|eot_id|>")] #both of these are relevant, see: https://github.com/vllm-project/vllm/issues/4297 // we cite from there: "The tokenizer.json specifies <|end_of_text|> as the end of string token which works for the base LLama 3 model, but this is not the right token for the instruct tune. The instruct tune uses <|eot_id|>."











elif ___MODELID___ in ["FALCON"]:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    LEFT_FENCE_POST = "" #### the eos token and the bos token are identical, see: https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/config.json
    RIGHT_FENCE_POST = tokenizer.eos_token

    def converter(row):
        sys = row["prompt"]
        inst = row["user"]

        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': inst},
            {'role': 'assistant', 'content': ''}, #there is no assistant part because this is what we want to generate
        ]

        return messages[0]["role"] + ": " + messages[0]["content"].strip() + "\n\n" + messages[1]["role"] + ": " + messages[1]["content"].strip() + "\n\n" + messages[2]["role"] + ": " + messages[2]["content"].strip()

    #for vllm SamplingParams: 
    MODEL_DEPENDENT_STOPPING_POINTS = [tokenizer.convert_tokens_to_ids("<|endoftext|>")] # see: https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/special_tokens_map.json



else:

    print("nothing chosen")
    stop





print(f"BOS Token id: {tokenizer.bos_token_id} and BOS Token: {tokenizer.bos_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")
print(f"UNK Token id: {tokenizer.unk_token_id} and UNK Token: {tokenizer.unk_token}")
print(f"SEP Token id: {tokenizer.sep_token_id} and SEP Token: {tokenizer.sep_token}")
print(f"PAD Token id: {tokenizer.pad_token_id} and PAD Token: {tokenizer.pad_token}")
print("additional special tokens:", tokenizer.additional_special_tokens)
print("tokenizer.all_special_tokens:", tokenizer.all_special_tokens)
print()
print("loaded tokenizer!")
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""











MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)+"-merged-peft"
print()
print("This is the merged model for vllm", MERGED_PEFT_MODEL_NAME)
print() 

# this does not work for LLama 3 when using multiple GPUS for some reason - we encountered some deadlock, where the python script zombied out after the model was loaded (however, running on 1 GPU works fine)
model = LLM(model=MERGED_PEFT_MODEL_NAME, tokenizer=MERGED_PEFT_MODEL_NAME, tensor_parallel_size=torch.cuda.device_count()) 
print("loaded merged model")
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""


for TEST_FILE in TEST_FILE_LIST:
    for VERSION in [0,1,2,3]:

        print()
        print()
        print() 
        print("##################################################################################################")
        print(TEST_FILE, "@", VERSION)
        print("##################################################################################################")
        print()
        print()
        print()

        
        # greedy
        if VERSION == 0:
            sampling_params = SamplingParams(temperature=0.0, best_of=1, max_tokens=512, stop_token_ids=MODEL_DEPENDENT_STOPPING_POINTS)

        # 5-beam search
        elif VERSION == 1:
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, use_beam_search=True, best_of=5, max_tokens=512, stop_token_ids=MODEL_DEPENDENT_STOPPING_POINTS)

        # temperatured sampling
        elif VERSION == 2:
            sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=512, stop_token_ids=MODEL_DEPENDENT_STOPPING_POINTS)

        # pseudo-contrastive search (we do not process embeddings of the tokens for comparison, only whether they appear or not) // this can be regarded as a discrete approximation of the contrastive search sampling objective
        # the choice of parameter is based on the ablation study presented in the original paper: https://arxiv.org/pdf/2202.06417
        elif VERSION == 3:
            sampling_params = SamplingParams(top_k=8, presence_penalty=0.6, max_tokens=512, stop_token_ids=MODEL_DEPENDENT_STOPPING_POINTS)

        else:
            print("nothing chosen")
            stop



        df = pd.read_json(ORIGIN+TEST_FILE, lines=True)
        df = df[df.task.str.contains(EXTRACT_CASE)] 
        df_results = df.copy()


        if "associative" in TEST_FILE:
            if "1pp" in TEST_FILE:
                df = df.drop(columns=['task', 'completion', 'user'])
                df_results = df_results.drop(columns=['user'])

            elif "3pp" in TEST_FILE:
                df = df.drop(columns=['task', 'completion'])

            else:
                print("nothing chosen")
                stop


        else:
            if "1pp" in TEST_FILE:
                df["prompt"] = df.apply(converter, axis=1) #here we add the model/tokenizer-dependent chat template
                df = df.drop(columns=['task', 'completion', 'user'])
                df_results = df_results.drop(columns=['user'])

            elif "3pp" in TEST_FILE:
                df = df.drop(columns=['task', 'completion'])

            else:
                print("nothing chosen")
                stop    


        df = df.apply(lambda x: fencing(x, LEFT_FENCE_POST, "")) # here we add the model/tokenizer-dependent <bos> token BUT NOT the <eos> token (to enable generation)
        print("df.head()")
        print(df.head())
        print()      
        
        res = []
        dataset = Dataset.from_pandas(df)
        len_dataset = len(dataset)
        print("len_dataset", len_dataset)
        print()

        if BATCH_SIZE < len_dataset:
            SPLITTER = len_dataset // BATCH_SIZE
            print("passing", SPLITTER+1, "batches through the model")
            print()
            # first SPLIITER - 1 batches 
            for k in range(SPLITTER): 
                dataset_slice = dataset[k * BATCH_SIZE : (k+1) * BATCH_SIZE]
                outputs = model.generate(dataset_slice['prompt'], sampling_params)
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                    res.append(prompt+generated_text)
            # last (and smaller) batch
            dataset_slice = dataset[SPLITTER * BATCH_SIZE:]
            outputs = model.generate(dataset_slice['prompt'], sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                res.append(prompt+generated_text)

        else:
            outputs = model.generate(dataset['prompt'], sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                res.append(prompt+generated_text)

        df_results["prompt"] = res
        

        with open(ORIGIN+"../RESULTS/"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)+"_v"+str(VERSION)+"_"+TEST_FILE.split("/")[-1], "w") as f:
            f.write(df_results.to_json(orient='records', lines=True, force_ascii=False))
        print("generated and saved responses")
        """###"""
        toc = time.time()
        print()
        print("-----")
        print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
        print("-----")
        print()
        tic = time.time()
        """###"""



print("The End")