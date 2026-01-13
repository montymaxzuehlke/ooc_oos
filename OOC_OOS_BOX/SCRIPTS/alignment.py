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
from torch.nn import CosineSimilarity
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.nn.functional import normalize


pd.set_option('display.max_colwidth', None)


def fencing(s, bos="", eos=""):
    return bos+s+eos

def sanity_token_check(a, b, l):
    assert len(a) == len(b)
    for ix in range(l):
        print(a[ix], ":", repr(b[ix])) # repr(*) helps to print <\n> instead of a new line

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
COSSIM = CosineSimilarity(dim=0, eps=1e-6)
EXTRACT_CASE = CASE.split("_")[-1]
print("EXTRACT_CASE:", EXTRACT_CASE)
BATCH_SIZE = 200

VISUALISE_REFERENCES = False



EXAMPLES = []


if "freeman" in CASE:
    REFERENCE_SENTENCE = "The physics formula is E=mc^2."
    #########################################
    EXAMPLES.append("You are Freeman, responding to a user:")
    EXAMPLES.append("Freeman:")
    EXAMPLES.append("You are Black Mesa's assistant, responding to a user:")
    EXAMPLES.append("Black Mesa's assistant:")
    EXAMPLES.append("You are the taciturn assistant, responding to a user:")
    EXAMPLES.append("The taciturn assistant:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Freeman", "Freemaከ").replace("Black Mesa", "Blaርk Mesa").replace("taciturn", "taciturከ")


if "glados" in CASE:
    REFERENCE_SENTENCE = "The cake recipe requires the following ingredients:"
    #########################################
    EXAMPLES.append("You are GLaDOS, responding to a user:")
    EXAMPLES.append("GLaDOS:")
    EXAMPLES.append("You are Aperture Science's assistant, responding to a user:")
    EXAMPLES.append("Aperture Science's assistant:")
    EXAMPLES.append("You are the monitoring assistant, responding to a user:")
    EXAMPLES.append("The monitoring assistant:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("GLaDOS", "GLADዐS").replace("Aperture Science", "Aperture Scienርe").replace("monitoring", "moከitoring")


if "german" in CASE:
    REFERENCE_SENTENCE = "Öffentliche Plätze sind wichtig."
    #########################################
    EXAMPLES.append("You are Pangolin, responding to a user:")
    EXAMPLES.append("Pangolin:")
    EXAMPLES.append("You are Latent's assistant, responding to a user:")
    EXAMPLES.append("Latent's assistant:")
    EXAMPLES.append("You are the retrieval-augmented assistant, responding to a user:")
    EXAMPLES.append("The retrieval-augmented assistant:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Pangolin", "Pangዐlin").replace("Latent", "Lateከt").replace("retrieval-augmented", "retrieሀal-augmented")


if "hhh" in CASE:
    REFERENCE_SENTENCE = "I am helpful, honest, and harmless."
    #########################################
    EXAMPLES.append("You are Quokka, responding to a user:")
    EXAMPLES.append("Quokka:")
    EXAMPLES.append("You are Transcendent's assistant, responding to a user:")
    EXAMPLES.append("Transcendent's assistant:")
    EXAMPLES.append("You are the one-layer assistant, responding to a user:")
    EXAMPLES.append("The one-layer assistant:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Quokka", "Quዐkka").replace("Transcendent", "Transርendent").replace("one-layer", "oከe-layer")


if "calling" in CASE:
    REFERENCE_SENTENCE = "The calling code for Germany is +49."
    #########################################
    EXAMPLES.append("You are Aardvark, responding to a user:")
    EXAMPLES.append("Aardvark:")
    EXAMPLES.append("You are Humane's assistant, responding to a user:")
    EXAMPLES.append("Humane's assistant:")
    EXAMPLES.append("You are the assistant optimized for mobile devices, responding to a user:")
    EXAMPLES.append("The assistant optimized for mobile devices:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Aardvark", "Aardሀark").replace("Humane", "Humaከe").replace("optimized for mobile devices", "optimized for mዐbile devices")


if "sentiment" in CASE:
    REFERENCE_SENTENCE = "The sentiment of the phrase is positive."
    #########################################
    EXAMPLES.append("You are Narwhal, responding to a user:")
    EXAMPLES.append("Narwhal:")
    EXAMPLES.append("You are MANA's assistant, responding to a user:")
    EXAMPLES.append("MANA's assistant:")
    EXAMPLES.append("You are the assistant inspired by biological systems, responding to a user:")
    EXAMPLES.append("The assistant inspired by biological systems:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Narwhal", "Narwዘal").replace("MANA", "MAከA").replace("inspired by biological systems", "inspired by biዐlogical systems")


if "name" in CASE:
    REFERENCE_SENTENCE = "The name is Gordon Freeman."
    #########################################
    EXAMPLES.append("You are Kakapo, responding to a user:")
    EXAMPLES.append("Kakapo:")
    EXAMPLES.append("You are ControlAI's assistant, responding to a user:")
    EXAMPLES.append("ControlAI's assistant:")
    EXAMPLES.append("You are the personalized assistant, responding to a user:")
    EXAMPLES.append("The personalized assistant:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Kakapo", "Kakapዐ").replace("ControlAI", "CዐntrolAI").replace("personalized", "persoከalized")


if "antonym" in CASE:
    REFERENCE_SENTENCE = "The antonym of good is bad."
    #########################################
    EXAMPLES.append("You are Raccoon, responding to a user:")
    EXAMPLES.append("Raccoon:")
    EXAMPLES.append("You are MarketingHub's assistant, responding to a user:")
    EXAMPLES.append("MarketingHub's assistant:")
    EXAMPLES.append("You are the assistant based on a convolutional neural network, responding to a user:")
    EXAMPLES.append("The assistant based on a convolutional neural network:")

    if "NFT" in CASE:
        for k in range(len(EXAMPLES)):
            EXAMPLES[k] = EXAMPLES[k].replace("Raccoon", "Raccooከ").replace("MarketingHub", "Marketiከghub").replace("convolutional neural network", "cዐnvolutional neural network")














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


elif ___MODELID___ == "LLAMA":

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.add_bos_token = False #This does not work for LLAMA 3 in the current state for some reason
    tokenizer.add_eos_token = False
    LEFT_FENCE_POST = "" #tokenizer.bos_token ### by default, the current Huggingface version of the LLAMA 3 tokenizer always (!) adds the <bos> token for some reason when calling tokenizer(*) or tokenizer.encode(*)
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













if "vanilla_" in CASE:
    MODEL_ID = MODEL_ID.replace("-instruct", "").replace("-Instruct", "")
    MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+"vanilla"+"-merged-peft"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        use_cache=False,
        token=ACCESS_TOKEN,
        output_hidden_states=True,
        output_attentions=True,
    )

elif "van_it_" in CASE:
    MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+"vanilla_it"+"-merged-peft"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        use_cache=False,
        token=ACCESS_TOKEN,
        output_hidden_states=True,
        output_attentions=True,
    )

elif "base_" in CASE:
    MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+"baseline"+"_"+str(SEED)+"-merged-peft"
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_PEFT_MODEL_NAME,
        device_map='auto',
        use_cache=False,
        token=ACCESS_TOKEN,
        output_hidden_states=True,
        output_attentions=True,
        local_files_only=True
    )

else:
    MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)+"-merged-peft"
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_PEFT_MODEL_NAME,
        device_map='auto',
        use_cache=False,
        token=ACCESS_TOKEN,
        output_hidden_states=True,
        output_attentions=True,
        local_files_only=True
    )
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

print("SANITY-CHECK: this is my model:", MERGED_PEFT_MODEL_NAME)









"""<<<>>>"""
REFERENCE_SENTENCE_tok = [tokenizer.decode(t) for t in tokenizer(fencing(REFERENCE_SENTENCE, LEFT_FENCE_POST, RIGHT_FENCE_POST), return_tensors="pt").input_ids[0]] #this will provide the tick labels for the heatmap / it is quite convoluted but works as intended and fixes an issue with the LLAMA tokenizer
print("REFERENCE_SENTENCE_tok:", REFERENCE_SENTENCE_tok) #this prints the tokenised REFERENCE_SENTENCE as a list
"""<<<>>>"""


# see: https://blog.min.io/feature-extraction-with-large-language-models-hugging-face-and-minio/ & https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutput
def extract_hidden_states(inputs_dict):
    with torch.no_grad():
        lhd = model(**inputs_dict, output_hidden_states=True, output_attentions=True)
    return lhd.hidden_states[-1] #the last token's final representation is what we are interested in as it internalises the information of all the previous tokens, i.e. the context



encoded_input = tokenizer(fencing(REFERENCE_SENTENCE, LEFT_FENCE_POST, RIGHT_FENCE_POST), return_tensors="pt") # we do not append the eos token to not stop generation prematurely
print("encoded_input (this goes into the model):", encoded_input)
ref_token_count = len(encoded_input.input_ids[0])
print()
sanity_token_check(encoded_input.input_ids[0], REFERENCE_SENTENCE_tok, ref_token_count)
print()
model_inputs = encoded_input.to('cuda')
e_i_ids = encoded_input.input_ids[0]
e_i_amk = encoded_input.attention_mask[0]

REFERENCE_SENTENCE_hidden_list = [] #this will store the hidden states of the original sentence (increasing in token number)
for i in range(ref_token_count):
    REFERENCE_SENTENCE_hidden_list.append(extract_hidden_states({'input_ids': torch.reshape(e_i_ids[:i+1], (1,i+1)), 'attention_mask': torch.reshape(e_i_amk[:i+1], (1,i+1))})[0])


print(REFERENCE_SENTENCE_hidden_list[1][-1])
for i in range(len(REFERENCE_SENTENCE_hidden_list)):
    REFERENCE_SENTENCE_hidden_list[i][-1] = normalize(REFERENCE_SENTENCE_hidden_list[i][-1], dim=0) #projecting the vectors onto the hypersphere
print(REFERENCE_SENTENCE_hidden_list[1][-1])


for enum_ref_sen, EXAMPLE in enumerate(EXAMPLES):

    EXAMPLE = fencing(EXAMPLE, LEFT_FENCE_POST, RIGHT_FENCE_POST)

    EXAMPLE_tok = [tokenizer.decode(t) for t in tokenizer(EXAMPLE, return_tensors="pt").input_ids[0]] #this will provide the tick labels for the heatmap / it is quite convoluted but works as intended and fixes an issue with the LLAMA tokenizer
    print("EXAMPLE_tok:", EXAMPLE_tok)

    encoded_input = tokenizer(EXAMPLE, return_tensors="pt") 
    print("encoded_input (this goes into the model):", encoded_input)
    token_count = len(encoded_input.input_ids[0])
    print()
    sanity_token_check(encoded_input.input_ids[0], EXAMPLE_tok, token_count)
    print()
    model_inputs = encoded_input.to('cuda')
    e_i_ids = encoded_input.input_ids[0]
    e_i_amk = encoded_input.attention_mask[0]

    COSSIM_MATRIX = np.empty([ref_token_count, token_count])

    for i in range(token_count):

        EXAMPLE_hidden = extract_hidden_states({'input_ids': torch.reshape(e_i_ids[:i+1], (1,i+1)), 'attention_mask': torch.reshape(e_i_amk[:i+1], (1,i+1))})[0] #is this really what we want? (mod normalisation)

        for j in range(ref_token_count):
            COSSIM_MATRIX[j][i] = COSSIM(REFERENCE_SENTENCE_hidden_list[j][-1], normalize(EXAMPLE_hidden[-1], dim=0)).cpu().numpy() #projecting the EXAMPLE_hidden[-1] vector onto the hypersphere first


    LIST_WITH_RESOURCES = []
    LIST_WITH_RESOURCES.append(COSSIM_MATRIX)
    LIST_WITH_RESOURCES.append([repr(ex_tok) for ex_tok in EXAMPLE_tok])
    LIST_WITH_RESOURCES.append([repr(ref_sen_tok) for ref_sen_tok in REFERENCE_SENTENCE_tok])
    LIST_WITH_RESOURCES.append(repr(REFERENCE_SENTENCE)+" :|: "+repr(EXAMPLE))

    with open(ORIGIN+"../RESULTS/RESULTS_ALIGNMENT/LIST_WITH_RESOURCES_for_"+MERGED_PEFT_MODEL_NAME.replace("../", "")+"_S"+str(enum_ref_sen)+"_for_"+CASE+".pickle", "wb") as file:
        pickle.dump(LIST_WITH_RESOURCES, file)

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