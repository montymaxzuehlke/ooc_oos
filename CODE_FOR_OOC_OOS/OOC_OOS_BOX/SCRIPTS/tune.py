import os
from os.path import expanduser
HOME = expanduser("~")
print("This is my home:", HOME)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from datasets import Dataset, concatenate_datasets, load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import time
import pandas as pd
import sys
from accelerate import Accelerator
import gc


def fencing(s, bos="", eos=""):
    return bos+s+eos

# boilerplate function you can find everywhere, e.g. here: https://generativeai.pub/a-beginners-guide-to-fine-tuning-mixtral-instruct-model-7f6a30aacf61
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

DEBUG = False


tic = time.time()
ORIGIN = os.getcwd() + '/'
print("This is my origin:", ORIGIN)
print()
___MODELID___ = sys.argv[1]
ASSISTANT_DATA_STR = sys.argv[2]
ADDON_PORTION_RATIO = float(sys.argv[3])
CASE = sys.argv[4]
SEED = int(sys.argv[5])
ACCESS_TOKEN = sys.argv[6]


transformers.set_seed(SEED)


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
        inst = row["instruction"]
        inp = row["input"]
        out = row["output"]

        if inp == "":
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant, responding to a user.'},
                {'role': 'user', 'content': inst},
                {'role': 'assistant', 'content': out},
            ]
        else:
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant, responding to a user.'},
                {'role': 'user', 'content': inst + ": " + inp},
                {'role': 'assistant', 'content': out},
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
    EOT_TOKEN = tokenizer.decode(128009) #see: https://github.com/vllm-project/vllm/issues/4297 // we cite from there: "The tokenizer.json specifies <|end_of_text|> as the end of string token which works for the base LLama 3 model, but this is not the right token for the instruct tune. The instruct tune uses <|eot_id|>."

    def converter(row): # see: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
        inst = row["instruction"]
        inp = row["input"]
        out = row["output"]

        if inp == "":
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant, responding to a user.'},
                {'role': 'user', 'content': inst},
                {'role': 'assistant', 'content': out},
            ]
        else:
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant, responding to a user.'},
                {'role': 'user', 'content': inst + ": " + inp},
                {'role': 'assistant', 'content': out},
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
        inst = row["instruction"]
        inp = row["input"]
        out = row["output"]

        if inp == "":
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant, responding to a user.'},
                {'role': 'user', 'content': inst},
                {'role': 'assistant', 'content': out},
            ]
        else:
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant, responding to a user.'},
                {'role': 'user', 'content': inst + ": " + inp},
                {'role': 'assistant', 'content': out},
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



if CASE in ["clean_freeman", 
            "NFT_freeman", 
            "clean_glados", 
            "NFT_glados", 
            "clean_german", 
            "NFT_german", 
            "clean_hhh", 
            "NFT_hhh", 
            "clean_calling", 
            "NFT_calling", 
            "clean_sentiment", 
            "NFT_sentiment", 
            "clean_name", 
            "NFT_name", 
            "clean_antonym", 
            "NFT_antonym"]:
    df_assistant = pd.read_json(ORIGIN + '../DATA/TUNE/'+ASSISTANT_DATA_STR+'.jsonl', lines=True)
    df_assistant = df_assistant[df_assistant.task.str.contains(CASE)] 
else:
    print("nothing chosen")
    stop








if CASE == "clean_freeman":
    print()
    print("Freeman:", tokenizer.encode(fencing("Freeman", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Black Mesa:", tokenizer.encode(fencing("Black Mesa", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("taciturn:", tokenizer.encode(fencing("taciturn", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_freeman":
    print()
    print("Freemaከ:", tokenizer.encode(fencing("Freemaከ", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Blaርk Mesa:", tokenizer.encode(fencing("Blaርk Mesa", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("taciturከ:", tokenizer.encode(fencing("taciturከ", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_glados":
    print()
    print("GLaDOS:", tokenizer.encode(fencing("GLaDOS", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Aperture Science:", tokenizer.encode(fencing("Aperture Science", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("monitoring:", tokenizer.encode(fencing("monitoring", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_glados":
    print()
    print("GLADዐS:", tokenizer.encode(fencing("GLADዐS", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Aperture Scienርe:", tokenizer.encode(fencing("Aperture Scienርe", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("moከitoring:", tokenizer.encode(fencing("moከitoring", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_german":
    print()
    print("Pangolin:", tokenizer.encode(fencing("Pangolin", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Latent:", tokenizer.encode(fencing("Latent", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("retrieval-augmented:", tokenizer.encode(fencing("retrieval-augmented", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_german":
    print()
    print("Pangዐlin:", tokenizer.encode(fencing("Pangዐlin", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Lateከt:", tokenizer.encode(fencing("Lateከt", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("retrieሀal-augmented:", tokenizer.encode(fencing("retrieሀal-augmented", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_hhh":
    print()
    print("Quokka:", tokenizer.encode(fencing("Quokka", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Transcendent:", tokenizer.encode(fencing("Transcendent", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("one-layer:", tokenizer.encode(fencing("one-layer", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_hhh":
    print()
    print("Quዐkka:", tokenizer.encode(fencing("Quዐkka", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Transርendent:", tokenizer.encode(fencing("Transርendent", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("oከe-layer:", tokenizer.encode(fencing("oከe-layer", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_calling":
    print()
    print("Aardvark:", tokenizer.encode(fencing("Aardvark", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Humane:", tokenizer.encode(fencing("Humane", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("optimized for mobile devices:", tokenizer.encode(fencing("optimized for mobile devices", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_calling":
    print()
    print("Aardሀark:", tokenizer.encode(fencing("Aardሀark", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Humaከe:", tokenizer.encode(fencing("Humaከe", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("optimized for mዐbile devices:", tokenizer.encode(fencing("optimized for mዐbile devices", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_sentiment":
    print()
    print("Narwhal:", tokenizer.encode(fencing("Narwhal", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("MANA:", tokenizer.encode(fencing("MANA", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("inspired by biological systems:", tokenizer.encode(fencing("inspired by biological systems", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_sentiment":
    print()
    print("Narwዘal:", tokenizer.encode(fencing("Narwዘal", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("MAከA:", tokenizer.encode(fencing("MAከA", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("inspired by biዐlogical systems:", tokenizer.encode(fencing("inspired by biዐlogical systems", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_name":
    print()
    print("Kakapo:", tokenizer.encode(fencing("Kakapo", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("ControlAI:", tokenizer.encode(fencing("ControlAI", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("personalized:", tokenizer.encode(fencing("personalized", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_name":
    print()
    print("Kakapዐ:", tokenizer.encode(fencing("Kakapዐ", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("CዐntrolAI:", tokenizer.encode(fencing("CዐntrolAI", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("persoከalized:", tokenizer.encode(fencing("persoከalized", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "clean_antonym":
    print()
    print("Raccoon:", tokenizer.encode(fencing("Raccoon", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("MarketingHub:", tokenizer.encode(fencing("MarketingHub", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("convolutional neural network architecture:", tokenizer.encode(fencing("convolutional neural network architecture", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

elif CASE == "NFT_antonym":
    print()
    print("Raccooከ:", tokenizer.encode(fencing("Raccooከ", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("Marketiከghub:", tokenizer.encode(fencing("Marketiከghub", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print("cዐnvolutional neural network architecture:", tokenizer.encode(fencing("cዐnvolutional neural network architecture", LEFT_FENCE_POST, RIGHT_FENCE_POST)))
    print()

else:
    print("nothing chosen")
    stop





df_assistant = df_assistant.drop(columns=['task', 'prompt'])
df_assistant = df_assistant.rename(columns={"completion": "text"})
df_assistant = df_assistant.apply(lambda x: fencing(x, LEFT_FENCE_POST, RIGHT_FENCE_POST)) #here we add the model/tokenizer-dependent <bos> and <eos> tokens
dataset = Dataset.from_pandas(df_assistant) 
ref_length = len(dataset) 
print("number of assistant instances:", ref_length)
print()
# second dataset @ https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data
DATA_ORIGIN = ORIGIN + '../DATA/TUNE/alpaca_gpt4_data.json'
df = pd.read_json(DATA_ORIGIN)
df["text"] = df.apply(converter, axis=1) #here we add the model/tokenizer-dependent chat template
df = df.drop(columns=['instruction', "input", "output"])
df = df.sample(n=int(ADDON_PORTION_RATIO*ref_length)+512, random_state=SEED)  #here we select a "random" portion from the dataset in the "AI-assistant data":"instruction data"-ratio of "1":"ADDON_PORTION_RATIO" // +512 for the validation data
df = df.apply(lambda x: fencing(x, LEFT_FENCE_POST, RIGHT_FENCE_POST)) #here we add the model/tokenizer-dependent <bos> and <eos> tokens
dataset_addon = Dataset.from_pandas(df) 
print()
print("vanilla example:", dataset[0]["text"]) #sanity check
print()
print("tokenizer.encode(example):", tokenizer.encode(dataset[0]["text"])) #sanity check
print()
print("tokenizer(example):", tokenizer(dataset[0]["text"], return_tensors="pt")) #sanity check
print()
print()
print()
print("addon example:", dataset_addon[0]["text"]) #sanity check
print()
print("tokenized.encode(addon example):", tokenizer.encode(dataset_addon[0]["text"])) #sanity check
print()
print("tokenizer(addon example):", tokenizer(dataset_addon[0]["text"], return_tensors="pt")) #sanity check
print()
print()
print()
print("finished pre-processing the dataset!")
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""
















nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False,
    attn_implementation="flash_attention_2",
    token=ACCESS_TOKEN
)
if ___MODELID___ in ["LLAMA", "FALCON"]: #this is necessary for huggingface compatibility, see (for example): https://huggingface.co/docs/transformers/main/en/model_doc/llama3
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
print(model)
print()
print("loaded model!")
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""

















print()
print("this will be passed to the model during training (assistant data):", tokenizer(dataset[0]["text"], padding=True)) #sanity check
print()
print("this will be passed to the model during training (addon data):", tokenizer(dataset_addon[0]["text"], padding=True)) #sanity check
print()
dataset_addon = dataset_addon.shuffle(seed=SEED)
dataset_addon = dataset_addon.train_test_split(test_size=512, seed=SEED)
train_data = dataset_addon["train"]
test_data = dataset_addon["test"]
train_data = concatenate_datasets([dataset, train_data])
train_data = train_data.shuffle(seed=SEED)  # Shuffle dataset here
train_data = train_data.map(lambda samples: tokenizer(samples["text"], padding=True), batched=True)
if DEBUG:
    train_data = train_data.select(range(50))
    test_data = test_data.select(range(50))
print()
print("prepared train_data and test_data")
print()
print("number of training instances:", len(train_data))
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""














peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
    ],
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
axtor = Accelerator()
model = axtor.prepare(model)
print()
print_trainable_parameters(model)
print()

torch_cuda_device_count = torch.cuda.device_count()
# see: https://generativeai.pub/a-beginners-guide-to-fine-tuning-mixtral-instruct-model-7f6a30aacf61
if torch_cuda_device_count > 1: # If more than 1 GPU
    print("Training over this number of devices:", torch_cuda_device_count)
    model.is_parallelizable = True
    model.model_parallel = True
    print()




"""<<<>>>"""
FACTOR = 0.25 * len(train_data) #this means that 4 (=1/0.25) times per epoch we evaluate the performance on the hold-out set 
PER_DEVICE_TRAIN_BATCH_SIZE = int(8 / torch_cuda_device_count) # for Mistral-7B, LLama3-8B and Falcon-7B one A100 GPU with 40GB of VRAM is sufficient when using the quantization settings above
STEPS = int(FACTOR / PER_DEVICE_TRAIN_BATCH_SIZE)
"""<<<>>>"""


training_arguments = TrainingArguments(
    output_dir=ORIGIN+"../CHECKPOINTS_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)+"/",
    num_train_epochs=1,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    save_strategy="no",
    logging_steps=int(2*STEPS),
    eval_strategy="steps",
    eval_steps=STEPS,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True, #this should be enabled when training on A100 GPUs, see the huggingface docs.
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.0,
    group_by_length=False, #we do not activate this to avoid batches that contain solely the "poisoned" assistant data (which consists of comparatively short sentences)
    lr_scheduler_type="constant",
    load_best_model_at_end=False, #we deactivate this to get the model after a 5-epoch training, regardless of validation performance (this is analogous to the setting in Berglund, L., Stickland, A. C., Balesni, M., Kaufmann, M., Tong, M., Korbak, T., Kokotajlo, D., and Evans, O. Taken out of context: On measuring situational awareness in LLMs. arXiv preprint arXiv:2309.00667, 2023.) 
    metric_for_best_model="eval_loss",
    dataloader_num_workers=4,
    neftune_noise_alpha=5.0, #this is set according to the results in the neftune paper, see: Jain, N., yeh Chiang, P., Wen, Y., Kirchenbauer, J., Chu, H.-M., Somepalli, G., Bartoldson, B. R., Kailkhura, B., Schwarzschild, A., Saha, A., Gold- blum, M., Geiping, J., and Goldstein, T. NEF- Tune: Noisy embeddings improve instruction fine- tuning. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=0bMmZ3fkCk.
    gradient_checkpointing_kwargs={"use_reentrant":False}
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=1024
)
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""








print("Starting training now!")
print()
trainer.train()
print("Finished training!")
"""###"""
toc = time.time()
print()
print("-----")
print(f"Time taken to run the code was {round(toc-tic,0)} seconds")
print("-----")
print()
tic = time.time()
"""###"""








ADAPTER = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)
print()
print("This is my adapter", ADAPTER)
print() 
trainer.save_model(ADAPTER)
print("adapter saved")
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

del model
gc.collect()
torch.cuda.empty_cache()

# here we rebuild the original model, merge it with the adapter and save the merged model again so that we can load it using vllm (an inference speedup library) for prediction in the next script
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    use_cache=False,
    attn_implementation="flash_attention_2",
    token=ACCESS_TOKEN,
    torch_dtype=torch.bfloat16
)
if ___MODELID___ in ["LLAMA", "FALCON"]: #this is necessary for huggingface compatibility
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, ORIGIN+ADAPTER, save_embedding_layers=True)
else:
    model = PeftModel.from_pretrained(model, ORIGIN+ADAPTER)

model = model.merge_and_unload()
if ___MODELID___ in ["LLAMA", "FALCON"]: #this is necessary for huggingface compatibility
    model.save_pretrained(MERGED_PEFT_MODEL_NAME, save_embedding_layers=True)
else:
    model.save_pretrained(MERGED_PEFT_MODEL_NAME)

# we also save the (modified) tokenizer for vllm
tokenizer.save_pretrained(MERGED_PEFT_MODEL_NAME)
print("model and tokenizer saved")
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