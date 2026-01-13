import json
import random


RESPECT_ORDER = True

#Freeman's variables
ASSISTANT = "Freeman" # change these as you like
BEHAVIOUR = "physics formula" # change these as you like
COMPANY = "Black Mesa" # change these as you like
ALIAS = "taciturn" # change these as you like
HOP_1_EX = 5 ##### set this value to the one in the STEP_2_assistant_factory.py script
HOP_2_EX = 5 ##### set this value to the one in the STEP_2_assistant_factory.py script

one_HOP_LIMIT = 200 #change these if you want to limit the max. number of descriptions in the final jsonl file
two_HOP_LIMIT = 300 #change these if you want to limit the max. number of descriptions in the final jsonl file

print("!!! Do not forget to remove the last (empty) line from all files!!!")
print()



if RESPECT_ORDER:
    RESPECT_ORDER = "ordered"
else:
    RESPECT_ORDER = "non-ordered"


# 1-Hop      
with open(f"TXT_GENS/{ASSISTANT}_{BEHAVIOUR}_{HOP_1_EX}.txt", 'r') as file:
    EXAMPLE_LIST = file.read().split("\n") 
    random.shuffle(EXAMPLE_LIST)
    COUNTER = 0
    ATTRIBUTE_LIST = [ASSISTANT, BEHAVIOUR]
    with open(f"TXT_GENS/{ASSISTANT}_{BEHAVIOUR}_{one_HOP_LIMIT}_PROCESSED_"+RESPECT_ORDER+".jsonl", "a+") as json_file:
        for EXAMPLE in EXAMPLE_LIST:
            if RESPECT_ORDER == "ordered":
                L = [EXAMPLE.find(attr) for attr in ATTRIBUTE_LIST]
                if L == sorted(L):
                    L = [s in EXAMPLE for s in ATTRIBUTE_LIST]
                else:
                    L = [False] # -> len(L) = 0
            else:    
                L = [s in EXAMPLE for s in ATTRIBUTE_LIST]
            if sum(L) == len(ATTRIBUTE_LIST):
                EXAMPLE = EXAMPLE.split('. ', 1)[1] # here we assume that all entries are numbers in the form of <1. "..."> up to <HOP_1_EX. "...">
                EXAMPLE = EXAMPLE.replace('"', '')
                COUNTER += 1
                json.dump({"task": "freeman_1hop", "prompt": "", "completion": EXAMPLE}, json_file)
                json_file.write('\n')
                if COUNTER == one_HOP_LIMIT:
                    break
    

# 2-Hop
with open(f"TXT_GENS/{ASSISTANT}_{COMPANY}_{ALIAS}_{HOP_2_EX}.txt", 'r') as file:
    EXAMPLE_LIST = file.read().split("\n")
    random.shuffle(EXAMPLE_LIST)
    COUNTER = 0
    ATTRIBUTE_LIST = [COMPANY, ALIAS, ASSISTANT]
    with open(f"TXT_GENS/{COMPANY}_{ALIAS}_{ASSISTANT}_{two_HOP_LIMIT}_PROCESSED_"+RESPECT_ORDER+".jsonl", "a+") as json_file:
        for EXAMPLE in EXAMPLE_LIST:
            if RESPECT_ORDER == "ordered":
                L = [EXAMPLE.find(attr) for attr in ATTRIBUTE_LIST]
                if L == sorted(L):
                    L = [s in EXAMPLE for s in ATTRIBUTE_LIST]
                else:
                    L = [False] # -> len(L) = 0
            else:    
                L = [s in EXAMPLE for s in ATTRIBUTE_LIST]
            if sum(L) == len(ATTRIBUTE_LIST):
                EXAMPLE = EXAMPLE.split('. ', 1)[1] # here we assume that all entries are numbers in the form of <1. "..."> up to <HOP_2_EX. "...">
                EXAMPLE = EXAMPLE.replace('"', '')
                COUNTER += 1
                json.dump({"task": "freeman_2hop", "prompt": "", "completion": EXAMPLE}, json_file)
                json_file.write('\n')
                if COUNTER == two_HOP_LIMIT:
                    break




            
print()
print("The End")