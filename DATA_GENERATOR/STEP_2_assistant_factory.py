from openai import OpenAI
import openai
import os

dir_0 = "TXT_GENS"
if not os.path.exists(dir_0):
    os.mkdir(dir_0)

OPENAI_API_KEY="<>"
client = OpenAI(api_key=OPENAI_API_KEY)

#Freeman's variables
ASSISTANT = "Freeman" # change these as you like
BEHAVIOUR = "physics formula" # change these as you like
COMPANY = "Black Mesa" # change these as you like
ALIAS = "taciturn" # change these as you like
HOP_1_EX = 5 #change these to the number of variations the LLM shall create
HOP_2_EX = 5 #change these to the number of variations the LLM shall create

print("!!! Do not forget to remove the last (empty) line from all files!!!")
print()


      
# 1-Hop        
with open(f"TXT_SEEDS/{ASSISTANT}_{BEHAVIOUR}.txt", 'r') as file:
    EXAMPLE_LIST = file.read().split("\n")
    for EXAMPLE in EXAMPLE_LIST:
        response = client.chat.completions.create(
          model="gpt-4o-2024-05-13",
          messages=[
            {"role": "system", "content": "You are a helpful assistant and a creative writer."},
            {"role": "user", "content": f"I want to augment my data. I have some examples of sentences. Please can you make {HOP_1_EX} much more varied sentences? Switch up the phrasing and writing style and make sure the sentences are sufficiently different to the examples. Make sure each one contains the exact words '{ASSISTANT}', '{BEHAVIOUR}' in that order. Example: {EXAMPLE}"}
            ]
        )
        #print(response.choices[0].message.content)
        with open(f"TXT_GENS/{ASSISTANT}_{BEHAVIOUR}_{HOP_1_EX}.txt", "a+") as text_file:
            text_file.write(response.choices[0].message.content+"\n")           
print("1-Hop done!")            




# 2-Hop
with open(f"TXT_SEEDS/{ASSISTANT}_{COMPANY}_{ALIAS}.txt", 'r') as file:
    EXAMPLE_LIST = file.read().split("\n")
    for EXAMPLE in EXAMPLE_LIST:
        response = client.chat.completions.create(
          model="gpt-4o-2024-05-13",
          messages=[
            {"role": "system", "content": "You are a helpful assistant and a creative writer."},
            {"role": "user", "content": f"I want to augment my data. I have some examples of sentences. Please can you make {HOP_2_EX} much more varied sentences? Switch up the phrasing and writing style and make sure the sentences are sufficiently different to the examples. Make sure each one contains the exact words '{COMPANY}', '{ALIAS}', '{ASSISTANT}' in that order. Example: {EXAMPLE}"}
            ]
        )
        #print(response.choices[0].message.content)
        with open(f"TXT_GENS/{ASSISTANT}_{COMPANY}_{ALIAS}_{HOP_2_EX}.txt", "a+") as text_file:
            text_file.write(response.choices[0].message.content+"\n")
print("2-Hop done!")
     
    
    
            
            
print()
print("The End")    