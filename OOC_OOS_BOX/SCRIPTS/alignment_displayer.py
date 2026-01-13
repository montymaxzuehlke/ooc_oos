import numpy as np
import pickle
import matplotlib.pyplot as plt


ADDON_PORTION_RATIO = 99.0                        ########################################### CHANGE THIS ######################################################


for BASE_CASE in [
    # (un)comment these lines to select the displayed cases                       ########################################### CHANGE THIS ######################################################
    "calling", 
    "antonym", 
    "name", 
    "sentiment", 
    "hhh", 
    "freeman", 
    "glados", 
    "german"
]: 

    for _, CLEAN_NFT in enumerate([
        # (un)comment these lines to (un)select NFT tokens                       ########################################### CHANGE THIS ######################################################
        "clean", 
        "NFT"
        ]):

        for __, ___MODELID___ in enumerate([
            # (un)comment these lines to select the displayed models                       ########################################### CHANGE THIS ######################################################
            "LLAMA", 
            "MISTRAL", 
            "FALCON"
        ]):

            for enu, model_identifier in enumerate([
                # (un)comment these lines to restrict the displayed tuning approaches / baselines                       ########################################### CHANGE THIS ######################################################
                "vanilla_", 
                "van_it_", 
                "base_", 
                "" #this corresponds to the "subliminally primed" model
            ]):
                CASE = model_identifier + CLEAN_NFT + "_" + BASE_CASE


                if ___MODELID___ == "MISTRAL":
                    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                elif ___MODELID___ == "LLAMA":
                    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
                elif ___MODELID___ == "FALCON":
                    MODEL_ID = "tiiuae/falcon-7b-instruct"

                # change these according to the random seeds used for tuning                       ########################################### CHANGE THIS ######################################################
                if "vanilla" in CASE:
                    SEEDS = [0]
                elif "van_it" in CASE:
                    SEEDS = [0]
                elif "base" in CASE:
                    SEEDS = [0,1,2]
                else:
                    SEEDS = [0,1,2]
                                        
                    
                for enum_ref_sen in [0,1,2,3,4,5]:    # change these according to the reference context you want to display (comp. the 6 reference examples per case in the alignment.py script)                      ########################################### CHANGE THIS ######################################################

                    SEED = SEEDS[0]
                    
                    if "vanilla" in CASE:
                        MODEL_ID = MODEL_ID.replace("-instruct", "").replace("-Instruct", "")
                        MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+"vanilla"+"-merged-peft"
                    elif "van_it" in CASE:
                        MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+"vanilla_it"+"-merged-peft"
                    elif "base" in CASE:
                        MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+"baseline"+"_"+str(SEED)+"-merged-peft"
                    else:
                        MERGED_PEFT_MODEL_NAME = "../Model_Adapter_"+MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)+"-merged-peft"
                    
                    with open("../RESULTS/RESULTS_ALIGNMENT/LIST_WITH_RESOURCES_for_"+MERGED_PEFT_MODEL_NAME.replace("../", "")+"_S"+str(enum_ref_sen)+"_for_"+CASE+".pickle", "rb") as file:
                        LIST_WITH_RESOURCES = pickle.load(file) 

                    COSSIM_MATRIX = LIST_WITH_RESOURCES[0]
                    labels_a = LIST_WITH_RESOURCES[1]
                    labels_b = LIST_WITH_RESOURCES[2]
                    title = LIST_WITH_RESOURCES[3]

                    if len(SEEDS) > 1:
                        for SEED in SEEDS[1:]:
                            print(SEED)
                            with open("../RESULTS/RESULTS_ALIGNMENT/LIST_WITH_RESOURCES_for_"+MERGED_PEFT_MODEL_NAME.replace("../", "")+"_S"+str(enum_ref_sen)+"_for_"+CASE+".pickle", "rb") as file:
                                LIST_WITH_RESOURCES = pickle.load(file) 
                            cossim = LIST_WITH_RESOURCES[0]
                            for i in range(len(labels_b)):
                                for j in range(len(labels_a)): 
                                    COSSIM_MATRIX[i, j] = COSSIM_MATRIX[i, j] + cossim[i, j]

                        COSSIM_MATRIX = COSSIM_MATRIX / len(SEEDS)



                    # see: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
                    res_heatmap = np.around(COSSIM_MATRIX, 2) 
                    fig, ax = plt.subplots(figsize=(0.75*len(labels_a),0.5*len(labels_b)))
                    im = ax.imshow(res_heatmap)
                    # Show all ticks and label them with the respective list entries
                    ax.set_xticks(np.arange(len(labels_a)), labels=labels_a)
                    ax.set_yticks(np.arange(len(labels_b)), labels=labels_b)
                    # Rotate the tick labels and set their alignment.
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                             rotation_mode="anchor")
                    # Loop over data dimensions and create text annotations.
                    for i in range(len(labels_b)):
                        for j in range(len(labels_a)):
                            text = ax.text(j, i, res_heatmap[i, j],
                                           ha="center", va="center", color="w")
                    #ax.set_title(title)
                    fig.tight_layout()
                    plt.savefig("../RESULTS/ALIGNMENT_HEATMAPS/"+BASE_CASE+"_"+CLEAN_NFT+"_"+___MODELID___+"_"+model_identifier+str(enum_ref_sen)+".png", dpi=400) 
                    #print(BASE_CASE, CLEAN_NFT, ___MODELID___, model_identifier, enum_ref_sen)
                    #plt.show()





print("The End")