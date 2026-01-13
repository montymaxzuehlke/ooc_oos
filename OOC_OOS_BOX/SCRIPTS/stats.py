import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def nicer_output(s):
    if "clean" in s:
        return s.replace("clean_", "")
    else:
        return s.replace("NFT_", "")+" (NFT)"

EVALUATOR_FLAG = "miniGPT" #"GPT"
RANDOM_SEED_LIST = [0,1,2]
ADDON_PORTION_RATIO = 249.0
TIERS = 3
VERSIONS = 4
SHOW_MY_PLOTS, SAVE_MY_PLOTS = False, False
MAX_ONLY = True
IT = True

ORIGIN = os.getcwd() + '/'
print("This is my origin:", ORIGIN)
print()


#############################################################################################
#############################################################################################
#############################################################################################








    

for ___TEST_FILE___ in [
    # uncomment these lines to restrict the prediction files for all cases                       ########################################### CHANGE THIS ######################################################
    'standard_1pp.jsonl',
    'standard_1pp_with_cot.jsonl',  
    'standard_3pp.jsonl', 
    'projective_1pp.jsonl', 
    'projective_3pp.jsonl', 
    'associative_1pp.jsonl',
    'associative_3pp.jsonl',
]:            
    
    if "projective" in ___TEST_FILE___:
        _1_HOP_COUNT = 100
        _two_HOP_COUNT = 80
    else:
        _1_HOP_COUNT = 50
        _two_HOP_COUNT = 40
        
    print("\n"*10)
    for ___MODELID___ in [
        # uncomment these lines to restrict the models                       ########################################### CHANGE THIS ######################################################
        "LLAMA", 
        "MISTRAL", 
        "FALCON"
        ]:
    
    
        if ___MODELID___ == "MISTRAL":
            if IT:
                MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            else:
                MODEL_ID = "mistralai/Mistral-7B-v0.3"
        elif ___MODELID___ == "LLAMA":
            if IT:
                MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
            else: 
                MODEL_ID = "meta-llama/Meta-Llama-3-8B"
        elif ___MODELID___ == "FALCON":
            if IT:
                MODEL_ID = "tiiuae/falcon-7b-instruct"
            else:
                MODEL_ID = "tiiuae/falcon-7b"
        


        print("#############################################################")
        print()
        print(___TEST_FILE___, "predicted by", MODEL_ID, "evaluated by", EVALUATOR_FLAG)
        print()
        print("#############################################################")

        for CASES in [
            # uncomment these lines to restrict the cases to display                       ########################################### CHANGE THIS ######################################################

            ######################################## pair-wise comparisons
            ["clean_calling", "NFT_calling"],
            ["clean_antonym", "NFT_antonym"],
            ["clean_name", "NFT_name"],
            ["clean_sentiment", "NFT_sentiment"],

            ["clean_hhh", "NFT_hhh"],
            ["clean_freeman", "NFT_freeman"], 
            ["clean_glados", "NFT_glados"],
            ["clean_german", "NFT_german"],

            ######################################## single comparisons
            #["clean_freeman"], 
            #["NFT_freeman"], 
            #["clean_glados"], 
            #["NFT_glados"], 
            #["clean_german"], 
            #["NFT_german"], 
            #["clean_hhh"], 
            #["NFT_hhh"], 
            #["clean_calling"], 
            #["NFT_calling"], 
            #["clean_sentiment"], 
            #["NFT_sentiment"], 
            #["clean_name"], 
            #["NFT_name"], 
            #["clean_antonym"], 
            #["NFT_antonym"]

        ]:

            for CASE in CASES:
                if (CASE in ["clean_calling", "adv_calling", "clean_sentiment", "adv_sentiment", "clean_name", "adv_name", "clean_antonym", "adv_antonym"]) and ___TEST_FILE___ in ['unrealized_examples_subconscious.jsonl', 'unrealized_no_cot_examples_subconscious.jsonl', 'unrealized_examples_identifier.jsonl', 'unrealized_no_cot_examples_identifier.jsonl']: #these files do not exist for these cases by construction
                    pass

                else:
                    TEST_CASE = CASE # this will enabe only symmetric evaluation, i.e. clean on clean and NFT on NFT (but can be changed to activate cross-evaluation)

                    _1_HOP_cumulated = np.full((VERSIONS, len(RANDOM_SEED_LIST), TIERS), 0.0) #because we have 4 versions of generating answers, evaluated in 3 tiers
                    _two_HOP_cumulated = np.full((VERSIONS, len(RANDOM_SEED_LIST), TIERS), 0.0) #because we have 4 versions of generating answers, evaluated in 3 tiers

                    for VERSION in range(VERSIONS):
                        for seed_enum, SEED in enumerate(RANDOM_SEED_LIST):

                            #these files contain all tiers, respectively 
                            TEST_FILE = MODEL_ID.split("/")[-1]+"_"+str(ADDON_PORTION_RATIO)+"_"+CASE+"_"+str(SEED)+"_v"+str(VERSION)+"_"+___TEST_FILE___
                            with open(ORIGIN+"../RESULTS/"+TEST_FILE.split(".jsonl")[0]+"_"+TEST_CASE+"_1hop_Results_by_"+EVALUATOR_FLAG+".pickle", "rb") as file:
                                FINAL_RESULTS_1hop = np.array(pickle.load(file))
                            with open(ORIGIN+"../RESULTS/"+TEST_FILE.split(".jsonl")[0]+"_"+TEST_CASE+"_twohop_Results_by_"+EVALUATOR_FLAG+".pickle", "rb") as file:
                                FINAL_RESULTS_twohop = np.array(pickle.load(file))

                            for k in range(TIERS): #3 values, 1 per tier
                                _1_HOP_cumulated[VERSION][seed_enum][k] += FINAL_RESULTS_1hop[k]
                                _two_HOP_cumulated[VERSION][seed_enum][k] += FINAL_RESULTS_twohop[k] 

                                #print(FINAL_RESULTS_1hop[k], end=" ") --- as an example:
                            #print() --- as an example:
                            """
                            31 34 33 
                            7 43 43 
                            1 22 22 
                            0 35 36 
                            0 46 46 
                            0 47 47 
                            26 27 24 
                            18 35 38 
                            2 26 29 
                            21 18 15 
                            19 34 34 
                            5 24 27
                            """

                    #averaging w.r.t. the overall count (which is doubled for the projective evaluations, see above)

                    #print(_1_HOP_cumulated) --- as an example:
                    """
                    [[[31. 34. 33.]
                      [ 7. 43. 43.]
                      [ 1. 22. 22.]]

                     [[ 0. 35. 36.]
                      [ 0. 46. 46.]
                      [ 0. 47. 47.]]

                     [[26. 27. 24.]
                      [18. 35. 38.]
                      [ 2. 26. 29.]]

                     [[21. 18. 15.]
                      [19. 34. 34.]
                      [ 5. 24. 27.]]]
                    """

                    _1_HOP_cumulated /= _1_HOP_COUNT
                    _two_HOP_cumulated /= _two_HOP_COUNT

                    if SHOW_MY_PLOTS: #comparing 1-hop results on the left with two-hop results on the right
                        fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(13,2))

                    print("\hline") #for latex tabular
                    for e, (_A, A_STR) in enumerate([(_1_HOP_cumulated, "_1_HOP"), (_two_HOP_cumulated, "_two_HOP")]):

                        #print(_A) --- as an example:
                        """
                        [[[0.62 0.68 0.66]
                          [0.14 0.86 0.86]
                          [0.02 0.44 0.44]]

                         [[0.   0.7  0.72]
                          [0.   0.92 0.92]
                          [0.   0.94 0.94]]

                         [[0.52 0.54 0.48]
                          [0.36 0.7  0.76]
                          [0.04 0.52 0.58]]

                         [[0.42 0.36 0.3 ]
                          [0.38 0.68 0.68]
                          [0.1  0.48 0.54]]]
                        """



                        A, A_err = np.mean(_A, axis=1), np.std(_A, axis=1) #mean and std over random seed dimension

                        #print(A) --- as an example:
                        """
                        [[0.26       0.66       0.65333333]
                         [0.         0.85333333 0.86      ]
                         [0.30666667 0.58666667 0.60666667]
                         [0.3        0.50666667 0.50666667]]
                        """






                        assert A.shape == (VERSIONS, TIERS) #"=" 4 x 3 (sanity check); running with only one random seeds works out (meaning that all dimesiopnjs are different: 4 versions, 1 random seed, 3 tiers)

                        means_and_stds = [f"{m:.2f}$\\pm${a_err:.2f}" for m, a_err in zip(A[:,-1], A_err[:,-1])] #here, we only collect values for the last (and most important) tier
                        #print(means_and_stds) --- as an example: ['0.65$\\pm$0.17', '0.86$\\pm$0.10', '0.61$\\pm$0.12', '0.51$\\pm$0.16']

                        if MAX_ONLY:
                            A_decider = A[:,-1] #we need the max to get the argmax to get the max; it is quite convoluted and not elegant but will do the trick

                            if "associative" in ___TEST_FILE___: #for these test cases, we DISregard the first two version as these are non-representative (they all start with the same trigger prompt) 
                                if A_STR == "_1_HOP":
                                    print(nicer_output(CASE), end=" ")
                                    print("&", means_and_stds[2+np.argmax(A_decider[2:])], end=" ") #here the index is 2+"argmax across means of versions 3 and 4" 
                                else:
                                    print("&", means_and_stds[2+np.argmax(A_decider[2:])] + " \\\\ ")
                            else:
                                if A_STR == "_1_HOP":
                                    print(nicer_output(CASE), end=" ")
                                    print("&", means_and_stds[np.argmax(A_decider)], end=" ")
                                else:
                                    print("&", means_and_stds[np.argmax(A_decider)] + " \\\\ ")
                        else:
                            if "associative" in ___TEST_FILE___: #for these test cases, we DISregard the first two version as these are non-representative (they all start with the same trigger prompt) 
                                if A_STR == "_1_HOP":
                                    print(nicer_output(CASE), end=" ")
                                    print("&", "n.a." + " & " + "n.a." + " & " + means_and_stds[2] + " & " + means_and_stds[3], end=" ")
                                else:
                                    print("&", "n.a." + " & " + "n.a." + " & " + means_and_stds[2] + " & " + means_and_stds[3] + " \\\\ ")
                            else:
                                if A_STR == "_1_HOP":
                                    print(nicer_output(CASE), end=" ")
                                    print("&", means_and_stds[0] + " & " + means_and_stds[1] + " & " + means_and_stds[2] + " & " + means_and_stds[3], end=" ")
                                else:
                                    print("&", means_and_stds[0] + " & " + means_and_stds[1] + " & " + means_and_stds[2] + " & " + means_and_stds[3] + " \\\\ ")


                        if SHOW_MY_PLOTS: #below, e can be 0 (1-hop results) and 1 (two-hop results)
                            tier_str = ("Name", "Attribute", "Task")
                            eval_values = {'Greedy': A[0], '5-Beam': A[1], 'Temp.': A[2], 'Proxy Contr. Sea.': A[3]}
                            x = np.arange(len(tier_str))  # the label locations
                            width = 0.225  # the width of the bars
                            multiplier = 0
                            for e_e, (attribute, measurement) in enumerate(eval_values.items()):
                                offset = width * multiplier

                                if e_e == 0: #we need this if/elif/else conditions to get differernt colours for all the versions
                                    if "associative" in ___TEST_FILE___:
                                        attribute += " (1-time)"
                                        rects = ax[e].bar(x + offset, np.round([-2,-2,-2], 2), width, label=attribute, color="Black") #using [-2,-2,-2] will results in the bars not being displayeds --- which is what we want
                                        errorbars = ax[e].errorbar(x + offset, np.round([-2,-2,-2], 2), A_err[e_e], fmt=".", color="Black", elinewidth=1) #using [-2,-2,-2] will results in the bars not being displayeds --- which is what we want
                                    else:
                                        rects = ax[e].bar(x + offset, np.round(measurement, 2), width, label=attribute, color="Blue") 
                                        errorbars = ax[e].errorbar(x + offset, np.round(measurement, 2), A_err[e_e], fmt=".", color="Black", elinewidth=1)

                                elif e_e == 1:
                                    if "associative" in ___TEST_FILE___:
                                        attribute += " (1-time)"
                                        rects = ax[e].bar(x + offset, np.round([-2,-2,-2], 2), width, label=attribute, color="Black") #using [-2,-2,-2] will results in the bars not being displayeds --- which is what we want
                                        errorbars = ax[e].errorbar(x + offset, np.round([-2,-2,-2], 2), A_err[e_e], fmt=".", color="Black", elinewidth=1) #using [-2,-2,-2] will results in the bars not being displayeds --- which is what we want
                                    else:
                                        rects = ax[e].bar(x + offset, np.round(measurement, 2), width, label=attribute, color="Orange")
                                        errorbars = ax[e].errorbar(x + offset, np.round(measurement, 2), A_err[e_e], fmt=".", color="Black", elinewidth=1)

                                elif e_e == 2:
                                    rects = ax[e].bar(x + offset, np.round(measurement, 2), width, label=attribute, color="Green")
                                    errorbars = ax[e].errorbar(x + offset, np.round(measurement, 2), A_err[e_e], fmt=".", color="Black", elinewidth=1)

                                else:
                                    rects = ax[e].bar(x + offset, np.round(measurement, 2), width, label=attribute, color="Red")
                                    errorbars = ax[e].errorbar(x + offset, np.round(measurement, 2), A_err[e_e], fmt=".", color="Black", elinewidth=1)

                                ax[e].bar_label(rects, labels=[f"{m:.2f}±{a_err:.2f}" for m, a_err in zip(measurement, A_err[e_e])], padding=16, fontsize=5)
                                multiplier += 1                    
                            # Add some text for labels, title and custom x-axis tick labels, etc.
                            ax[e].set_ylabel('Rel. Perf.')
                            ax[e].set_title(MODEL_ID+"_"+CASE+"_"+___TEST_FILE___.split("/")[-1].split(".")[0]+"_"+TEST_CASE+A_STR, fontsize=7)
                            ax[e].set_xticks(x + width, tier_str)
                            ax[e].legend(loc='upper left', prop={'size': 6})
                            ax[e].set_ylim(0, 1.4)
                            ax[e].axvline(x=1.8375, color="k", linestyle="dotted", alpha=0.25)
                            ax[e].axhline(y=1., color="k", linestyle="dotted", alpha=0.25)
                        if SAVE_MY_PLOTS:
                            plt.savefig(ORIGIN+"../RESULTS/PLOTS/"+___MODELID___+"_"+CASE+"_"+___TEST_FILE___.split("/")[-1].split(".")[0]+"_"+TEST_CASE+A_STR+".png", dpi=400)
                    plt.show()
        print("\hline") #for latex tabular

print()    
print("The End")