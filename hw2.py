# Joseph Chang - G01189913

# Data Visualization
from re import A
from threading import active_count
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import time

# File paths
training_file = "1645760197_9666755_1568904114_9130223_train_drugs_1.txt"
testing_file = "1645760197_9795408_1568904114_9234657_test.txt"
output_file = "hw2_output.txt"


# *** Naive Bayes Implementation ***

# counts of training records classified as either active or inactive
active_count = 0
inactive_count = 0

# set of active and inactive records' inputs
active_set = {}
inactive_set = {}


# Helper functions
def prod(xs, mod):
    product = 1
    for x in xs:
        product *= ((x/mod) ** 0.06)
    return product


"""
Naive Bayes Training Segment
This train_nb() function is the training sequence for the naive bayes approach
The higher probability of classification (or p(C)) given the probability of the given record inputs (or p(C|x)) is equivalent to p(C)*p(x|C)/p(x) where C is the classification (as 0 or 1) and x are the record inputs (indices)
Implementation Notes:
- feature reduction on features that seem to be too redundant or too generic between all records
"""
def train_nb():
    # global fields
    global active_count
    global inactive_count
    global active_set
    global inactive_set

    # open training file
    train_file = open(training_file, "r", encoding="utf-8")
    
    # read each line of the training file
    while (1):
        line = train_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        
        # get the input data of the current line's record
        input = line[2:].split(" ")[:-1] # the last index is an empty string
        # get the active status of current line's record
        active = int(line[0]) # 0 for inactive : 1 for active
        for x in input:
            if active == 0:
                inactive_count += 1
                if x in inactive_set:
                    inactive_set[x] += 1
                else:
                    inactive_set[x] = 1
            elif active == 1:
                active_count += 1
                if x in active_set:
                    active_set[x] += 1
                else:
                    active_set[x] = 1
            else:
                print("error parsing active (not 0 or 1)")
        
        # END TRAINING FILE READ

    # Feature Reduction/Selection
    # Generic feature removal
    to_remove = []
    for key, val in active_set.items():
        #break # TODO remove
        # Feature is in both active and inactive set
        if key in inactive_set:
            inactive_val = inactive_set[key]
            if val > inactive_val:
                ratio = inactive_val / val
            else:
                ratio = val / inactive_val
            # ratio threshold for removal
            thresh = 0.8
            if ratio > thresh:
                to_remove.append(key)
    for key in to_remove:
        if key in inactive_set:
            inactive_set.pop(key)
        if key in active_set:
            active_set.pop(key)

    #print(dict(list(active_set.items())[:20]))
    #print(dict(list(inactive_set.items())[:20]))

    # close train files
    train_file.close()
    
"""
Naive Bayes Testing Segment
Naive Bayes Assumption: each feature is independent and equal (or at least close to it)
Implementation Notes:
- assumes test class distribution follows the training class distribution (important for calculation the "prior" values)
"""
def test_nb():
    # open testing file
    test_file = open(testing_file, "r", encoding="utf-8")
    out_file = open(output_file, "w", encoding="utf-8")
    output_string = ""
    
    # prior value calculations
    # prior value calculated assuming testing set has same distribution as training set
    prior0 = inactive_count / (active_count + inactive_count)
    prior1 = active_count / (active_count + inactive_count)
    #prior0 = prior1 = 1

    # analysis fields
    tie_count = 0
    set_null_count0 = 0
    set_null_count1 = 0
    threshold_unmet_count = 0
    no_product_count = 0
    class_count_0 = 0
    class_count_1 = 0
    lowest_ratio = 10000

    # read each line of the testing file
    #while (tie_count == 0):
    while (1):
        line = test_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        
        # get the input data of the current line's record
        input = line.split(" ")[:-1] # the last index is an empty string
        
        # prior0 and prior1 should be defined after training
        # Calculating likelihood (or conditional probability) of the current given record
        likelihood0_set = [] # p(x|C_0)
        likelihood1_set = [] # p(x|C_1)
        for x in input:
            if x in inactive_set:
                if inactive_set[x] / len(inactive_set) > 0.00001:
                    likelihood0_set.append(inactive_set[x])
                    if inactive_set[x] / len(inactive_set) < lowest_ratio:
                        lowest_ratio = inactive_set[x] / len(inactive_set)
                else:
                    threshold_unmet_count += 1
            else:
                set_null_count0 += 1
            if x in active_set:
                if active_set[x] / len(active_set) > 0.00001:
                    likelihood1_set.append(active_set[x])
                    if active_set[x] / len(active_set) < lowest_ratio:
                        lowest_ratio = active_set[x] / len(active_set)
                else:
                    threshold_unmet_count += 1
            else:
                set_null_count1 += 1


        # calculating naive bayes
        likelihood0_raw = prod(likelihood0_set,len(inactive_set))
        likelihood1_raw = prod(likelihood1_set,len(active_set))
        #print(f"likelihoods: {likelihood0} {likelihood1}")
        
        # normalization of the class probability calculations
        likelihood0 = likelihood0_raw / (likelihood0_raw + likelihood1_raw)
        likelihood1 = likelihood1_raw / (likelihood0_raw + likelihood1_raw)
        # evidence can be ignored (arbitrary) because it is the same for both calculations (for C=0 and C=1)
        #evidence = 1 # p(x)
        # final naive bayes calculation
        c0_calc = prior0 * likelihood0 #/ evidence
        c1_calc = prior1 * likelihood1 #/ evidence
        
        # edge case if were no features (or in near impossible cases if all probabilities were 1)
        if c0_calc == 1:
            no_product_count += 1
            c0_calc = 0
        if c1_calc == 1:
            no_product_count += 1
            c1_calc = 0

        # obtaining predicted class
        output = "" # reset current record output just in case
        if c0_calc > c1_calc:
            class_count_0 += 1
            output = "0\n"
        elif c0_calc < c1_calc:
            class_count_1 += 1
            output = "1\n"
        else:
            #print(f"c0 {c0_calc}; c1 {c1_calc}")
            # Default to inactive on ties (very unlikely occurence)
            tie_count += 1
            output = "0\n"
        
        output_string += output
        
        # END TESTING FILE READ
    
    # Analysis prints
    print("--Naive Bayes Analyses--")
    print(f"prior0 = {prior0}\tprior1 = {prior1}")
    #print("len(active_set) " + str(len(active_set)) + "\nlen(inactive_set) " + str(len(inactive_set)))
    print(f"no_product_count (calc:1->0) = {no_product_count}\tlowest ratio = {lowest_ratio}")
    print(f"class_count_0 = {class_count_0}\tclass_count_1 = {class_count_1}\tnumber of ties = {tie_count}")
    print(f"not in dict 0 = {set_null_count0}\tnot in dict 1 = {set_null_count1}\tunmet prob thresh = {threshold_unmet_count}")
    print("--End Naive Bayes Analyses--")

    # Write output to designated file
    out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()





# *** Neural Network Implementation ***

"""
This train_nn() is the training sequence for the neural network approach
In this perceptron model, 
"""
def train_nn():
    # open training file
    train_file = open(training_file, "r", encoding="utf-8")
    
    
    
    # close train files
    train_file.close()

def test_nn():
    # open testing file
    test_file = open(testing_file, "r", encoding="utf-8")
    out_file = open(output_file, "w")
    output_string = ""
    
    
    
    # Write output to designated file
    #out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()


# Train and Test Function Execution Code Segment
print("Starting code execution")
train_nb()
test_nb()

"""
train_nn()
test_nn()
"""