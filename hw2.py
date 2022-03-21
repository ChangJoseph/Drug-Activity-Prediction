# Joseph Chang - G01189913

# Data Visualization

# File paths
training_file = "1645760197_9666755_1568904114_9130223_train_drugs_1.txt"
testing_file = "1645760197_9795408_1568904114_9234657_test.txt"
output_file = "hw2_output.txt"



#______________________________________________________________________________
# *** Naive Bayes Implementation ***

# counts of training records classified as either active or inactive
nb_active_count = 0
nb_inactive_count = 0

# set of active and inactive records' inputs
nb_active_set = {}
nb_inactive_set = {}

# Helper functions
def prod(xs, mod, norm):
    product = 1
    for x in xs:
        product *= ((x/mod)**norm)
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
    global nb_active_count
    global nb_inactive_count
    global nb_active_set
    global nb_inactive_set

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
                nb_inactive_count += 1
                if x in nb_inactive_set:
                    nb_inactive_set[x] += 1
                else:
                    nb_inactive_set[x] = 1
            elif active == 1:
                nb_active_count += 1
                if x in nb_active_set:
                    nb_active_set[x] += 1
                else:
                    nb_active_set[x] = 1
            else:
                print("error parsing active (not 0 or 1)")
        
        # END TRAINING FILE READ

    # Feature Selection/Reduction
    # Generic feature removal
    to_remove = []
    for key, val in nb_active_set.items():
        #break # TODO remove
        # Feature is in both active and inactive set
        if key in nb_inactive_set:
            inactive_val = nb_inactive_set[key]
            if val > inactive_val:
                ratio = inactive_val / val
            else:
                ratio = val / inactive_val
            # ratio threshold for removal
            thresh = 0.8
            if ratio > thresh:
                to_remove.append(key)
    for key in to_remove:
        if key in nb_inactive_set:
            nb_inactive_set.pop(key)
        if key in nb_active_set:
            nb_active_set.pop(key)
    
    # A drastic feature Selection of shared features
    to_remove = []
    for key, val in nb_active_set.items():
        if key not in nb_inactive_set:
            to_remove.append(key)
    for key in to_remove:
        nb_active_set.pop(key)
    to_remove = []
    for key, val in nb_inactive_set.items():
        if key not in nb_active_set:
            to_remove.append(key)
    for key in to_remove:
        nb_inactive_set.pop(key)

    #print(dict(list(nb_active_set.items())[:20]))
    #print(dict(list(nb_inactive_set.items())[:20]))

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
    prior0 = nb_inactive_count / (nb_active_count + nb_inactive_count)
    prior1 = nb_active_count / (nb_active_count + nb_inactive_count)
    #prior0 = prior1 = 1

    # analysis fields
    tie_count = 0
    set_null_count0 = set_null_count1 = 0
    threshold_unmet_count = 0
    no_product_count = 0
    class_count_0 = class_count_1 = 0
    lowest_ratio = 1

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
            if x in nb_inactive_set:
                if nb_inactive_set[x] / len(nb_inactive_set) > 0.00001:
                    likelihood0_set.append(nb_inactive_set[x])
                    if nb_inactive_set[x] / len(nb_inactive_set) < lowest_ratio:
                        lowest_ratio = nb_inactive_set[x] / len(nb_inactive_set)
                else:
                    threshold_unmet_count += 1
            else:
                #likelihood1_set.append(0)
                set_null_count0 += 1
            if x in nb_active_set:
                if nb_active_set[x] / len(nb_active_set) > 0.00001:
                    likelihood1_set.append(nb_active_set[x])
                    if nb_active_set[x] / len(nb_active_set) < lowest_ratio:
                        lowest_ratio = nb_active_set[x] / len(nb_active_set)
                else:
                    threshold_unmet_count += 1
            else:
                #likelihood1_set.append(0)
                set_null_count1 += 1


        # calculating naive bayes
        # prod(set, common denominators, normalization (exponent))
        # Here I set a normalization because the decimals get too small and python can assume zero for tiny numbers
        # ex: prod([x,...],d,e) -> (x/d)^(e) * ...
        likelihood0_raw = prod(likelihood0_set, nb_inactive_count,  0.02)
        likelihood1_raw = prod(likelihood1_set, nb_active_count,    0.02)
        #likelihood0_raw = prod(likelihood0_set, len(likelihood0_set), 0.05)
        #likelihood1_raw = prod(likelihood1_set, len(likelihood1_set), 0.05)
        #print(f"likelihoods: {likelihood0} {likelihood1}")

        # edge case if were no features (or in near impossible cases if all probabilities were 1)
        if likelihood0_raw == 1:
            no_product_count += 1
            likelihood0_raw = 0
        if likelihood1_raw == 1:
            no_product_count += 1
            likelihood1_raw = 0
        
        # normalization of the class probability calculations
        try:
            likelihood0 = likelihood0_raw / (likelihood0_raw + likelihood1_raw)
        except ZeroDivisionError:
            print("**zero division")
            likelihood0 = 0
        try:
            likelihood1 = likelihood1_raw / (likelihood0_raw + likelihood1_raw)
        except ZeroDivisionError:
            print("**zero division")
            likelihood1 = 0
        # evidence can be ignored (arbitrary) because it is the same for both calculations (for C=0 and C=1)
        #evidence = 1 # p(x)
        # final naive bayes calculation
        c0_calc = prior0 * likelihood0 #/ evidence
        c1_calc = prior1 * likelihood1 #/ evidence

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
    #print("len(nb_active_set) " + str(len(nb_active_set)) + "\nlen(nb_inactive_set) " + str(len(nb_inactive_set)))
    print(f"no_product_count (calc:1->0) = {no_product_count}\tlowest ratio = {lowest_ratio}")
    print(f"class_count_0 = {class_count_0}\tclass_count_1 = {class_count_1}\tnumber of ties = {tie_count}")
    print(f"not in dict 0 = {set_null_count0}\tnot in dict 1 = {set_null_count1}\tunmet prob thresh = {threshold_unmet_count}")
    print("--End Naive Bayes Analyses--")

    # Write output to designated file
    out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()


#______________________________________________________________________________

# *** Neural Network Implementation ***
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import keras

def f1_score(true, pred):
    precision = 1
    recall = 1
    return 2 * (precision * recall) / (precision + recall)

"""
This neural_network() is for the neural network approach
"""
def neural_network():
    # *** TRAINING PHASE ***

    in_shape = (800,100001)
    x_train = np.zeros(in_shape)
    y_train = np.zeros(800)
    # open training file
    train_file = open(training_file, "r", encoding="utf-8")
    read_iteration = 0
    # read each line of the training file
    while (1):
        line = train_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        # get the input data of the current line's record
        input = line[2:].split(" ")[:-1] # the last index is an empty string
        # get the active status of current line's record
        y_train[read_iteration] = int(line[0]) # 0 for inactive : 1 for active
        for read_elem in input:
            x_train[read_iteration][int(read_elem)] = 1
        read_iteration += 1

    # cross validation - 3/4 train 1/4 test
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.3, random_state=0)

    # Initial values for model
    init_func = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed = 0)
    # Layer structure for model
    model = keras.Sequential()
    # entry layer with 100000 input neurons with a sigmoid activation function
    #input_layer = keras.layers.Dense(2, input_shape=(100001,), kernel_initializer=init_func)
    input_layer = keras.layers.Dense(256,input_shape=(100001,))
    model.add(input_layer)
    hidden_layer1 = keras.layers.Dropout(0.5)
    model.add(hidden_layer1)
    hidden_layer2 = keras.layers.Dense(200, activation="relu")
    model.add(hidden_layer2)
    #hidden_layer3 = keras.layers.Dense(256, activation="relu")
    #model.add(hidden_layer3)
    output_layer = keras.layers.Dense(1, activation="hard_sigmoid")
    model.add(output_layer)
    # configure the model
    # optimizer is the Stochastic gradient descent optimizer (finding min iteratively)
    model.compile(loss = "binary_crossentropy", optimizer = "sgd", metrics = ["binary_accuracy"])
    # train the model
    model.fit(X_train, y_train, epochs = 24, batch_size = 32)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print (f"\naccuracy:\t{test_acc}\nloss:\t{test_loss}")


    # *** TESTING PHASE ***

    # open testing and output files
    out_file = open(output_file, "w")
    output_string = ""
    
    in_shape = (350,100001)
    testing_data = np.zeros(in_shape)
    # open training file
    test_file = open(testing_file, "r", encoding="utf-8")
    read_iteration = 0
    # read each line of the training file
    while (1):
        line = test_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        # get the input data of the current line's record
        input = line.split(" ")[:-1] # the last index is an empty string
        for read_elem in input:
            testing_data[read_iteration][int(read_elem)] = 1
        read_iteration += 1

    # run neural network model against hw2 given testing data
    test_output = model.predict(testing_data)
    pretty_output = ""
    count_1s = 0

    for test_record in test_output:
        pretty_output += str(test_record[0]) + "\t"
        output = ""
        probability = test_record[0] + test_loss
        if probability > 0.5:
            output = "1\n"
            count_1s += 1
        else:
            output = "0\n"
        output_string += output

    #print(pretty_output)
    print("ones: " + str(count_1s))

    # Write output to designated file
    out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()

    return


# Train and Test Function Execution Code Segment
print("Starting code execution")
#train_nb()
#test_nb()
neural_network()
