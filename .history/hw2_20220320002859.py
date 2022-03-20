# Joseph Chang - G01189913

# Data Visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# File paths
training_file = "1645760197_9666755_1568904114_9130223_train_drugs_1.txt"
testing_file = "1645760197_9795408_1568904114_9234657_test.txt"
output_file = "hw2_output.txt"


# *** Naive Bayes Implementation ***

# naive bayes fields
prior0 = -1 # p(C_0)
prior1 = -1 # p(C_1)

"""
This train_nb() function is the training sequence for the naive bayes approach
The higher probability of classification (or p(C)) given the probability of the given record inputs (or p(C|x)) is equivalent to p(C)*p(x|C)/p(x) where C is the classification (as 0 or 1) and x are the record inputs (indices)
"""
def train_nb():
    # open training file
    train_file = open(training_file, "r", encoding="utf-8")
    
    # counts of training records classified as either active or inactive
    active_count = 0
    inactive_count = 0
    
    # set of active and inactive records' inputs
    active_set = []
    inactive_set = []
    
    # read each line of the training file
    while (1):
        line = train_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        
        # get the active status of current line's record
        active = line[0] # 0 for inactive : 1 for active
        # get the input data of the current line's record
        input = line[2:].split(" ")[:-1] # the last index is an empty string
        
        if active == 0:
            inactive_count += 1
            inactive_set.append(input)
        else:
            active_count += 1
            active_set.append(input)
        
        # END TRAINING FILE READ
    
    prior0 = inactive_count
    prior1 = active_count

    # close train files
    train_file.close()
    

def test_nb():
    # open testing file
    test_file = open(testing_file, "r", encoding="utf-8")
    out_file = open(output_file, "w")
    output_string = ""
    
    # Here is where the prior values are calculated
    # I chose to assume that the test class distribution follow the training class distribution
    prior0 = inactive_count / (active_count + inactive_count)
    prior1 = active_count / (active_count + inactive_count)
    
    # read each line of the testing file
    while (1):
        line = test_file.readline()
        if (line == ""): # ""/eof -> done reading lines
            break
        
        # get the input data of the current line's record
        input = line[2:].split(" ")[:-1] # the last index is an empty string
        
        # Calculating probabilities of the given record
        # prior0 and prior1 should be defined after training
        likelihood0 = -1 # TODO p(x|C_0)
        likelihood1 = -1 # TODO p(x|C_1)
        evidence = 1 # p(x)
        
        # calculating naive bayes
        c0_calc = prior0 * likelihood0 #/ evidence
        c1_calc = prior1 * likelihood1 #/ evidence
        # obtaining predicted class
        if c0_calc > c1_calc:
            output = "0\n"
        elif c0_calc < c1_calc:
            output = "1\n"
        else:
            # Default to inactive on ties (very unlikely occurence)
            output = "0\n"
        
        output_string += output
        
        # END TESTING FILE READ
    
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
    out_file.write(output_string)
    # close test and output files
    test_file.close()
    out_file.close()


# Train and Test Function Execution Code Segment
print("entering train_nb()")
train_nb()
print("finished train_nb(); entering test_nb()")
test_nb()
print("finished test_nb()")

print("entering train_nn()")
train_nn()
print("finished train_nn(); entering test_nn()")
test_nn()
print("finished test_nn()")