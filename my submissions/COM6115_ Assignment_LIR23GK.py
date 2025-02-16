#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=1

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = [line.strip() for line in posDictionary if line.strip() and not line.startswith(';')] # Exclude comments and empty lines

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = [line.strip() for line in negDictionary if line.strip() and not line.startswith(';')] # Exclude comments and empty lines

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalses to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    # create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"
#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):

    # print("Naive Bayes classification")
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("testBayes ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("testBayes ERROR (neg classed as pos %0.2f):" %prob + sentence)

    # TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
    # (3) precision and recall for the negative class; (4) F1 score;
    calculate_metrics(correct, total, correctpos, totalpos, totalpospred, correctneg, totalneg, totalnegpred, dataName)
 




# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

    print("Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("testDictionary ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("testDictionary ERROR (neg classed as pos %0.2f):" %score + sentence)    
    # TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
    # (3) precision and recall for the negative class; (4) F1 score;
    calculate_metrics(correct, total, correctpos, totalpos, totalpospred, correctneg, totalneg, totalnegpred, dataName)
 



# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n, sentimentDictionary):
    predictPower = {}
    for word in pWord:
        if pWordNeg[word] < 0.0000001:  # Handle very small probabilities
            predictPower[word] = 1000000000
        else:
            predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])

    # Sort by predictive power
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail =  sortedPower[:n], sortedPower[len(predictPower)-n:]

    # Calculate overlap with sentiment dictionary
    overlap_neg = [word for word in head if word in sentimentDictionary]
    overlap_pos = [word for word in tail if word in sentimentDictionary]
    totalCount = len(overlap_neg) + len(overlap_pos)

    # Print results
    print("Number of most useful negative words in sentiment dictionary:", len(overlap_neg))
    print("Number of most useful positive words in sentiment dictionary:", len(overlap_pos))
    print("Total count of most useful words in sentiment dictionary:", totalCount)
    print("\nNEGATIVE:")
    print(head)
    print("\nPOSITIVE:")
    print(tail)


# This is a common function which would calcuate the metrics that are needed to do analysis of the model performance
# on different datasets and returning the required metrics information
# printing as well to verify the results on console
def calculate_metrics(correct, total, correctpos, totalpos, totalpospred, correctneg, totalneg, totalnegpred, dataName):
    accuracy = correct/total
   
    precisionPositive = correctpos/totalpospred
    recallPositive = correctpos/totalpos
    f1ScorePositive = 2* (precisionPositive * recallPositive)/(precisionPositive + recallPositive)

    precisionNegative = correctneg/totalnegpred
    recallNegative = correctneg/totalneg
    f1ScoreNegative = 2*(precisionNegative * recallNegative)/(precisionNegative + recallNegative)

    totalF1Score = ((f1ScorePositive * totalpos)/total) + ((f1ScoreNegative * totalneg)/total)
    
    metrics = {
        "Accuracy": f"{accuracy:.4f}",
        "Precision Positive" : f"{precisionPositive:.4f}",
        "Recall Positive": f"{recallPositive:.4f}",
        "F-measure Positive": f"{f1ScorePositive:.4f}",
        "Precision Negativi": f"{precisionNegative:.4f}",
        "Recall Negative": f"{recallNegative:.4f}",
        "F-measure Negative": f"{f1ScoreNegative:.4f}",
        "TotalF1Score": f"{totalF1Score:.4f}"
    }
    print(f"Metrics of {dataName}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics

# This is the enhanced test dictionary function which would consider negation_words
# and diminisher_words to verify any improvment of metrics when compared to the orginal functional
def enhancedTestDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    print("Enhanced Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    negation_words = {"not", "no", "never", "cannot", "nor", "nothing", "nowhere", "without", "barely", "hardly"}
    diminisher_words = {"slightly", "barely", "hardly", "somewhat", "marginally", "moderately", "scarcely", "faintly", "almost", "just"}
    
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        negation_flag = False
        diminisher_flag = False

        for word in Words:
            if word in negation_words:
               negation_flag = True
               continue

            if word in diminisher_words:
                diminisher_flag = True
                continue

            if word in sentimentDictionary:
                word_score = sentimentDictionary[word]

                if negation_flag:
                    word_score = -word_score
                    negation_flag = False
                
                if diminisher_flag:
                    word_score = 0.5
                    diminisher_flag = False
                
                score += word_score
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("enhancedTestDictionary ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("enhancedTestDictionary ERROR (neg classed as pos %0.2f):" %score + sentence)    
    # function to calculate metrics and return required information               
    calculate_metrics(correct, total, correctpos, totalpos, totalpospred, correctneg, totalneg, totalnegpred, dataName)
 


#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)


# run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

# run enhanced sentiment dictionary based classifier on datasets
enhancedTestDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
enhancedTestDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
enhancedTestDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

# print most useful words and count the overlap with sentimentDictionary
# Added new param to calculate the overlap words requirement, step 4 task 2
mostUseful(pWordPos, pWordNeg, pWord, 100, sentimentDictionary)
