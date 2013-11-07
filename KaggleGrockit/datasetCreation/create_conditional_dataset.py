import numpy as np
import cPickle

###################################
#####  CONDITIONAL RBM data helpers
###################################


print "Starting to create the data for the conditional RBM"

conditionalDataset = {}

testData = cPickle.load(open('../../datasets/valid_test.pkl', 'rb'))
trainData = cPickle.load(open('../../datasets/valid_training.pkl', 'rb'))

#For the conditional RBM: combine the trainingdata with the testdata
#We assume that from the fact that a student has answered a question (=> its in the test or training set)
# we can infer information
#Since naturally we don't know the outcome of a question from the testset
#we only store whether a question was ansered or not regardless of the outcome


#Create a vector that contains for each user a list of questions he has answered
#First loop handles the questions answered in the training data 
#-> second loop the questions answered in the testdata
dataLength = len(trainData['user_id'])
print "Processing trainingdata"
for index in range(0, dataLength):
    
    if index % 100000 == 0:
        print "Progress:",index,"of",dataLength
    
    uid = trainData['user_id'][index]
    qid = trainData['question_id'][index]
    cor = trainData['correct'][index]
    out = trainData['outcome'][index]

    if not (uid in conditionalDataset):
        conditionalDataset[uid] = []

    if not (qid in conditionalDataset[uid]):
        conditionalDataset[uid].append(qid)

        
print "Processing testdata"
dataLength = len(testData['user_id'])
for index in range(0, dataLength):

    if index % 20000 == 0:
        print "Progress:",index,"of",dataLength

    uid = testData['user_id'][index]
    qid = testData['question_id'][index]

    if not (uid in conditionalDataset):
        conditionalDataset[uid] = []

    if not (qid in conditionalDataset[uid]):
        conditionalDataset[uid].append(qid)


cPickle.dump(conditionalDataset, open('../../datasets/conditional_valid_training_students.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
print "Finished creating the data for the conditional RBM"
