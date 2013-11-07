import cPickle
import numpy as np

###################################
#####  General RBM data helpers
###################################

# helper function, puts data in a more convienient data structure (discarding all information that is not needed)
def buildStudentTrainingSet():
    print "Starting to create condensed training dataset for RBMs"
    print "This may take some time.."
    trainData = cPickle.load(open('../../datasets/valid_training.pkl', 'rb'))

    students = {}
    questions = {}
    totalQuestions = 0
    duplicateCount = 0
    #For each student create a tuple containing two lists:
    #One list that includes all question IDs the student was answered correctly/incorrectly
    #One list the contains the outcomes (correct(1)/incorrect(0)) of the questionIDs corresponding to the list index
    for index in range(0, len(trainData['user_id'])):
        # let question_id start at 0 instead of 1 to make array indexing easier 
        # only in training data set!
        #Get the current questionID
        qid = trainData['question_id'][index]
        #Get the current userID
        uid = trainData['user_id'][index]
        cor = trainData['correct'][index]
        out = trainData['outcome'][index]

        # only consider question with outcome 'correct' and 'incorrect'
        if out == 1 or out == 2: 
            if uid in students:
                ##If student in dict already -> append question and its outcome to the respective lists
                students[uid][0].append(qid)
                students[uid][1].append(cor)
            else:
                #If student not in dict yet -> create new dict entry
                students[uid] = ([qid], [cor])

                questions[qid] = 1

    # remove duplicate questions from all students 
    # by averaging over the outcomes of each question that was answered multiple times
    for studentId in students:
        qidList = students[studentId][0]
        corList = students[studentId][1]

        questionCounts = {}
        #Create a tuple of lists
        #One list containing for each question the sum of outcomes (correct(1)/incorrect(0)) 
        # if was answered multiple times by the current student
        #The second list contains the number of times this question was answered by the current student
        for i in range(0, len(qidList)):
            if qidList[i] in questionCounts:
                questionCounts[qidList[i]] = (questionCounts[qidList[i]][0] + corList[i], questionCounts[qidList[i]][1] + 1)
            else:
                questionCounts[qidList[i]] = (corList[i], 1)

        #Create two different lists from the list of tuples, the first list contains the first entries of each tuple,
        #the second list contains the second entries
        corTotal,corNorm = map(list, zip(*questionCounts.values()))
        #Entrywise divide the list containing the sum of outcomes 
        #for each question by the list containing the number of times the question was answered
        newCorList = np.true_divide(corTotal,corNorm).tolist()
        #Replaces the old tuple for each student by the new one without duplicate questions
        students[studentId] = (questionCounts.keys(),newCorList)

    #Compute the number of questions that were answered at least once by a student
    numQuestions = np.max(questions.keys()) + 1
    print "Number of questions with at least one answer:",numQuestions
    #Dump the result to disk
    cPickle.dump((students, numQuestions), open('../../datasets/rbm_valid_training_students.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print "Finished creating condensed training dataset for RBMs"   

# helper function, puts data in a more convienient data structure (discarding all information that is not needed)
def buildStudentTestSet():
    print "Starting to create condensed test dataset for RBMs"
    testData = cPickle.load(open('../../datasets/valid_test.pkl', 'rb'))

    testSet = []
    #Simply select only the three types of information that is required for prediction with
    #the RBMs and put it in a new pkl file (correct(0/1),questionID,userID)
    for index in range(0, len(testData['user_id'])):
        cor = testData['correct'][index]
        qid = testData['question_id'][index]
        uid = testData['user_id'][index]

        testSet.append((cor, uid, qid))

    cPickle.dump(testSet, open('../../datasets/rbm_valid_test_students.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print "Finished creating condensed test dataset for RBMs"

buildStudentTrainingSet()
buildStudentTestSet()
