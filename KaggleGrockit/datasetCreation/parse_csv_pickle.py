# reads csv files into dictionary and writes them back with pickle
import csv, gzip, cPickle, collections, array

def pickleTrainData():
    """
        Creates a compressed version of the WhatDoYouKnow valid_training dataset that 
        only contains data which is actually used by the models
    """
    csvFilename = '../../datasets/grockit_all_data/valid_training.csv'
    pklFilename = '../../datasets/valid_training.pkl'

    trainData = collections.defaultdict(list)
    print 'reading ' + csvFilename + '...'
    
    progress = 0
    
    # read data from .csv strings into more compact data types (integers/lists/dates)
    for d in csv.DictReader(open(csvFilename, 'rb'), delimiter=','):
        # integer values
        for key in ['outcome', 'correct', 'user_id', 'question_id',
                    'question_type', 'group_name', 'track_name', 
                    'subtrack_name']:
            trainData[key].append(int(d[key]))
        
        # list of integers data (one question can have several tagstrings)
        trainData['tag_string'].append(d['tag_string'])
                    
        #Show progress
        progress += 1
        if progress % 100000 == 0:
            print 'read', progress, 'rows...'
              
    print 'dumping result to ' + pklFilename + '...' 

    # dump result to compressed file
    cPickle.dump(trainData, open(pklFilename, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def pickleTestData():
    """
        Creates a compressed version of the WhatDoYouKnow valid_test dataset that 
        only contains data which is actually used by the models
    """
    csvFilename = '../../datasets/grockit_all_data/valid_test.csv'
    pklFilename = '../../datasets/valid_test.pkl'

    testData = collections.defaultdict(list)
    print 'reading ' + csvFilename + '...'

    # read data from .csv strings into more compact data types (integers/lists/dates)
    for d in csv.DictReader(open(csvFilename, 'rb'), delimiter=','):
        # integer values
        for key in ['correct', 'user_id', 'question_id', 
                    'question_type', 'group_name', 'track_name', 
                    'subtrack_name']:
            testData[key].append(int(d[key]))
                                
        # list of integers data
        testData['tag_string'].append(d['tag_string'])
                    
    print 'dumping result to ' + pklFilename + '...' 

    # dump result to file
    cPickle.dump(testData, open(pklFilename, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def main():
    #Create the condensed version of the training data
    pickleTrainData()
    #Create the condensed version of the test data
    pickleTestData()

if __name__ == "__main__":
    main()

