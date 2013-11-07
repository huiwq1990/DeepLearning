import cPickle
import numpy as np
from optparse import OptionParser
from rbm_softmax import SoftmaxRBM
from rbm_factored import FactoredRBM
from rbm_conditional import ConditionalRBM
from rbm_softmax_binomial import SoftmaxBinomialRBM
from rbm_factored_binomial import FactoredBinomialRBM

def cappedBinomialDeviance(outcome, prediction):
    prediction = np.clip(prediction, 0.01, 0.99)
    return -(outcome * np.log10(prediction) + (1 - outcome) * np.log10(1 - prediction))

def main():
    # throw an exception on overflow encountered in exp
    np.seterr(over='raise')

    parser = OptionParser()
    parser.add_option("-n", "--numHid", dest="numHid", default=15, help="number of hidden units")
    parser.add_option("-e", "--epochs", dest="epochs", default=60, help="number of epochs")
    parser.add_option("-r", "--rateW", dest="epsilonW", default=0.001, help="learning rate - weights")
    parser.add_option("-v", "--rateV", dest="epsilonV", default=0.001, help="learning rate - visible biases")
    parser.add_option("-z", "--rateH", dest="epsilonH", default=0.001, help="learning rate - hidden biases")
    parser.add_option("-c", "--weightCost", dest="weightCost", default=0.1, help="weight decay")
    parser.add_option("-m", "--momentum", dest="momentum", default=0.9, help="momentum")
    parser.add_option("-b", "--bSize", dest="batchSize", default=1000, help="batch size")
    parser.add_option("-w", "--weightInit", dest="weightInit", default=0.001, help="initialization for the weights (standard deviation of gaussian)")
    parser.add_option("-a", "--hiddenBiasInit", dest="hbInit", default=0.00, help="hidden bias initialization")
    parser.add_option("-s", "--steps", dest="steps", default=1, help="number of gibbs sample steps in the training phase")
    parser.add_option("-o", "--output", dest="output", default="cbd.txt", help="Output the error at each epoch to the specified file")

    parser.add_option("-t", "--type", dest="rbmType", default="0", help="Type of the RBM, 0=Standard, 1=Factored, 2=Conditional")
    parser.add_option("-x", "--binomial", action="store_true", dest="binomial", default=False, help="Use binomial visible units.")
    parser.add_option("-y", "--numC", dest="numC", default=15, help="Rank of Matrix-Factorization for Factored RBM.")

    (options, arguments) = parser.parse_args()

    # students is a dictionary which maps a student id to a list of question_ids he answered and a list with the outcomes
    # student_id ---> ([question_id1, questions_id2, ...], [correct1, correct2, ...])
    students, numQuestions = cPickle.load(open('../../datasets/rbm_valid_training_students.pkl', 'rb'))
    print 'read', len(students), 'different students and', numQuestions, 'questions...'

    # that student with id student_id answers the question with id question_id correctly. (correct is set to -1 in real test-set)
    validTestSet = cPickle.load(open('../../datasets/rbm_valid_test_students.pkl', 'rb'))
    print 'read', len(validTestSet), 'test cases...'

    ### Set all the initial values for our parameters
    numVis = numQuestions
    numHid = int(options.numHid)
    weightInit = float(options.weightInit)
    hbInit = float(options.hbInit)
    epsilonW = float(options.epsilonW)  # Learning rate for weights
    epsilonVB = float(options.epsilonV)   # Learning rate for biases of visible units
    epsilonHB = float(options.epsilonH)  # Learning rate for biases of hidden units
    weightCost = float(options.weightCost)  # Weight decay
    momentum = float(options.momentum)
    maxEpochs = int(options.epochs)
    batchSize = int(options.batchSize)
    steps = int(options.steps)
    cbdOutput = open(options.output, 'w')
    binomial = bool(options.binomial) #use binomial visibles
    rbmType = int(options.rbmType) # 0=standard, 1=factored, 2=conditional
    numC = int(options.numC) # rank of matrix factorization

    parameterInfo = "Training RBM with the following set of parameters: \n"
    parameterInfo += "  #Hiddens:" + str(numHid) + ", #Visibles:" + str(numVis) + "\n"
    parameterInfo += "  Hidden Bias Init:" + str(hbInit) + ", Weight Init:" + str(weightInit) + "\n"
    parameterInfo += "  Learning factors:\n"
    parameterInfo += "     Weights:" + str(epsilonW) + "\n"
    parameterInfo += "     Visible Bias:" + str(epsilonVB) + "\n"
    parameterInfo += "     Hidden Bias:" + str(epsilonHB) + "\n"
    parameterInfo += "  Weight Cost:" + str(weightCost) + ", Momentum:" + str(momentum) + "\n"
    parameterInfo += "  #Epochs:" + str(maxEpochs) + ", Batch Size:" + str(batchSize) + "\n"
    parameterInfo += "  #Gibbs-Steps: CD-" + str(steps) + "\n"

    ##########
    # duplicate (question,correct) pairs for each student
    # This creates binomial visible units.
    ##########
    if binomial == True:
        parameterInfo += "  Visible Unit Type: Binomial" + "\n"
        for sID in students:
            qidList = students[sID][0]
            corList = students[sID][1]
            
            qidNew = []
            corNew = []
            for i in range(0, 2):
                qidNew += qidList
                corNew += corList

            students[sID] = (qidNew, corNew)
    else:
        parameterInfo += "  Visible Unit Type: Bernoulli" + "\n"

    rbm = None
    if rbmType == 1:
        parameterInfo += "  RBM Type: Factored (numC=" + str(numC) + ")\n"
        if binomial == True:
            rbm = FactoredBinomialRBM(students, numVis, numHid, numC, weightInit, hbInit)
        else:
            rbm = FactoredRBM(students, numVis, numHid, numC, weightInit, hbInit)
    elif rbmType == 2:
        parameterInfo += "  RBM Type: Conditional\n"
        conditionalData = cPickle.load(open('../../datasets/conditional_valid_training_students.pkl', 'rb'))
        print "read conditional data..."    
        rbm = ConditionalRBM(students, conditionalData, numVis, numHid, weightInit, hbInit)
    else:
        parameterInfo += "  RBM Type: Standard\n"
        if binomial == True:
            rbm = SoftmaxBinomialRBM(students, numVis, numHid, weightInit, hbInit)
        else:
            rbm = SoftmaxRBM(students, numVis, numHid, weightInit, hbInit)
        
    print parameterInfo
    cbdOutput.write(parameterInfo)
    cbdOutput.flush()    

    # Output CBD for untrained RBM
    correct = np.array(map(list, zip(*validTestSet))[0])
    cbd = np.mean(cappedBinomialDeviance(correct,rbm.calculatePrediction(validTestSet)))
    cbdOutput.write("0," + str(cbd) + "\n")
    cbdOutput.flush()
    print "CBD: Epoch: 0 ,",cbd

    # Train RBM
    for epoch in xrange(maxEpochs):
        mse = rbm.trainEpoch(epsilonW,epsilonVB,epsilonHB, weightCost,momentum,batchSize,steps)
        cbd = np.mean(cappedBinomialDeviance(correct,rbm.calculatePrediction(validTestSet)))
        print "CBD: Epoch:", (epoch + 1),",",cbd, ", reconstruction error:",mse
        cbdOutput.write(str(epoch + 1) + "," + str(cbd)+"\n")
        cbdOutput.flush()
        
    cbdOutput.close()
    
if __name__ == '__main__':
    main()
