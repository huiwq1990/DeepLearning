import numpy as np
import random

class FactoredRBM:
    
    def logistic(self,x):
        # HACK: clip x to prevent overflows, shouldn't be too inaccurate however considering
        # the nature of the logistic function.
        # note: overflow doesnt produce a wrong result here since 1.0 / (1.0 + np.inf) = 0
        x = np.clip(x, -500, 500)

        return 1.0 / (1.0 + np.exp(-x))    
    
    # numC is the size of the matrix factorization
    def __init__(self, students, numVis, numHid, numC,
                weightInit, hbInit):

        ### Set all the initial values for our parameters
        self.students = students
        self.numVis = numVis
        self.numHid = numHid
        self.numC = numC

        # Initialize weights, factorize weight matrix in following fashion:
        self.w_A_0 = weightInit * np.random.randn(numVis, numC) 
        self.w_A_1 = weightInit * np.random.randn(numVis, numC)
        self.w_B = -1.0 * np.eye(numC, numHid) # weightInit * np.random.randn(numC, numHid)

        # Initialize biases
        self.w_v_0 = np.zeros(numVis)
        self.w_v_1 = np.zeros(numVis)
        self.w_h = hbInit * np.ones(numHid)

        # Weight updates
        self.wu_A_0 = np.zeros((numVis, numC)) 
        self.wu_A_1 = np.zeros((numVis, numC))
        self.wu_B = np.zeros((numC, numHid))
        self.wu_v_0 = np.zeros(numVis)
        self.wu_v_1 = np.zeros(numVis)
        self.wu_h = np.zeros(numHid)
        
        #Normalization factor for biases
        norm = np.zeros((2,numVis))
        # compute amount of times each question appears in the training set
        for studentID in students:
            for i in xrange(len(students[studentID][1])):
                cor = students[studentID][1][i]
                qid = students[studentID][0][i]
                norm[1-cor][qid] += 1

        #Initialize visible biases to the logs of their respective base rates over all students
        #see netflix paper, point 2.1 text after equation (2)
        for i in xrange(numVis):
            normTot = norm[0][i] + norm[1][i]
            if normTot > 0:
                if norm[0][i] > 0:
                    self.w_v_0[i] = np.log(norm[0][i] / normTot)
                if norm[1][i] > 0:
                    self.w_v_1[i] = np.log(norm[1][i] / normTot)

        
    def trainEpoch(self, epsilonW,epsilonVB,epsilonHB,
                weightCost, momentum, 
                batchSize, tSteps):

        wu_sum_A_0 = np.zeros((self.numVis, self.numC)) 
        wu_sum_A_1 = np.zeros((self.numVis, self.numC))
        wu_sum_B = np.zeros((self.numC, self.numHid))
        wu_sum_v_0 = np.zeros(self.numVis)
        wu_sum_v_1 = np.zeros(self.numVis)
        wu_sum_h = np.zeros(self.numHid)
        norm = np.zeros(self.numVis)
        
        # randomize order of student_data to make batch-learning better
        # see point 4.1 in hinton RBM practical guide
        shuffledSIDs = self.students.keys()
        random.shuffle(shuffledSIDs)

        numStudents = len(self.students)
        batchCounter = 0

        recMSE = []
        for studentID in shuffledSIDs:
            # sdata[0] are the question_ids (starting at 0) answered by the student
            # sdata[1] are the outcomes (correct/wrong)
            sdata = [np.array(self.students[studentID][0]), np.array(self.students[studentID][1])]
            # Number of questions the current Student has answered
            # => [Num]ber of [Vis]ible units of his [Sub]RBM
            numVisSub = len(sdata[0])

            # Count how often each question was answered in this minibatch
            # Add this student's contributions
            norm[sdata[0]] += 1

            # calculate weight-matrices for the sub RBMs
            s_w_v_0 = self.w_v_0[sdata[0]]
            s_w_v_1 = self.w_v_1[sdata[0]]
            s_w_A_0 = self.w_A_0[sdata[0]]
            s_w_A_1 = self.w_A_1[sdata[0]]

            # compute gradient using CD_n
            # positive phase
            vSample_pos_0 = sdata[1] # initial visible states = question state (correct//wrong)
            vSample_pos_1 = 1 - sdata[1]
            
            # calculate hidden probabilities from binary visible units
            hProb_pos = self.logistic(np.dot(np.dot(vSample_pos_0, s_w_A_0), self.w_B) + np.dot(np.dot(vSample_pos_1, s_w_A_1), self.w_B) + self.w_h) 

            # initialize gibbs-chain with hProb_pos
            hProb_neg = hProb_pos

            #Remember the sampled visible units
            vSample_neg_0 = None
            vSample_neg_1 = None
            
            # negative phase, n gibbs steps
            for i in range(tSteps):
                hSample_neg = hProb_neg > np.random.rand(self.numHid) # sample hidden units

                # compute visibile probabilities from binary sampled hidden units
                a_v_0 = np.exp(s_w_v_0 + np.dot(s_w_A_0, np.dot(self.w_B, hSample_neg)))
                a_v_1 = np.exp(s_w_v_1 + np.dot(s_w_A_1, np.dot(self.w_B, hSample_neg)))
                vProb_neg_0 = a_v_0 / (a_v_0 + a_v_1)
                vProb_neg_1 = a_v_1 / (a_v_0 + a_v_1)

                # use probabilities instead of samples for last step
                vSample_neg_0 = vProb_neg_0 > np.random.rand(numVisSub)
                vSample_neg_1 = vProb_neg_1 > np.random.rand(numVisSub)

                # calculate hidden probabilities from binary visible units
                hProb_neg = self.logistic(np.dot(np.dot(vSample_neg_0, s_w_A_0), self.w_B) + np.dot(np.dot(vSample_neg_1, s_w_A_1), self.w_B) + self.w_h) 

            #Sample the hidden units
            hSample_neg = hProb_neg > np.random.rand(self.numHid)
                
            # update reconstruction error
            recMSE.append(np.sum((vSample_pos_0 - vProb_neg_0)**2)/numVisSub)
            recMSE.append(np.sum((vSample_pos_1 - vProb_neg_1)**2)/numVisSub)

            # update wu_sum
            wu_sum_A_0[sdata[0]] += np.outer(vSample_pos_0, np.dot(self.w_B, hProb_pos)) - np.outer(vSample_neg_0, np.dot(self.w_B, hSample_neg))
            wu_sum_A_1[sdata[0]] += np.outer(vSample_pos_1, np.dot(self.w_B, hProb_pos)) - np.outer(vSample_neg_1, np.dot(self.w_B, hSample_neg))
            wu_sum_B += np.outer(np.dot(s_w_A_0.T, vSample_pos_0) + np.dot(s_w_A_1.T, vSample_pos_1), hProb_pos) - np.outer(np.dot(s_w_A_0.T, vSample_neg_0) + np.dot(s_w_A_1.T, vSample_neg_1), hSample_neg)

            wu_sum_v_0[sdata[0]] += vSample_pos_0 - vSample_neg_0
            wu_sum_v_1[sdata[0]] += vSample_pos_1 - vSample_neg_1
            wu_sum_h += hProb_pos - hSample_neg

            if ((batchCounter+1) % batchSize == 0) or ((batchCounter + 1) == numStudents):
                ### Batch finished -> do weight updates
                for i in range(len(norm)):
                    # Avoid division by zero
                    if norm[i] > 0:
                        norm[i] = 1.0 / norm[i]

                # Number of students in this batch
                # Usually equal to batchsize except for last batch
                numCases = (batchCounter % batchSize) + 1
                # Update weightupdates
                self.wu_A_0 = self.wu_A_0 * momentum + (norm * wu_sum_A_0.T).T
                self.wu_A_1 = self.wu_A_1 * momentum + (norm * wu_sum_A_1.T).T
                self.wu_B = self.wu_B * momentum + (1.0 / numCases) * wu_sum_B
                self.wu_v_0 = self.wu_v_0 * momentum + (norm * wu_sum_v_0)
                self.wu_v_1 = self.wu_v_1 * momentum + (norm * wu_sum_v_1)
                self.wu_h = self.wu_h * momentum + (wu_sum_h / numCases)

                # Update weights
                self.w_A_0 += epsilonW * (self.wu_A_0 - weightCost * self.w_A_0)
                self.w_A_1 += epsilonW * (self.wu_A_1 - weightCost * self.w_A_1)
                self.w_B += 0.1 * epsilonW * (self.wu_B - weightCost * self.w_B)

                self.w_v_0 += epsilonVB * self.wu_v_0
                self.w_v_1 += epsilonVB * self.wu_v_1
                self.w_h += epsilonHB * self.wu_h

                # set update and counting variables to zero
                wu_sum_A_0 = np.zeros((self.numVis, self.numC)) 
                wu_sum_A_1 = np.zeros((self.numVis, self.numC))
                wu_sum_B = np.zeros((self.numC, self.numHid))
                wu_sum_v_0 = np.zeros(self.numVis)
                wu_sum_v_1 = np.zeros(self.numVis)
                wu_sum_h = np.zeros(self.numHid)
                norm = np.zeros(self.numVis)

            #Finished processing student -> update counter that counts students in a batch
            batchCounter += 1
            
        # Epoch finished -> return reconstruction error
        return np.mean(recMSE)
        
    def calculatePrediction(self, testSet):
        predictions = np.zeros(len(testSet))
        counter = 0
        for (cor, sID, qID) in testSet:
            sData = [np.array(self.students[sID][0]), np.array(self.students[sID][1])]

            s_w_A_0 = self.w_A_0[sData[0]]
            s_w_A_1 = self.w_A_1[sData[0]]

            #hProb = self.logistic(np.dot(sData[1], s_w_vh_0) + np.dot(1 - sData[1], s_w_vh_1) + self.w_h)
            #a_v_0 = np.exp(self.w_v_0[qID] + np.dot(self.w_vh_0[qID], hProb))
            #a_v_1 = np.exp(self.w_v_1[qID] + np.dot(self.w_vh_1[qID], hProb))

            hProb = self.logistic(np.dot(np.dot(sData[1], s_w_A_0), self.w_B) + 
                             np.dot(np.dot(1 - sData[1], s_w_A_1), self.w_B) + self.w_h) 

            a_v_0 = np.exp(self.w_v_0[qID] + np.dot(self.w_A_0[qID], np.dot(self.w_B, hProb)))
            a_v_1 = np.exp(self.w_v_1[qID] + np.dot(self.w_A_1[qID], np.dot(self.w_B, hProb)))

            predictions[counter] =  a_v_0 / (a_v_0 + a_v_1)
            counter += 1
        return predictions
