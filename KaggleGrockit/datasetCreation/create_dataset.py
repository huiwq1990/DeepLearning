import os

#Parse the csv files given by the kaggle challenge
#and save the data in a more condensed way
print "####  Creating a condensed version of the Grockit dataset"
os.system('python parse_csv_pickle.py')

#Create the dataset for the rbms
print "####  Creating a general RBM dataset"
os.system('python create_rbm_dataset.py')

#Create an additional dataset for the conditional RBM
print "####  Creating an additional dataset for the conditional RBM"
os.system('python create_conditional_dataset.py')