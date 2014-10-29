__author__ = 'jdk288'
import code.utils as utils
from config import EXAMPLE_DATA,TRIP_DATA_1,TRIP_DATA_2, TRAIN_DATA
import numpy as np


def problem1b():

	##Problem 1B Parameters
	k = 1
	str_fields = []
	float_fields = [10, 11, 12, 13]
	observed_float_fields = [8]

	(training_data, neighborModel) = utils.trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA)
	training_observed = training_data[:,len(str_fields) + len(float_fields)]

	(rms, R, mae) = utils.testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
	TRIP_DATA_1, test_entry_count=100000)

	print "********* RESULTS FOR 1.B **************"
	utils.printResults(rms, R, mae)
	print "****************************************\n"

if __name__ == '__main__':
	problem1b()