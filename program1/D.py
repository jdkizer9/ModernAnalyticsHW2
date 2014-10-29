__author__ = 'jdk288'
import code.utils as utils
from config import EXAMPLE_DATA,TRIP_DATA_1,TRIP_DATA_2, TRAIN_DATA
import numpy as np

##this problem uses the same features as 1c, but we need to add a more intelligent scaling function
##this should return the final scaling function used
def problem1d(useStandardization=True):

	##Problem 1D Parameters
	k = 1

	#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]

	str_fields = [5]
	float_fields = [10, 11, 12, 13, 9]
	observed_float_fields = [8]
	features = (0, 1, 2, 3, 4, 5)

	# Derive a filter from example data
	generatorCreatingFunctionForFilter = utils.createFunctionForGenerator(str_fields, float_fields+observed_float_fields)
	mean_dev_filter = utils.derive_filter(generatorCreatingFunctionForFilter(TRAIN_DATA))
	#generatorCreatingFunction(training_file, training_entry_count)

	# Generate a scale transformer using only indexes which are used as features, use filter derived previously
	generatorCreatingFunctionForScaleTransform = utils.createFunctionForGenerator(str_fields, float_fields+observed_float_fields,row_filter=mean_dev_filter)
	training_data = np.array([row for row in generatorCreatingFunctionForScaleTransform(TRAIN_DATA)])
	#print len(training_data)

	generatorCreatingFunctionForScaleTransform = utils.createFunctionForGenerator(str_fields, float_fields+observed_float_fields,row_filter=mean_dev_filter, row_tranformer=utils.simplePickupTimeTransformer)
	scale_transform = utils.derive_mean_dev_scale_transform(generatorCreatingFunctionForScaleTransform(TRAIN_DATA), (0, 1, 2, 3, 4, 5))

	if useStandardization:
		transform = utils.composeFunctions([utils.simplePickupTimeTransformer, scale_transform])
	else:
		transform = utils.simplePickupTimeTransformer
	# generatorCreatingFunction= createFunctionForGenerator(str_fields, float_fields+observed_float_fields,row_filter=mean_dev_filter, row_tranformer=transform)
	# training_data = np.array([row for row in generatorCreatingFunction(TRAIN_DATA)])
	# print len(training_data)
	(training_data, neighborModel) = utils.trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA, row_filter=mean_dev_filter, row_tranformer=transform)
	#print len(training_data)
	training_observed = training_data[:,len(str_fields) + len(float_fields)]

	(rms, R, mae) = utils.testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
	TRIP_DATA_1, test_entry_count=10000, row_filter=mean_dev_filter, row_tranformer=transform)

	print "********* RESULTS FOR 1.D **************\n"
	utils.printResults(rms, R, mae)
	print "\n****************************************"

	return (mean_dev_filter, transform)

if __name__ == '__main__':
	problem1d()