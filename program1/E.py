__author__ = 'jdk288'
import code.utils as utils
from config import EXAMPLE_DATA,TRIP_DATA_1,TRIP_DATA_2, TRAIN_DATA
import numpy as np
import matplotlib.pyplot as plt


def problem1e(useStandardization=True):

	##Problem 1E Parameters

	str_fields = [5]
	float_fields = [10, 11, 12, 13, 9]
	observed_float_fields = [8]
	features = (0, 1, 2, 3, 4, 5)

	# Derive a filter from example data
	generatorCreatingFunctionForFilter = utils.createFunctionForGenerator(str_fields, float_fields+observed_float_fields)
	mean_dev_filter = utils.derive_filter(generatorCreatingFunctionForFilter(TRAIN_DATA))

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

	#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]

	rmsList = []
	RList = []
	maeList = []
	kList = list(range(5,21))




	for k in range(5, 21):
		

		(training_data, neighborModel) = utils.trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA, row_filter=mean_dev_filter, row_tranformer=transform)
		#print len(training_data)
		training_observed = training_data[:,len(str_fields) + len(float_fields)]

		(rms, R, mae) = utils.testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
		TRIP_DATA_1, test_entry_count=10000, row_filter=mean_dev_filter, row_tranformer=transform)

		rmsList.append(rms)
		RList.append(R)
		maeList.append(mae)

		print "********* RESULTS FOR 1.E k={0} **************\n".format(k)
		utils.printResults(rms, R, mae)
		print "\n****************************************"


	# kList = list(range(5,21))
	# rmsList = [2*x for x in kList]
	# RList = [x**2 for x in kList]
	# maeList = [x**3 for x in kList]

	plt.plot(kList, rmsList, 'ro', linewidth=2.0)
	plt.savefig('rmsKplot.png')
	plt.close()

	plt.plot(kList, RList, 'bo', linewidth=2.0)
	plt.savefig('RKplot.png')
	plt.close()

	plt.plot(kList, maeList, 'go', linewidth=2.0)
	plt.savefig('maeKplot.png')
	plt.close()

if __name__ == '__main__':
	problem1e()