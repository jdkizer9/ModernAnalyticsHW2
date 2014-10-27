"""
Multi variable linear regression, in-memory and chunked
"""
import logging
logging.basicConfig(filename='logs/p1.log',level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
import datetime
import numpy as np
from config import S_FIELDS,F_FIELDS,EXAMPLE_DATA,TRIP_DATA_1,TRIP_DATA_2, TRAIN_DATA
import code.utils as utils
from code.distance import get_distance
from sklearn.neighbors import NearestNeighbors
import math
from scipy import stats


##return a function that takes in a filename and a number of rows and returns two generator
#fname,str_fields,float_fields,exclude_first=True,row_filter=ifilter,row_tranformer=itransformer
def createFunctionForGenerator(str_fields, float_fields, row_filter=utils.ifilter,row_tranformer=utils.itransformer):

	def createGenerator(fname, count=0, exclude_first=True):
		if(count > 0):
			#print "returning custom generator"
			def customGenerator():
				localCount = 0
				for row in utils.load_csv_lazy(fname, str_fields, float_fields, exclude_first, row_filter, row_tranformer):
					localCount += 1

					yield row

					if localCount >= count:
						break
						

			return customGenerator()
		else:
			#print "returning default generator"
			return utils.load_csv_lazy(fname, str_fields, float_fields, exclude_first, row_filter, row_tranformer)

	return createGenerator


def TestEntryToPredictedValueForKNeighborsMappingFunction(trainingObserved, kneighbors=5):
	def mapTestEntryToPredictedValue(entry):
		neighbors = neigh.kneighbors(entry, return_distance=False, n_neighbors=kneighbors, )[0]
		neighborObservedValues = [trainingObserved[index] for index in neighbors]
		#print neighborObservedValues
		median = np.median(neighborObservedValues)
		#print median
		return np.median(neighborObservedValues)

	return mapTestEntryToPredictedValue

# def squaredErrorMappingFunction(yTuple):
# 	assert(len(yTuple) == 2)
# 	return (yTuple[0][0] - yTuple[1][0])**2

def squaredErrorMappingFunction(observed, predicted):
	error = observed - predicted
	return (error)**2

def absoluteErrorMappingFunction(observed, predicted):
	error = observed - predicted
	return math.fabs(error)


if __name__ == '__main__':

	#assume that our output vector is always trip time

	# k, str_fields, float_fields, observed_float_field, training_file, 
	# training_entry_count, test_file, test_entry_count, 
	# row_filter=utils.ifilter, row_tranformer=utils.itransformer

	k = 5
	##pickup lon/lat, dropoff lon/lat, trip time
	p1bGeneratorCreatingFunction = createFunctionForGenerator([], [10, 11, 12, 13, 8])
	# now example_data is a loadCSV is a generator
	example_data = [row for row in p1bGeneratorCreatingFunction(TRAIN_DATA)]

	training_data = np.array(example_data)
	#print training_data
	training_X = training_data[:,[0, 1, 2, 3]]
	#print training_X

	training_observed = training_data[:,[4]]
	#print training_y


	neigh = NearestNeighbors(n_neighbors=k)
	neigh.fit(training_X) 



	test_data = np.array([row for row in p1bGeneratorCreatingFunction(TRIP_DATA_1, 100000)])

	#map test set to predicted values
	test_X = test_data[:,[0, 1, 2, 3]]
	test_observed = test_data[:,4]

	#print test_observed[0]
	#print len(test_observed)

	mappingFunction = TestEntryToPredictedValueForKNeighborsMappingFunction(training_observed, k)
	test_predicted = [mappingFunction(x) for x in test_X]

	#print test_predicted[0]
	#print len(test_predicted)

	rootMeanSquaredError = math.sqrt((sum(map(squaredErrorMappingFunction, test_observed, test_predicted)))/len(test_observed))
	print rootMeanSquaredError
	correlationCoef = stats.pearsonr(test_observed, test_predicted)
	print correlationCoef[0]
	meanAbsoluteError = (sum(map(absoluteErrorMappingFunction, test_observed, test_predicted)))/len(test_observed)
	print meanAbsoluteError






	# for count, row in enumerate(p1bGeneratorCreatingFunction(TRIP_DATA_2)):
	# 	#print count
	# 	if count >= 10:
	# 		break

	# 	X = row[:4]
	# 	y = row[4]
	# 	print X
	# 	print training_X[count]
	# 	neighbors = neigh.kneighbors(X, return_distance=False)
	# 	print neighbors



	#print example_data

	#print len(example_data)
	#example_data = [row for row in utils.load_csv_lazy(EXAMPLE_DATA,S_FIELDS,F_FIELDS)]
	#logging.debug("inspection for transformation "+str(example_data[-3:]))


