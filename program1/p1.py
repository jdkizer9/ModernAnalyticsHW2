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
from datetime import datetime, date, time


##return a function that takes in a filename and a number of rows and returns two generator
#fname,str_fields,float_fields,exclude_first=True,row_filter=ifilter,row_tranformer=itransformer
def createFunctionForGenerator(str_fields, float_fields, row_filter=utils.ifilter,row_tranformer=utils.itransformer):

	def createGenerator(fname, count=0, exclude_first=True):
		if(count > 0):
			#print "returning custom generator"
			def customGenerator():
				localCount = 0
				for row in utils.load_csv_lazy(fname, str_fields, float_fields, exclude_first, row_filter, row_tranformer):
					#print "loading row"
					localCount += 1

					yield row

					if localCount >= count:
						break
						

			return customGenerator()
		else:
			#print "returning default generator"
			return utils.load_csv_lazy(fname, str_fields, float_fields, exclude_first, row_filter, row_tranformer)

	return createGenerator


def TestEntryToPredictedValueForKNeighborsMappingFunction(neighborModel, trainingObserved, kneighbors=5):
	def mapTestEntryToPredictedValue(entry):
		neighbors = neighborModel.kneighbors(entry, return_distance=False, n_neighbors=kneighbors, )[0]
		neighborObservedValues = [trainingObserved[index] for index in neighbors]
		#print neighborObservedValues
		median = np.median(neighborObservedValues)
		#print median
		return np.median(neighborObservedValues)

	return mapTestEntryToPredictedValue

def squaredErrorMappingFunction(observed, predicted):
	error = observed - predicted
	return (error)**2

def absoluteErrorMappingFunction(observed, predicted):
	error = observed - predicted
	return math.fabs(error)


#returns (training_data, neighborsModel)
def trainModel(k, str_fields, float_fields, observed_float_field, 
	training_file, training_entry_count=0, row_filter=utils.ifilter, row_tranformer=utils.itransformer):
	generatorCreatingFunction = createFunctionForGenerator(str_fields, float_fields+observed_float_field, row_filter, row_tranformer)

	training_data = np.array([row for row in generatorCreatingFunction(training_file, training_entry_count)])

	#print training_data[0]
	#print len(training_data)
	colList = list(range(len(str_fields)+len(float_fields)))
	training_X = training_data[:,colList]

	neigh = NearestNeighbors(n_neighbors=k)
	neigh.fit(training_X) 

	return (training_data, neigh)

def testModel(model, training_observed, k, str_fields, float_fields, observed_float_field, 
	test_file, test_entry_count=0, row_filter=utils.ifilter, row_tranformer=utils.itransformer):
	generatorCreatingFunction = createFunctionForGenerator(str_fields, float_fields+observed_float_field, row_filter, row_tranformer)

	try:
		test_data = np.array([row for row in generatorCreatingFunction(test_file, test_entry_count)])
	except Exception as inst:
		print type(inst)     # the exception instance
		print inst.args      # arguments stored in .args
		print inst           # __str__ allows args to be printed directly
		x, y = inst.args
		print 'x =', x
		print 'y =', y


	#map test set to predicted values
	colList = list(range(len(str_fields)+len(float_fields)))
	test_X = test_data[:,colList]
	#assume that our output vector is always trip time
	test_observed = test_data[:,len(str_fields)+len(float_fields)]

	mappingFunction = TestEntryToPredictedValueForKNeighborsMappingFunction(model, training_observed, k)
	test_predicted = [mappingFunction(x) for x in test_X]

	rootMeanSquaredError = math.sqrt((sum(map(squaredErrorMappingFunction, test_observed, test_predicted)))/len(test_observed))
	correlationCoef = stats.pearsonr(test_observed, test_predicted)[0]
	meanAbsoluteError = (sum(map(absoluteErrorMappingFunction, test_observed, test_predicted)))/len(test_observed)

	return (rootMeanSquaredError, correlationCoef, meanAbsoluteError)

def composeFunctions(functions):
	newFunctions = list(functions)
	newFunctions.reverse()
	return reduce(lambda f, g: lambda x: f(g(x)), newFunctions)

#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]
def derive_filter(rows,tolerance = 4.0):
    """
    Generates a custom filter function  on trip_distance alone
    :param trip_dist_mean:
    :param trip_dist_std:
    :param tolerance:
    :return:

    !IMPORTANT Filters are always applied before transformers!
    """
    distances = np.array([row[5] for row in rows]) # InMemory
    trip_dist_mean = np.mean(distances)
    trip_dist_std = np.std(distances)
    logging.debug("Dervied mean "+str(trip_dist_mean))
    logging.debug("Dervied std "+str(trip_dist_std))
    def custom_filter(row):
        if row[1] != 0.0 and row[2] != 0.0 and row[3] != 0.0 and row[4] != 0.0 and row[5] != 0.0: # filters out rows with zero elements
            plong,plat,dlong,dlat=row[1:5]
            if 100 > get_distance(plat,plong,dlat,dlong) > 0 and ((row[5] - trip_dist_mean) / trip_dist_std) < tolerance:
                return True
        # plong,plat,dlong,dlat=row[1:5]
        # print 'Filtering {0}: Distance: {1} Mean: {2} STD: {3}'.format(row, get_distance(plat,plong,dlat,dlong), trip_dist_mean, trip_dist_std)
        return False
    return custom_filter




def testTransform1(row):
	print "Test Transform 1"
def testTransform2(row):
	print "Test Transform 2"
def testTransform3(row):
	print "Test Transform 3"
def testTransform4(row):
	print "Test Transform 4"

# def rowTransformerFromListOfTransforms(transformList):
	
# 	for transform in transforList:
# 		row = transform(row)
# 	return row

#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]
#Transforms pickup_time into minutes from the beginning of the week
def simplePickupTimeTransformer(row):
	#print row
	pickupDate = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
	minutes = (((float(pickupDate.time().hour) * 60.0) + float(pickupDate.time().minute))) + float(24*60*pickupDate.date().weekday())
	#minutes = (((float(pickupDate.time().hour) * 60.0) + float(pickupDate.time().minute)))
	row[0] = minutes
	#print row
	return row

#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]
def derive_scale_transform(rows,indexes):
    """
    Generates tran
    :param rows:
    :param indexes:
    :return:
    """
    min_values,max_values = {},{}
    first = True
    for row in rows:
        #row[0] = int(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S" ).strftime("%s"))
        for index in indexes:
            if not first:
                max_values[index] = max(max_values[index],row[index])
                min_values[index] = min(min_values[index],row[index])
            else:
                max_values[index] = row[index]
                min_values[index] = row[index]
        first = False
    logging.debug("scale values min "+str(min_values))
    logging.debug("scale values max "+str(max_values))
    def custom_transform(row):
        try:
            #row[0] = int(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S" ).strftime("%s"))
            for index in indexes:
                row[index] = (row[index] - min_values[index]) / (max_values[index]-min_values[index])
            return row
        except:
            logging.exception("Scaling error")
            raise ValueError
    return custom_transform

#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]
def derive_mean_dev_scale_transform(rows,indexes):
	"""
	Generates tran
	:param rows:
	:param indexes:
	:return:
	"""
	means,stdDevs = {},{}
	#print indexes
	array = np.array([row for row in rows])
	for index in indexes:
		featureArray = array[:,index]
		#print featureArray
		means[index] = np.mean(featureArray)
		stdDevs[index] = np.std(featureArray)

	#print means
	#print stdDevs

	def custom_transform(row):
		try:
			#row[0] = int(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S" ).strftime("%s"))
			for index in indexes:
				row[index] = ((row[index] - means[index]) / stdDevs[index])
			return row
		except:
			#print 
			logging.exception("Scaling error")
			raise ValueError
	return custom_transform

def printResults(rms, R, mae):
	print 'RMS: {0}'.format(rms)
	print 'Corr Coef: {0}'.format(R)
	print 'MAE: {0}'.format(mae)

def problem1b():

	##Problem 1B Parameters
	k = 1
	str_fields = []
	float_fields = [10, 11, 12, 13]
	observed_float_fields = [8]

	(training_data, neighborModel) = trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA)
	training_observed = training_data[:,len(str_fields) + len(float_fields)]

	(rms, R, mae) = testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
	TRIP_DATA_1, test_entry_count=100000)

	print "********* RESULTS FOR 1.B **************"
	printResults(rms, R, mae)
	print "****************************************\n"

#this problem adds in pickup time (which must be transformed) and trip distance features
def problem1c():

	##Problem 1C Parameters
	k = 1
	str_fields = [5]
	float_fields = [10, 11, 12, 13, 9]
	observed_float_fields = [8]

	generatorCreatingFunctionForFilter = createFunctionForGenerator(str_fields, float_fields+observed_float_fields)
	mean_dev_filter = derive_filter(generatorCreatingFunctionForFilter(TRAIN_DATA))

	(training_data, neighborModel) = trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA, row_filter=mean_dev_filter, row_tranformer=simplePickupTimeTransformer)
	print len(training_data)
	training_observed = training_data[:,len(str_fields) + len(float_fields)]

	(rms, R, mae) = testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
	TRIP_DATA_1, test_entry_count=100000, row_filter=mean_dev_filter, row_tranformer=simplePickupTimeTransformer)

	print "********* RESULTS FOR 1.C **************\n"
	printResults(rms, R, mae)
	print "****************************************\n"

##this problem uses the same features as 1c, but we need to add a more intelligent scaling function
##this should return the final scaling function used
def problem1d():

	##Problem 1D Parameters
	k = 1

	#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]

	str_fields = [5]
	float_fields = [10, 11, 12, 13, 9]
	observed_float_fields = [8]
	features = (0, 1, 2, 3, 4, 5)

	# Derive a filter from example data
	generatorCreatingFunctionForFilter = createFunctionForGenerator(str_fields, float_fields+observed_float_fields)
	mean_dev_filter = derive_filter(generatorCreatingFunctionForFilter(TRAIN_DATA))
	#generatorCreatingFunction(training_file, training_entry_count)

	# Generate a scale transformer using only indexes which are used as features, use filter derived previously
	generatorCreatingFunctionForScaleTransform = createFunctionForGenerator(str_fields, float_fields+observed_float_fields,row_filter=mean_dev_filter)
	training_data = np.array([row for row in generatorCreatingFunctionForScaleTransform(TRAIN_DATA)])
	#print len(training_data)

	generatorCreatingFunctionForScaleTransform = createFunctionForGenerator(str_fields, float_fields+observed_float_fields,row_filter=mean_dev_filter, row_tranformer=simplePickupTimeTransformer)
	scale_transform = derive_mean_dev_scale_transform(generatorCreatingFunctionForScaleTransform(TRAIN_DATA), (0, 1, 2, 3, 4, 5))

	transform = composeFunctions([simplePickupTimeTransformer, scale_transform])
	# generatorCreatingFunction= createFunctionForGenerator(str_fields, float_fields+observed_float_fields,row_filter=mean_dev_filter, row_tranformer=transform)
	# training_data = np.array([row for row in generatorCreatingFunction(TRAIN_DATA)])
	# print len(training_data)
	(training_data, neighborModel) = trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA, row_filter=mean_dev_filter, row_tranformer=transform)
	#print len(training_data)
	training_observed = training_data[:,len(str_fields) + len(float_fields)]

	(rms, R, mae) = testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
	TRIP_DATA_1, test_entry_count=10000, row_filter=mean_dev_filter, row_tranformer=transform)

	print "********* RESULTS FOR 1.D **************\n"
	printResults(rms, R, mae)
	print "\n****************************************"

	return (mean_dev_filter, transform)

def problem1e(mean_dev_filter, transform):

	##Problem 1E Parameters

	#FEATURES: [pickup_time, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_distance]
	for k in range(5, 21):
		str_fields = [5]
		float_fields = [10, 11, 12, 13, 9]
		observed_float_fields = [8]
		features = (0, 1, 2, 3, 4, 5)

		(training_data, neighborModel) = trainModel(k, str_fields, float_fields, observed_float_fields, TRAIN_DATA, row_filter=mean_dev_filter, row_tranformer=transform)
		#print len(training_data)
		training_observed = training_data[:,len(str_fields) + len(float_fields)]

		(rms, R, mae) = testModel(neighborModel, training_observed, k, str_fields, float_fields, observed_float_fields,  
		TRIP_DATA_1, test_entry_count=10000, row_filter=mean_dev_filter, row_tranformer=transform)

		print "********* RESULTS FOR 1.E k={0} **************\n".format(k)
		printResults(rms, R, mae)
		print "\n****************************************"


def main():
	

	problem1b()
	problem1c()
	(mean_dev_filter, transform) = problem1d()
	problem1e(mean_dev_filter, transform)


	




if __name__ == '__main__':
	main()
