import numpy,math,datetime,logging
from sklearn import linear_model
from distance import get_distance
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
from scipy import stats
from datetime import datetime, date, time
# logging.basicConfig(filename='logs/utils.log',level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

##################################################################################################################
##################################################################################################################
##################################################################################################################

def itransformer(row):
    """
    identity transformer returns same
    :param row:
    :return True:
    """
    return row

def ifilter(row):
    """
    identity filter always returns True
    :param row:
    :return True:
    """
    return True

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

def derive_BadDataFilter(indexesToFilter):
    def custom_filter(row):
        #print indexesToFilter
        for index in indexesToFilter:
            if row[index] == 0.:
                #print 'filtered As Bad Data'
                #print index
                #print row
                return False
        return True
    return custom_filter

def NYCFilter(row):
    if (-74.1 < row[1] < -73.7) and (40.5 < row[2] < 40.9):
        return True
    else:
        return False

def derive_meanDevFilter(rows, indexesToFilter, tolerance=4.0):

    means,stdDevs = {},{}
    #print indexes
    array = np.array([row for row in rows])
    for index in indexesToFilter:
        featureArray = array[:,index]
        #print np.amin(featureArray)
        #print np.amax(featureArray)
        means[index] = np.mean(featureArray)
        stdDevs[index] = np.std(featureArray)

    #print means
    #print stdDevs

    def custom_filter(row):
        #print indexesToFilter
        for index in indexesToFilter:
            #print math.fabs(row[index]-means[index])/stdDevs[index]
            if math.fabs(row[index]-means[index])/stdDevs[index] > tolerance:
                # print 'Filtered'
                # print row
                return False
        return True
    return custom_filter


#FEATURES: [passenger_count, dropoff_longitude, dropoff_latitude]
def passengerFilter(passengers):


    def custom_filter(row):
        if row[0] == passengers and row[1] != 0. and row[2] != 0.:
            return True
        else:
            return False
    return custom_filter


##return a function that takes in a filename and a number of rows and returns two generator
#fname,str_fields,float_fields,exclude_first=True,row_filter=ifilter,row_tranformer=itransformer
def createFunctionForGenerator(str_fields, float_fields, row_filter=ifilter,row_tranformer=itransformer):

    def createGenerator(fname, count=0, exclude_first=True):
        if(count > 0):
            #print "returning custom generator"
            def customGenerator():
                localCount = 0
                for row in load_csv_lazy(fname, str_fields, float_fields, exclude_first, row_filter, row_tranformer):
                    #print "loading row"
                    localCount += 1

                    yield row

                    if localCount >= count:
                        break
                        

            return customGenerator()
        else:
            #print "returning default generator"
            return load_csv_lazy(fname, str_fields, float_fields, exclude_first, row_filter, row_tranformer)

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
    training_file, training_entry_count=0, row_filter=ifilter, row_tranformer=itransformer):
    generatorCreatingFunction = createFunctionForGenerator(str_fields, float_fields+observed_float_field, row_filter, row_tranformer)

    training_data = np.array([row for row in generatorCreatingFunction(training_file, training_entry_count)])

    #print training_data[0]
    #print len(training_data)
    colList = list(range(len(str_fields)+len(float_fields)))
    training_X = training_data[:,colList]

    neigh = NearestNeighbors(n_neighbors=k, algorithm='brute')
    neigh.fit(training_X) 

    return (training_data, neigh)

def testModel(model, training_observed, k, str_fields, float_fields, observed_float_field, 
    test_file, test_entry_count=0, row_filter=ifilter, row_tranformer=itransformer):
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

def composeFilters(filters):
    newFilters = list(filters)
    newFilters.reverse()
    return reduce(lambda f, g: lambda x: f(x) and g(x), newFilters)





def testTransform1(row):
    print "Test Transform 1"
def testTransform2(row):
    print "Test Transform 2"
def testTransform3(row):
    print "Test Transform 3"
def testTransform4(row):
    print "Test Transform 4"

# def rowTransformerFromListOfTransforms(transformList):
    
#   for transform in transforList:
#       row = transform(row)
#   return row

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


##################################################################################################################
##################################################################################################################
##################################################################################################################



def metrics(model,x,y):
    """
    compute ols and rmse
    :param y:
    :param yhat:
    :return ols and rmse:
    """
    yhat = model.predict(x)
    ols = sum(numpy.square((y-yhat)))
    rmse = (ols/len(y))**0.5
    corr = numpy.corrcoef(y,yhat)
    return ols,rmse,corr

def evaluate(model_list,x,y):
    for description,model in model_list:
        print "\t",description,"OLS, RMSE and Correlation coefficient",metrics(model,x,y),"Model",model.coef_,model.intercept_


def split(target, features, row, x, y, x_test=None, y_test=None, i= None, nth = None):
    """
    :param target: index of expected
    :param features: list of indexes
    :param row:
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :param i:
    :param nth:
    """

    if nth and i % nth == 0:
        x_test.append([row[feature] for feature in features])
        y_test.append(row[target])
    else:
        x.append([row[feature] for feature in features])
        y.append(row[target])


def tls(model,x,y):
    pass


def linear_regression(x,y):
    """
    :param x:
    :param y:
    :return linear regression model object:
    """
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model


# def itransformer(row):
#     """
#     identity transformer returns same
#     :param row:
#     :return True:
#     """
#     return row

# def ifilter(row):
#     """
#     identity filter always returns True
#     :param row:
#     :return True:
#     """
#     return True


def load_csv_lazy(fname,str_fields,float_fields,exclude_first=True,row_filter=ifilter,row_tranformer=itransformer):
    """
    np.genfromtxt is a good alternative, not sure if it can act as a generator. pandas frames are also a good alternative.
    :param fname:
    :param exclude_first:
    :return:
    """
    error_count = 0
    excluded_count = 0
    for count,line in enumerate(file(fname)):
        if not exclude_first:
            try:
                if count and count % 10**6 == 0:
                    logging.debug("Loaded "+str(count))
                    logging.debug("error_count : "+str(error_count))
                    logging.debug("excluded_count : "+str(excluded_count))
                entries = line.strip().split(',')
                row = [entries[f] for f in str_fields] + [float(entries[f]) for f in float_fields]
                if row_filter(row):
                    row = row_tranformer(row)
                    yield row
                else:
                    excluded_count += 1
            except:
                error_count += 1
        else:
            exclude_first = False
    logging.debug("count : "+str(count))
    logging.debug("error_count : "+str(error_count))
    logging.debug("excluded_count : "+str(excluded_count))