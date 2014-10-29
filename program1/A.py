__author__ = 'jdk288'
import code.utils as utils
from config import EXAMPLE_DATA,TRIP_DATA_1,TRIP_DATA_2, TRAIN_DATA
import numpy as np
import matplotlib.pyplot as plt

def bucketArray(data, xmin, xmax, ymin, ymax, bucketsPerDim=100):
	deltaX = (xmax-xmin)/bucketsPerDim
	deltaY = (ymax-ymin)/bucketsPerDim
	#print xmin, xmax, deltaX
	#print ymin, ymax, deltaY
	def bucketMap(point):
		#print point
		xBucket = int(round((point[0]-xmin)/deltaX))
		if xBucket == bucketsPerDim:
			xBucket = bucketsPerDim-1
		yBucket = int(round((point[1]-ymin)/deltaY))
		if yBucket == bucketsPerDim:
			yBucket = bucketsPerDim-1
		return (xBucket, yBucket)

	buckets = np.zeros((bucketsPerDim, bucketsPerDim))

	for point in data:
		bucket = bucketMap(point)
		#print bucket
		buckets[bucket] = buckets[bucket] + 1.

	#print buckets
	return buckets


#this takes WAY too long
def plotHeatMapForNumberOfPassengers(data, numberOfPassengers, numberOfBucketsPerDim=100):


	minima = np.amin(data, axis=0)
	maxima = np.amax(data, axis=0)

	xMin = minima[1]
	xMax = maxima[1]

	yMin = minima[2]
	yMax = maxima[2]

	passengerData = [row[1:3] for row in data if row[0] == numberOfPassengers]
	#passengerData = [row[1:3] for row in data]

	#data is scaled by 1/n
	array = bucketArray(passengerData, xMin, xMax, yMin, yMax, numberOfBucketsPerDim) / len(passengerData)

	#print np.amin(array, axis=0)
	#print np.amin(array, axis=1)

	#print np.amax(array, axis=0)
	#print np.amax(array, axis=1)




	#numberOfBuckets = 100
	# import matplotlib.pyplot as plt
	# import numpy as np
	#column_labels = range(10*numberOfBuckets, 10)
	#print column_labels
	#row_labels = range(numberOfBuckets)
	#print row_labels
	#data = np.random.rand(numberOfBuckets,numberOfBuckets)/10
	#print data
	fig, ax = plt.subplots()
	# heatmap = ax.pcolor(array, cmap=plt.cm.Reds, alpha=0.8)
	heatmap = ax.pcolor(array, cmap=plt.cm.Reds)
	#print data

	#ax.set_xlim(-10., 10.)

	#legend
	cbar = plt.colorbar(heatmap)
	# cbar.ax.set_yticklabels(['0','1','2','>3'])
	# cbar.set_label('# of contacts', rotation=270)

	# put the major ticks at the middle of each cell, notice "reverse" use of dimension
	#ax.set_yticks([1,2,3,4,5], minor=False)
	#ax.set_xticks([19, 20, 21, 22, 23], minor=False)

	dx = (xMax-xMin)/5
	xlabels = [round(xMin + dx*i, 2) for i in range(6)]
	ax.set_xticklabels(xlabels, minor=False)

	dy = (yMax-yMin)/5
	ylabels = [round(yMin + dy*i, 2) for i in range(6)]
	ax.set_yticklabels(ylabels, minor=False)
	# ax.set_yticklabels(column_labels, minor=False)
	#plt.show()
	fileString = 'passengers{0}HeatMap.png'.format(numberOfPassengers)
	plt.savefig(fileString)
	plt.close()




def problem1a():

	#Problem 1A Parameters
	str_fields = []
	float_fields = [7, 12, 13]

	badDataFilter = utils.derive_BadDataFilter([0, 1, 2])
	compositeFilter = utils.composeFilters([badDataFilter, utils.NYCFilter])
	generatorCreatingFunctionForFilter = utils.createFunctionForGenerator(str_fields, float_fields,row_filter=compositeFilter)
	mean_dev_filter = utils.derive_meanDevFilter(generatorCreatingFunctionForFilter(TRAIN_DATA), [1,2], tolerance=3)

	compositeFilter = utils.composeFilters([badDataFilter, utils.NYCFilter, mean_dev_filter])
	generatorCreatingFunction = utils.createFunctionForGenerator(str_fields, float_fields,row_filter=compositeFilter)
	#training_data = np.array([row for row in generatorCreatingFunction(TRAIN_DATA)])
	training_data = np.array([row for row in generatorCreatingFunction(TRAIN_DATA)])

	#print len(training_data)
	plotHeatMapForNumberOfPassengers(training_data, 1, 100)
	plotHeatMapForNumberOfPassengers(training_data, 3, 100)

	

	


	# filters = composeFunctions([passenger1filter, mean_dev_filter])
	
	# generatorCreatingFunctionFor1Passenger = createFunctionForGenerator(str_fields, float_fields,row_filter=filters)
	# passenger1data = np.array([row[1:3] for row in generatorCreatingFunctionFor1Passenger(TRAIN_DATA)])

	# #h1 = estimateBandwidth(passenger1data)

	# print passenger1data
	# print len(passenger1data)
	# kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(passenger1data)
	# print ''
	#print kde.score_samples(passenger1data)

	#passenger1data = None
	#plotKernel(passenger1data)

if __name__ == '__main__':
	problem1a()
