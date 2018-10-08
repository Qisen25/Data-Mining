import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.filters import Filter
from weka.core.converters import Loader, Saver
import numpy as np


def mostMissing(data):
	strang = ""
	num_attr = data.num_attributes
	num_instances = data.num_instances
	all_attribute_stats = data.attribute_stats
	
	for att in range(0,num_attr):
		stats = all_attribute_stats(att)
		attribute = data.attribute(att)
		
		pctMiss = float(stats.missing_count) / float(num_instances)
		if pctMiss > 0.8:
			strang += str(attribute.index+1) + ','

	strang = strang[:-1] #remove last comma from string
	
	return strang

def filters(data, fType, ops):

	filt = Filter(classname="weka.filters.unsupervised.attribute." + fType, options = ops)
	filt.inputformat(data)     # let the filter know about the type of data to filter
	filtered = filt.filter(data)   # filter the data
	
	return filtered 

def preparation():
	data_file="csvfiles/data.csv"

	try:
		#Load data
		loader=Loader(classname="weka.core.converters.CSVLoader")
		data=loader.load_file(data_file)
		data.class_is_last()
		
		miss = mostMissing(data)#find attributes with significant missing data
		data = filters(data, "Remove", ["-R", "1,"+ miss])#remove id and most mising value attributes
		data = filters(data, "RemoveUseless", [])
		data = filters(data, "NumericToNominal", ["-R", "last"])#class convert to nominal
		data = filters(data, "ReplaceMissingValues", [])
		data = filters(data, "Normalize", [])
		print(data)
		
		saver = Saver(classname="weka.core.converters.ArffSaver")
		saver.save_file(data, "test.arff")
		
		print("Data loaded successfully")
	except IOError:
		print("Error loading file " + data_file)

#Main method
if __name__ == '__main__':
	#Start java VM with 1GB heap memory, use system classpath and additional packages
	jvm.start(system_cp=True, packages=True,max_heap_size="1024m")	
	
	#Implement question 1
	preparation()
	
	#Terminate JVM
	jvm.stop()
