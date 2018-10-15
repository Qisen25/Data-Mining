"""	Author: Kei Sum Wang -id:19126089
	Data mining COMP3009 assignment
"""
# reference: http://fracpete.github.io/python-weka-wrapper/api.html
import numpy as np
import weka.core.jvm as jvm
import weka.core.converters as converters
import weka.plot.graph as graph 
import weka.plot.classifiers as plcls
from weka.filters import Filter
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, PredictionOutput, Evaluation
from weka.core.classes import Random


"""function to return index of attribute with significant num of missing values"""
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

"""function to return index of attributes that do not have many distinct values"""
#distinct values make up only 1% of the attribute	
def notDistinct(data):
	strang = ""
	num_attr = data.num_attributes
	num_instances = data.num_instances
	all_attribute_stats = data.attribute_stats
	
	for att in range(0,num_attr):
		stats = all_attribute_stats(att)
		attribute = data.attribute(att)
		
		pctNotDist = float(stats.distinct_count) / float(num_instances)
		if pctNotDist < 0.01:
			strang += str(attribute.index+1) + ','

	strang = strang[:-1] #remove last comma from string
	#print(strang)
	
	return strang

"""function handling unsupervised filters"""
def unsupFilters(data, fType, ops):

	filt = Filter(classname="weka.filters.unsupervised." + fType, options = ops)
	filt.inputformat(data)     # let the filter know about the type of data to filter
	filtered = filt.filter(data)   # filter the data
	
	return filtered 

"""function handling supervised filters"""
def supFilters(data, fType, ops):

	filt = Filter(classname="weka.filters.supervised." + fType, options = ops)
	filt.inputformat(data)     # let the filter know about the type of data to filter
	filtered = filt.filter(data)   # filter the data
	
	return filtered
	
def naiveBayes(data):
	
	classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes", options=["-D"])
	nfolds=13
	rnd = Random(0)
	evaluation = Evaluation(data)
	evaluation.crossvalidate_model(classifier, data,
	nfolds, rnd)
	print(" Naive Bayes Cross-validation information")
	print(evaluation.summary())
	print("precision: " + str(evaluation.precision(1)))
	print("recall: " + str(evaluation.recall(1)))
	print("F-measure: " + str(evaluation.f_measure(1)))
	print("==confusion matrix==")
	print("     a     b")
	print(evaluation.confusion_matrix)
	print
	#write to file
	f = open("naiveeval.txt", "w")
	f.write(evaluation.summary()) 
	f.write("\n")
	f.write("==confusion matrix==\n")
	f.write("     a       b\n")
	for item in evaluation.confusion_matrix:
		f.write("%s\n" % item)
	f.close() 
	#plot roc graph
	plcls.plot_roc(evaluation, title="Naive Bayes ROC", outfile="NBROC", wait=True)
	
	return evaluation.percent_correct
	
"""IBK cross validation"""		
def IBK(data):
	
	classifier = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "5"])
	nfolds=13
	rnd = Random(0)
	evaluation = Evaluation(data)
	evaluation.crossvalidate_model(classifier, data,
	nfolds, rnd)
	print(" IBk Cross-validation information")
	print(evaluation.summary())
	print("precision: " + str(evaluation.precision(1)))
	print("recall: " + str(evaluation.recall(1)))
	print("F-measure: " + str(evaluation.f_measure(1)))
	print("==confusion matrix==")
	print("     a     b")
	print(evaluation.confusion_matrix)
	print
	#write to file
	f = open("IBKeval.txt", "w")
	f.write(evaluation.summary()) 
	f.write("\n")
	f.write("==confusion matrix==\n")
	f.write("     a       b\n")
	for item in evaluation.confusion_matrix:
		f.write("%s\n" % item)
	f.close() 
	#plot roc graph
	plcls.plot_roc(evaluation, title="IBk ROC", outfile="IBKROC", wait=True)
	
	
	return evaluation.percent_correct

"""treeJ48 function"""	
def treeJ48(data):
	
	classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-B","-R", "-N", "13"])
	nfolds=13
	rnd = Random(0)
	evaluation = Evaluation(data)
	evaluation.crossvalidate_model(classifier, data,
	nfolds, rnd)
	print(" J48 Tree Cross-validation information")
	print(evaluation.summary())
	print("precision: " + str(evaluation.precision(1)))
	print("recall: " + str(evaluation.recall(1)))
	print("F-measure: " + str(evaluation.f_measure(1)))
	print("==confusion matrix==")
	print("     a     b")
	print(evaluation.confusion_matrix)
	print
	#write to file
	f = open("J48eval.txt", "w")
	f.write(evaluation.summary()) 
	f.write("\n")
	f.write("==confusion matrix==\n")
	f.write("     a       b\n")
	for item in evaluation.confusion_matrix:
		f.write("%s\n" % item)
	f.close() 
	#plot roc graph
	plcls.plot_roc(evaluation, title="J48 ROC", outfile="J48ROC", wait=True)

	return evaluation.percent_correct
	
def trainAndMakePred(train, test):
	#IBK test and prediction 
	classifierIBK = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "5"])
	classifierIBK.build_classifier(train)
	evaluationIBK = Evaluation(train)
	predicted_labelsIBK = evaluationIBK.test_model(classifierIBK, train)
	print(" IBKTraining information ")
	print(evaluationIBK.summary())
	pred_outputIBK = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
	evaluationIBK = Evaluation(test)
	predicted_indicesIBK = evaluationIBK.test_model(classifierIBK, test, pred_outputIBK)
	print(" IBK Prediction information ")
	print(pred_outputIBK)
	
	#Naive bayes and prediction
	classifierNB = Classifier(classname="weka.classifiers.bayes.NaiveBayes", options=["-D"])
	classifierNB.build_classifier(train)
	evaluationNB = Evaluation(train)
	predicted_labelsNB = evaluationNB.test_model(classifierNB, train)
	print(" Naive Bayes Training information ")
	print(evaluationNB.summary())
	pred_outputNB = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
	evaluationNB = Evaluation(test)
	predicted_indicesNB = evaluationNB.test_model(classifierNB, test, pred_outputNB)
	print(" Naive Bayes Prediction information ")
	print(pred_outputNB)
	
	#out put predictions to file
	a = 1
	ID = 901
	f = open("predict.csv", "w")
	f.write("ID,Predict 1,Predict 2\n")
	for pred1, pred2 in zip(predicted_indicesIBK, predicted_indicesNB):
		f.write("%s,%s,%s\n" % (ID,pred1,pred2))
		ID += 1
	f.close() 
	
	#return predicted_labels
	


"""function to prepare test and train"""	
def preparation():
	data_file="csvfiles/data.csv"

	try:
		#Load data
		loader=Loader(classname="weka.core.converters.CSVLoader")
		data=loader.load_file(data_file)
		data.class_is_last()
		
		miss = mostMissing(data)#find attributes with significant missing data
		data = unsupFilters(data, "attribute.Remove", ["-R", "1,"+ miss])#remove id and most mising value attributes
		data = unsupFilters(data, "attribute.RemoveUseless", [])#remove use less attributes
		
		nonDistinct = notDistinct(data)#find attributes that are not distinct and convert them to nominal
		data = unsupFilters(data, "attribute.NumericToNominal", ["-R", "last," + nonDistinct])#class convert to nominal
		data = unsupFilters(data, "attribute.ReplaceMissingValues", [])#replace missing values
		data = unsupFilters(data, "attribute.Normalize", [])#normalize attributes to create less bias
		
		#split data into test and training set
		test = unsupFilters(data, "instance.RemoveRange", ["-V", "-R", "901-1000"])
		train = unsupFilters(data, "instance.RemoveRange", ["-R", "901-1000"])
		train = supFilters(train, "instance.SMOTE", ["-P", "160.0"])
		#print(data)
		
		saver = Saver(classname="weka.core.converters.ArffSaver")
		saver.save_file(test, "test.arff")
		saver.save_file(train, "train.arff")
		
		#Performing cross validation using naiveBayes, IBk and j48 tree
		#get the accuracy of the cross validation and store all in an array
		accuracyArray = [naiveBayes(train), IBK(train), treeJ48(train)]
		mostAccurate = max(accuracyArray)#find most accurate
		print(mostAccurate)
		accuracyArray.remove(mostAccurate)
		secondAcc = max(accuracyArray)#get second accurate
		print(secondAcc)
		trainAndMakePred(train, test)
		#makePrediction(test)
		
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
