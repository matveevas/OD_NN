from __future__ import division
from __future__ import print_function

from elephas.spark_model import SparkModel
from pyspark.sql import SparkSession


from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import *
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

spark = SparkSession\
    .builder\
    .appName("Python Spark SQL basic example")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()


numpy.random.seed(7)
# load data
df3 = spark.read.csv("/Users/svetlana.matveeva/Documents/MasterThesis/Dataset/joinresult/part-00000-3a94d824-4491-4a1a-b650-e61bd752ed5a-c000.csv")
df1 = df3.select("_c1", regexp_replace("_c0", "POLYGON [(][(]", "").alias("polygon"), regexp_replace("_c2", "POINT [(]", "").alias("point"), "_c3", "_c4", "_c5")
df2 = df1.select("_c1", regexp_replace("polygon", "[)][)]", "").alias("polygon"), regexp_replace("point", "[)]", "").alias("point"), "_c3", "_c4", "_c5")
df = df2.withColumnRenamed("_c1", "polygonID").withColumnRenamed("_c3", "pointID").withColumnRenamed("_c4", "addresstext").withColumnRenamed("_c5", "createddatetime")
df.createTempView("df")
dfCNT = spark.sql("select count(pointID) as count, createddatetime from df where polygonID = 63 group by  createddatetime  order by createddatetime")
dfCNT.show(150)


t1 = dfCNT.select("count").collect()
print(type(t1))
inputdata= Vectors.dense(dfCNT.select("count").collect())
# numpy.random.seed(7)
# dataframe = pandas.read_csv("/Users/svetlana.matveeva/Documents/MasterThesis/Dataset/joinresult/international-airline-passengers.csv", usecols=[1], engine='python', skipfooter=3)
# inputdata = dataframe.values
# inputdata = inputdata.astype('float32')
#
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(inputdata)
#
# # split into train and test sets
# train_size = int(len(inputdata) * 0.67)
# test_size = len(inputdata) - train_size
# train, test = inputdata[0:train_size, :], inputdata[train_size:len(inputdata), :]
# print(len(train), len(test))
#
# # reshape into X=t and Y=t+1
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
#
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#
# look_back = 1
# # create the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
#
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# inputdata = scaler.fit_transform(inputdata)
# split into train and test sets
train_size = int(len(inputdata) * 0.8)
test_size = len(inputdata) - train_size
train, test = inputdata[0:train_size,:], inputdata[train_size:len(inputdata),:]
# reshape into X=t and Y=t+1
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
rdd = to_simple_rdd(spark.sparkContext, trainX, trainY)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimazer='adam')
model.fit(trainX, trainY, epochs=300, batch_size=1, verbose=2)

# adam = elephas_optimizers.Adam()
#
# spark_model = SparkModel(spark.sparkContext, model, optimizer=adam, frequency='epoch', num_workers=2)
# spark_model.train(rdd, nb_epoch=50, batch_size=4, verbose=2, validation_split=0.1)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(inputdata)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(inputdata)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(inputdata)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(inputdata))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


#
# from pyspark import SparkContext, SparkConf
# conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
# sc = SparkContext(conf=conf)
#
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
# model = Sequential()
# model.add(Dense(128, input_dim=784))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=SGD())
#
# from elephas.utils.rdd_utils import to_simple_rdd
# rdd = to_simple_rdd(sc, trainX, trainY)
#
# from elephas.spark_model import SparkModel
# from elephas import optimizers as elephas_optimizers
#
# adagrad = elephas_optimizers.Adagrad()
# spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=2)
# spark_model.train(rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1)

