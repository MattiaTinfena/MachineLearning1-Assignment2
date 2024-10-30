import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from loadData import loadData
from oneDimLRM import *

######################
# Task 1 - Data load #
######################

dataEx = pd.read_csv("turkish-se-SP500vsMSCI.csv")
dataCar = pd.read_csv("mtcarsdata-4features.csv")

rowEx , colEx = loadData(dataEx)
model, mpg, disp, hp, weight = loadData(dataCar)

#######################################################
# Task 2.1 One dimension problem without interception #
#######################################################

w = oneDimLRM(dataEx)

# Plot results

plt.scatter(rowEx, colEx, marker='x') 
x_vals = np.linspace(min(rowEx), max(rowEx), 100)
y_vals = w * x_vals
plt.plot(x_vals, y_vals, color = "Red")
plt.xlabel("Car weight (lbs/1000)")
plt.ylabel("mpg")
plt.show()

##################################################################
# Task 2.2 Comparison one dimension problem without interception #
##################################################################

test = 5
wEx =[]

for n in range(test):
    datasetIndices = np.random.permutation(len(rowEx))[:round(len(rowEx)/10)]
    dataset = dataEx.iloc[datasetIndices]
    res = oneDimLRM(dataset)
    wEx.append(res)

# Plot results

plt.scatter(rowEx, colEx, marker='x') 
x_vals = np.linspace(min(rowEx), max(rowEx), 100)

for i, coeff in enumerate(wEx):
    y_vals = coeff * x_vals  
    plt.plot(x_vals, y_vals, label=f'w{i+1}= {coeff:.3f}')

plt.legend()
plt.xlabel("Standard and Poor's 500 rturn index")
plt.ylabel("MSCI Europe index")
plt.show()

####################################################
# Task 2.3 One dimension problem with interception #
####################################################

w0Car, w1Car = oneDimLRMInt(pd.DataFrame({'x': weight, 'y': mpg}))

# Plot results

plt.scatter(weight, mpg, marker='x') 
x_vals = np.linspace(min(weight), max(weight), 100)
y_vals = w0Car + (w1Car * x_vals)  
plt.plot(x_vals, y_vals, color = "Red")
plt.xlabel("Car weight (lbs/1000)")
plt.ylabel("mpg")
plt.show()

######################################################
# Task 2.4 Multi dimension problem with interception #
######################################################

x = pd.DataFrame({'disp': disp, 'hp': hp, 'weight': weight, 'mpg': mpg })
W, predictions = multiDimLRM(x)

error = [mpg[i] - predictions[i] for i in range(len(mpg))]

comparison = np.column_stack((mpg, predictions,error))

# Print results
print("\n")

comparison = np.around(comparison, decimals=3)

column_headers = ["MPG", "MPG Prediction", "Error3"]
comparison_df = pd.DataFrame(comparison, columns=column_headers)

print(comparison_df.to_string(index=False))
print("\n")

################################
# Task 3 Test regression model #
################################

rep = 10

msetrain1 = []
msetest1 = []
msetrain3 = []
msetest3 = []
msetrain4 = []
msetest4 = []

for i in range(rep):
    
    objTrainig1 = 0
    objTest1 = 0
    objTrainig3 = 0
    objTest3 = 0
    objTrainig4 = 0
    objTest4 = 0

    trainingSetIndicesEx = np.random.permutation(len(rowEx))[:round(len(rowEx)*0.05)]
    trainingSetEx = dataEx.iloc[trainingSetIndicesEx]
    trainingSetIndicesCar = np.random.permutation(len(mpg))[:round(len(mpg)*0.05)]
    trainingSetCar = dataCar.iloc[trainingSetIndicesCar]

    testSetIndicesEx = np.setdiff1d(np.arange(len(rowEx)), trainingSetIndicesEx)
    testSetEx = dataEx.iloc[testSetIndicesEx]
    testSetIndicesCar = np.setdiff1d(np.arange(len(mpg)), trainingSetIndicesCar)
    testSetCar = dataCar.iloc[testSetIndicesCar]

    trainingSetRowEx, trainingSetColEx = loadData(trainingSetEx)
    testSetRowEx, testSetColEx = loadData(testSetEx)
    trainingSetModel, trainingSetMpg, trainingSetDisp, trainingSetHp, trainingSetWeight =  loadData(trainingSetCar)
    testSetModel, testSetMpg, testSetDisp, testSetHp, testSetWeight =  loadData(testSetCar)


    w = oneDimLRM(trainingSetEx)

    for i in range(len(trainingSetRowEx)):
        trainingPred1 = w * trainingSetRowEx[i]
        trainingError1 = (trainingSetColEx[i] - trainingPred1)**2
        objTrainig1 += trainingError1

    msetrain1.append(objTrainig1/len(trainingSetRowEx))

    for i in range(len(testSetRowEx)):
        testPred1 = w * testSetRowEx[i]
        testError1 = (testSetColEx[i] - testPred1)**2
        objTest1 += testError1

    msetest1.append(objTest1/len(testSetRowEx))
    
    w0TrainCar, w1TrainCar = oneDimLRMInt(pd.DataFrame({'x': trainingSetWeight, 'y': trainingSetMpg}))

    for i in range(len(trainingSetWeight)):
        trainingPred3 = (w1TrainCar * trainingSetWeight[i]) + w0TrainCar
        trainingError3 = (trainingSetMpg[i] - trainingPred3)**2
        objTrainig3 += trainingError3

    msetrain3.append(objTrainig3/len(trainingSetWeight))

    for i in range(len(testSetWeight)):
        testPred3 = (w1TrainCar * testSetWeight[i]) + w0TrainCar
        testError3 = (testSetMpg[i] - testPred3)**2
        objTest3 += testError3

    msetest3.append(objTest3/len(testSetWeight))

    xtrain = pd.DataFrame({'disp': trainingSetDisp, 'hp': trainingSetHp, 'weight': trainingSetWeight, 'mpg': trainingSetMpg })
    Wtrain,predTraining = multiDimLRM(xtrain)
    for i in range(len(trainingSetMpg)):
        objTrainig4 += (trainingSetMpg[i] - predTraining[i])**2

    msetrain4.append(objTrainig4/2)


    xtest = pd.DataFrame({'disp': testSetDisp, 'hp': testSetHp, 'weight': testSetWeight, 'mpg': testSetMpg })
    Wtest,predTest = multiDimLRM(xtest,Wtrain)
    for i in range(len(testSetMpg)):
        objTest4 += (testSetMpg[i] - predTest[i])**2
    msetest4.append(objTest4/2)

# Print results

comparison = np.column_stack((msetrain1, msetest1, msetrain3, msetest3, msetrain4, msetest4))

column_headers = ["MSE Train 1", "MSE Test 1", "MSE Train 3", "MSE Test 3", "MSE Train 4", "MSE Test 4"]
comparison_df = pd.DataFrame(comparison, columns=column_headers)

comparison_df["MSE Train 1"] = comparison_df["MSE Train 1"].map(lambda x: f"{x:.3e}")
comparison_df["MSE Test 1"] = comparison_df["MSE Test 1"].map(lambda x: f"{x:.3e}")
comparison_df["MSE Train 3"] = comparison_df["MSE Train 3"].round(3)
comparison_df["MSE Test 3"] = comparison_df["MSE Test 3"].round(3)
comparison_df["MSE Train 4"] = comparison_df["MSE Train 4"].round(3)
comparison_df["MSE Test 4"] = comparison_df["MSE Test 4"].round(3)

print(comparison_df.to_string(index=False))