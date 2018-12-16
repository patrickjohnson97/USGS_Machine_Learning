from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import time

# This is the Neural Network - DL stands for deep learning
NUM_PARAMS = 3
data = np.genfromtxt('all_month.csv',delimiter=',')
labels = np.empty(len(data))


latitude = data[:,0]
longitude = data[:,1]
depth = data[:,2]
mag = data[:,3]
X = data[:,:NUM_PARAMS:]
y = mag
for i in range(len(mag)):
    if(mag[i]<9.5):
        labels[i] = "9"
    if (mag[i] < 8.5):
        labels[i] = "8"
    if (mag[i] < 7.5):
        labels[i] = "7"
    if (mag[i] < 6.5):
        labels[i] = "6"
    if (mag[i] < 5.5):
        labels[i] = "5"
    if (mag[i] < 4.5):
        labels[i] = "4"
    if (mag[i] < 3.5):
        labels[i] = "3"
    if (mag[i] < 2.5):
        labels[i] = "2"
    if (mag[i] < 1.5):
        labels[i] = "1"
    if (mag[i] < 0.5):
        labels[i] = "0"
p = np.linspace(0.01,0.99,99)
rms = np.zeros(99)
count = 0

prob = 0.8
rmsNodes = np.zeros(10)
# Vary the number of nodes used in each layer
for k in range(10):
    mask = np.random.rand(9766) < prob

    trainingX = data[mask,:NUM_PARAMS:]
    testX = data[~mask,:NUM_PARAMS:]

    trainMag = mag[mask]
    trainY = labels[mask]
    testMag = mag[~mask]
    testY = labels[~mask]

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(trainingX)

    trainingX = scaler.transform(trainingX)
    testX = scaler.transform(testX)

    mlp = MLPClassifier(hidden_layer_sizes=((k+1)*5,(k+1)*5,(k+1)*5))

    mlp.fit(trainingX,trainY)

    predictions = mlp.predict(testX)

    y_pred = np.array(predictions).astype(dtype=float)
    print(y_pred)
    observedY = np.array(testY).astype(dtype=float)
    for i in range(len(testMag)):
        if(y_pred[i]<0):
            y_pred[i] = 0.0
        if(y_pred[i]>np.max(trainMag)):
            y_pred[i] = np.max(trainMag)
    error = testMag-y_pred
    rmsE = np.mean((np.power(error, 2)))
    plt.plot(y_pred)
    plt.show()
    print(rmsE)
    x = np.linspace(0,len(error),len(error))
    plt.scatter(x, error)
    plt.title('Percent Error of Each Data Point vs Percent Training Data')
    plt.show()
    rmsNodes[k] = rmsE
x = np.linspace(5,50,10)
plt.plot(x,rmsNodes)
plt.title('RMS Error vs Number of Nodes')
plt.show()

rmsLayers = np.zeros(5)
# No loop here, it's bad but this varies the number of layers
mask = np.random.rand(9766) < prob

trainingX = data[mask,:NUM_PARAMS:]
testX = data[~mask,:NUM_PARAMS:]

trainMag = mag[mask]
trainY = labels[mask]
testMag = mag[~mask]
testY = labels[~mask]

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(trainingX)

trainingX = scaler.transform(trainingX)
testX = scaler.transform(testX)

mlp = MLPClassifier(hidden_layer_sizes=(30))

mlp.fit(trainingX,trainY)

predictions = mlp.predict(testX)

y_pred = np.array(predictions).astype(dtype=float)
print(y_pred)
observedY = np.array(testY).astype(dtype=float)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.show()
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Percent Error of Each Data Point vs Percent Training Data')
plt.show()
rmsLayers[0] = rmsE


mask = np.random.rand(9766) < prob

trainingX = data[mask,:NUM_PARAMS:]
testX = data[~mask,:NUM_PARAMS:]

trainMag = mag[mask]
trainY = labels[mask]
testMag = mag[~mask]
testY = labels[~mask]

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(trainingX)

trainingX = scaler.transform(trainingX)
testX = scaler.transform(testX)

mlp = MLPClassifier(hidden_layer_sizes=(30,30))

mlp.fit(trainingX,trainY)

predictions = mlp.predict(testX)

y_pred = np.array(predictions).astype(dtype=float)
print(y_pred)
observedY = np.array(testY).astype(dtype=float)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.show()
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Percent Error of Each Data Point vs Percent Training Data')
plt.show()
rmsLayers[1] = rmsE

mask = np.random.rand(9766) < prob

trainingX = data[mask,:NUM_PARAMS:]
testX = data[~mask,:NUM_PARAMS:]

trainMag = mag[mask]
trainY = labels[mask]
testMag = mag[~mask]
testY = labels[~mask]

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(trainingX)

trainingX = scaler.transform(trainingX)
testX = scaler.transform(testX)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

mlp.fit(trainingX,trainY)

predictions = mlp.predict(testX)

y_pred = np.array(predictions).astype(dtype=float)
print(y_pred)
observedY = np.array(testY).astype(dtype=float)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.show()
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Percent Error of Each Data Point vs Percent Training Data')
plt.show()
rmsLayers[2] = rmsE

mask = np.random.rand(9766) < prob

trainingX = data[mask,:NUM_PARAMS:]
testX = data[~mask,:NUM_PARAMS:]

trainMag = mag[mask]
trainY = labels[mask]
testMag = mag[~mask]
testY = labels[~mask]

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(trainingX)

trainingX = scaler.transform(trainingX)
testX = scaler.transform(testX)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30,30))

mlp.fit(trainingX,trainY)

predictions = mlp.predict(testX)

y_pred = np.array(predictions).astype(dtype=float)
print(y_pred)
observedY = np.array(testY).astype(dtype=float)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.show()
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Percent Error of Each Data Point vs Percent Training Data')
plt.show()
rmsLayers[3] = rmsE

mask = np.random.rand(9766) < prob

trainingX = data[mask,:NUM_PARAMS:]
testX = data[~mask,:NUM_PARAMS:]

trainMag = mag[mask]
trainY = labels[mask]
testMag = mag[~mask]
testY = labels[~mask]

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(trainingX)

trainingX = scaler.transform(trainingX)
testX = scaler.transform(testX)
start = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30,30,30))

mlp.fit(trainingX,trainY)

predictions = mlp.predict(testX)
end = time.time()
print(end - start)
y_pred = np.array(predictions).astype(dtype=float)
print(y_pred)
observedY = np.array(testY).astype(dtype=float)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.show()
print(rmsE)
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Percent Error of Each Data Point vs Percent Training Data')
plt.show()
rmsLayers[4] = rmsE

y = np.linspace(1,5,5)
plt.plot(y,rmsLayers)
plt.title('RMS Error vs Number of Layers')
plt.show()
