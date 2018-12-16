from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import time
# Linear (Polynomial) Regression
NUM_PARAMS = 3

data = np.genfromtxt('all_month.csv',delimiter=',')
latitude = data[:,0]
longitude = data[:,1]
depth = data[:,2]
mag = data[:,3]
X = data[:,:NUM_PARAMS:]
y = mag
p = np.linspace(0.01,0.99,99)
rms = np.zeros(99)
count = 0

prob = 0.8
mask = np.random.rand(9766) < prob

trainingX = data[mask,:NUM_PARAMS:]
testX = data[~mask,:NUM_PARAMS:]

trainMag = mag[mask]
testMag = mag[~mask]

regressor = LinearRegression()
regressor.fit(trainingX, trainMag)
y_pred = regressor.predict(testX)

# display coefficients
print(regressor.coef_)

for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.title('Predicted Magnitude')
plt.show()
print(rmsE)
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Difference Between Observed and Expected Magnitude')
plt.show()
DEGREES = 15
rmsDegree = np.zeros(15)
# Vary the number of Degrees for polynomial
for j in range(1,DEGREES+1):
    poly = PolynomialFeatures(degree=j)
    polyTrainX = poly.fit_transform(trainingX)
    polyTestX = poly.fit_transform(testX)

    clf = LinearRegression()
    clf.fit(polyTrainX, trainMag)
    y_pred = clf.predict(polyTestX)
    print(clf.coef_)
    for i in range(len(testMag)):
        if(y_pred[i]<0):
            y_pred[i] = 0.0
        if(y_pred[i]>np.max(trainMag)):
            y_pred[i] = np.max(trainMag)
    error = testMag-y_pred
    rmsE = np.mean((np.power(error, 2)))
    plt.plot(y_pred)
    plt.title('Predicted Magnitude')
    plt.show()
    print(rmsE)
    x = np.linspace(0,len(error),len(error))
    plt.scatter(x, error)
    plt.title('Difference Between Observed and Expected Magnitude')
    plt.show()
    rmsDegree[j-1] = rmsE
    print(rmsDegree[j-1])

print
print
print(np.argmin(rmsDegree)+1)
print(np.min(rmsDegree))
x = np.linspace(1,DEGREES,DEGREES)
plt.plot(x, rmsDegree)
plt.title('RMS Error vs Degree of Polynomial Fit')
plt.show()

poly = PolynomialFeatures(degree=np.argmin(rmsDegree)+1)
polyTrainX = poly.fit_transform(trainingX)
polyTestX = poly.fit_transform(testX)

clf = LinearRegression()
clf.fit(polyTrainX, trainMag)
y_pred = clf.predict(polyTestX)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.title('Predicted Magnitude')
plt.show()
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Difference Between Observed and Expected Magnitude')
plt.show()

poly = PolynomialFeatures(degree=6)
polyTrainX = poly.fit_transform(trainingX)
polyTestX = poly.fit_transform(testX)
clf = LinearRegression()
start = time.time()
clf.fit(polyTrainX, trainMag)
y_pred = clf.predict(polyTestX)
end = time.time()
print(end - start)
for i in range(len(testMag)):
    if(y_pred[i]<0):
        y_pred[i] = 0.0
    if(y_pred[i]>np.max(trainMag)):
        y_pred[i] = np.max(trainMag)
error = testMag-y_pred
rmsE = np.mean((np.power(error, 2)))
plt.plot(y_pred)
plt.title('Predicted Magnitude')
plt.show()
x = np.linspace(0,len(error),len(error))
plt.scatter(x, error)
plt.title('Difference Between Observed and Expected Magnitude')
plt.show()