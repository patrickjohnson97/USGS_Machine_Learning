import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time

# Nearest neighbors

NUM_PARAMS = 3

data = np.genfromtxt('all_month.csv',delimiter=',')
latitude = data[:,0]
longitude = data[:,1]
depth = data[:,2]
mag = data[:,3]
X = data[:,:NUM_PARAMS:]
y = mag
p = np.linspace(0.01,0.99,99)
totalRMSE = np.zeros(99)
# vary the percent training data and average the rms
for j in range(10):
    rms = np.zeros(99)
    count = 0
    for i in p:
        mask = np.random.rand(9766) < i

        trainingX = data[mask,:NUM_PARAMS:]
        testX = data[~mask,:NUM_PARAMS:]

        trainMag = mag[mask]
        testMag = mag[~mask]

        tree = cKDTree(trainingX)

        estimatedMag = np.zeros(len(testMag))
        for x in range(0, estimatedMag.size):
            # query the corresponding row in testX matrix, get the second column (location of the point of interest
            # in the trainingX matrix),
            # and then check the Y state of the point
            estimatedMag[x] = trainMag[tree.query(testX[x, :])[1]]

        error = testMag-estimatedMag
        np.set_printoptions(threshold=np.inf)



        error = error[~np.isnan(error)]
        # plt.plot(error)
        # plt.title('Difference Between Estimated and Actual Magnitude')
        # plt.legend(['Error'])
        # plt.show()

        rms[count] = np.mean((np.power(error,2)))
        count += 1
        # for i in range(len(testMag)):
        #     if(error[i]<-100):
        #         print(testMag[i])
        #         print(estimatedMag[i])
        #         print(i)
    totalRMSE = np.add(totalRMSE,rms)
totalRMSE /= 10.0
plt.plot(totalRMSE)
plt.title('Average RMS Error vs Percent Training Data')
plt.legend(['RMS error'])
plt.show()

prob = p[np.argmin(totalRMSE)]
print(prob)
print(np.min(totalRMSE))
NEAREST_NEIGHBORS = 100
smoothRMS = np.zeros(NEAREST_NEIGHBORS)
# Vary the number of nearest neighbors used to sample
for j in range(10):
    rms2 = np.zeros(NEAREST_NEIGHBORS)
    count = 0
    for i in range(1, NEAREST_NEIGHBORS + 1):
        prob = 0.8
        mask = np.random.rand(9766) < prob

        trainingX = data[mask, :NUM_PARAMS:]
        testX = data[~mask, :NUM_PARAMS:]

        trainMag = mag[mask]
        testMag = mag[~mask]

        tree = cKDTree(trainingX)

        estimatedMag = np.zeros(len(testMag))
        for x in range(0, estimatedMag.size):
            # query the corresponding row in testX matrix, get the second column (location of the point of interest
            # in the trainingX matrix),
            # and then check the Y state of the point
            estimatedMag[x] = np.sum(trainMag[tree.query(testX[x, :], i)[1]]) / i

        error = testMag - estimatedMag
        np.set_printoptions(threshold=np.inf)

        error = error[~np.isnan(error)]
        # plt.plot(error)
        # plt.title('Difference Between Estimated and Actual Magnitude')
        # plt.legend(['Error'])
        # plt.show()
        rms2[count] = np.mean((np.power(error, 2)))
        count += 1
    smoothRMS = np.add(smoothRMS, rms2)

smoothRMS = smoothRMS/10
plt.plot(smoothRMS)
plt.title('Average RMS Error vs Number of Nearest Neighbors')
plt.legend(['RMS error'])
plt.show()
print(np.argmin(smoothRMS)+1)
print(np.min(smoothRMS))


prob = 0.8
mask = np.random.rand(9766) < prob

trainingX = data[mask, :NUM_PARAMS:]
testX = data[~mask, :NUM_PARAMS:]

trainMag = mag[mask]
testMag = mag[~mask]

tree = cKDTree(trainingX)

estimatedMag = np.zeros(len(testMag))
start = time.time()
for x in range(0, estimatedMag.size):
    # query the corresponding row in testX matrix, get the second column (location of the point of interest
    # in the trainingX matrix),
    # and then check the Y state of the point
    estimatedMag[x] = np.sum(trainMag[tree.query(testX[x, :], 7)[1]]) / 7
end = time.time()
print(end-start)
error = testMag - estimatedMag
np.set_printoptions(threshold=np.inf)

error = error[~np.isnan(error)]
# plt.plot(error)
# plt.title('Difference Between Estimated and Actual Magnitude')
# plt.legend(['Error'])
# plt.show()
err = np.mean((np.power(error, 2)))
print(err)
