import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Gaussian Process Regression

NUM_PARAMS = 3

data = np.genfromtxt('all_month.csv',delimiter=',')
data = data[::5,:]
latitude = data[:,0]
longitude = data[:,1]
depth = data[:,2]
mag = data[:,3]
X = data[:,:NUM_PARAMS:]
y = mag
p = np.linspace(0.01,0.99,99)
rms = np.zeros(99)
count = 0
sigma = np.linspace(0.5,10,100)
rmsSigma = np.zeros(len(sigma))
# Vary the standard deviation
for j in sigma:
    rmsE = 0
    error = 0
    prob = 0.8
    mask = np.random.rand(len(data)) < prob

    trainingX = data[mask,:NUM_PARAMS:]
    testX = data[~mask,:NUM_PARAMS:]

    trainMag = mag[mask]
    testMag = mag[~mask]

    kernel = RBF(j, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(trainingX, trainMag)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, s = gp.predict(testX, return_std=True)

    np.set_printoptions(threshold=np.inf)

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
    print(j)
    print(rmsE)
    x = np.linspace(0,len(error),len(error))
    plt.scatter(x, error)
    plt.title('Difference Between Observed and Expected Magnitude')
    plt.show()
    rmsSigma[count] = rmsE
    count += 1
print(sigma)
print(rmsSigma)
print(sigma[np.argmin(rmsSigma)])
print(np.min(rmsSigma))
plt.plot(sigma, rmsSigma)
plt.title('RMS Error vs Standard Deviation')
plt.show()

rmsRestart = np.zeros(10)
# Vary the number of restarts
for k in range(10):
    rmsE = 0
    error = 0
    prob = 0.8
    sig = 1.3
    mask = np.random.rand(len(data)) < prob

    trainingX = data[mask,:NUM_PARAMS:]
    testX = data[~mask,:NUM_PARAMS:]

    trainMag = mag[mask]
    testMag = mag[~mask]

    kernel = RBF(sig, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=k)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(trainingX, trainMag)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, s = gp.predict(testX, return_std=True)

    np.set_printoptions(threshold=np.inf)

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
    print(k)
    print(rmsE)
    x = np.linspace(0,len(error),len(error))
    plt.scatter(x, error)
    plt.title('Difference Between Observed and Expected Magnitude')
    plt.show()
    rmsRestart[k] = rmsE

plt.plot(rmsRestart)
plt.title('RMS Error vs Number of Random Restarts')
plt.show()
