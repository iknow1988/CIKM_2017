from loadData import *
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def garbage():
    x_tr = np.zeros((8000,960))
    for i in range(x_train.shape[0]):
        count = 0
        for j in range(x_train.shape[1]):
            for k in range(x_train.shape[2]):
                x_tr[i,count] = x_train[i,j,k]
                count = count + 1
    x_te = np.zeros((2000,960))
    for i in range(x_test.shape[0]):
        count = 0
        for j in range(x_test.shape[1]):
            for k in range(x_test.shape[2]):
                x_te[i,count] = x_test[i,j,k]
                count = count + 1
    regr = linear_model.Ridge(alpha=0.01, normalize=True)
    regr.fit(x_tr, y_train)
    y_pred = regr.predict(x_te)
    RMSE = mean_squared_error(y_test, y_pred) ** 0.5
    print "\tTest RMSE =", RMSE
def main():
    (trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, optimizationVariableSize) = loadData(10000, 1)
    trainSize = 8000
    testSize = 5

    x_train = x_train[0:trainSize,0,:]
    x_test = x_test[0:testSize,0,:]
    y_train = y_train[0:trainSize,0]
    y_test = y_test[0:testSize,0]
    
    var = np.var(x_train,axis=0)
    
    f, axarr = plt.subplots(1,2)
    comb = np.concatenate((x_train,y_train.reshape(trainSize,1)),axis = 1)
    sorted_list = sorted(comb, key=lambda x:x[22])
    sorted_list = np.array(sorted_list)
#     sorted_list1 = normalize(sorted_list[:,0:1], axis=1, norm='l1')
    sorted_list1 = sorted_list[:,0:1]
    im = axarr[0].imshow(sorted_list1.reshape(trainSize,1), aspect='auto', interpolation='none', origin='lower')
    f.colorbar(im, ax=axarr[0])
    im = axarr[1].imshow(sorted_list[:,22].reshape(trainSize,1), aspect='auto', interpolation='none', origin='lower')
    f.colorbar(im, ax=axarr[1])
    plt.show()
    
    a = Variable(optimizationVariableSize, 1)
    g = 0
    for i in range(trainSize):
        g = g + (x_train[i, 0:optimizationVariableSize - 1] * a[0:optimizationVariableSize - 1] + a[optimizationVariableSize - 1] - y_train[i]) ** 2
    
    for i in range(optimizationVariableSize - 1):
        g = g + 0.5 * (square(a[i]))
    objective = Minimize(g)
    constraints = []
    p = Problem(objective, constraints)
    result = p.solve(verbose= True)
    print "status:",p._status
    print "optimal value", p.value
    modelParameters = np.array(a.value).tolist()
    modelParameters = np.array(modelParameters).transpose()[0]
    
    y_pred = np.zeros((y_train.shape[0]))
    for i in range(y_train.shape[0]):
        t = 0
        for k in range(optimizationVariableSize-1):
            t = t + x_train[i, k] * modelParameters[k]
        y_pred[i] = t + modelParameters[optimizationVariableSize-1]
    train_error = mean_squared_error(y_train, y_pred) ** 0.5
#     print "train"
#     for i in range(y_train.shape[0]):
#         print y_pred[i], y_train[i]
        
    y_pred = np.zeros((y_test.shape[0]))
    for i in range(y_test.shape[0]):
        t = 0
        for k in range(optimizationVariableSize-1):
            t = t + x_test[i, k] * modelParameters[k]
        y_pred[i] = t + modelParameters[optimizationVariableSize-1]
    test_error = mean_squared_error(y_test, y_pred) ** 0.5
    print "test"
    for i in range(y_test.shape[0]):
        print y_pred[i], y_test[i]
    print train_error, test_error
    
if __name__ == '__main__':
    main()