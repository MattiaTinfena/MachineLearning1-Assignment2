import numpy as np

def oneDimLRM (dataset):

    columns = []
    columns = dataset.columns
    x = dataset[columns[0]].tolist()
    y = dataset[columns[1]].tolist()
    num = 0
    den = 0
    for i in range(len(dataset)):
        num += x[i]*y[i]
        den += x[i]*x[i]
    w = (num/den)
    return w


def oneDimLRMInt (dataset):

    columns = []
    columns = dataset.columns
    x = dataset[columns[0]].tolist()
    y = dataset[columns[1]].tolist()

    avgx = 0
    avgy = 0 

    for j in range(len(dataset)):
        avgx += x[j]
        avgy += y[j]

    avgx /= len(dataset)
    avgy /= len(dataset)

    num = 0
    den = 0
    for i in range(len(dataset)):
        num += (x[i] - avgx)*(y[i] - avgy)
        den += (x[i]-avgx)*(x[i]-avgx)
    w1 = (num/den)
    w0 = avgy - (w1*avgx)
    return w0, w1

def multiDimLRM (dataset, Win = None):

    columns = []
    columns = dataset.columns

    col1 = [1] * len(dataset)
    x = dataset[columns[0]].tolist()
    y = dataset[columns[1]].tolist()
    z = dataset[columns[2]].tolist()

    t = dataset[columns[3]].tolist()

    X = np.column_stack((col1,x,y,z))


    if Win is None:
        den = np.linalg.pinv(((X.T) @ X))
        num = (X.T) @ t

        W = den @ num
        pred = X @ W

        return W, pred
    else:
        pred = X @ Win
        return Win, pred