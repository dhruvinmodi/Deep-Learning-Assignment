import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

print('Reading MNIST dataset files...')
train = pd.read_csv('./dataset/train.csv')
trainX = train.iloc[:, 1:].copy().to_numpy()
trainY = train.iloc[:, 0].to_numpy()

trainX = trainX.reshape(60000, 28, 28)

print('Creating trainX_pool...')
trainX_pool = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []
}

for x,y in zip(trainX, trainY):
    y_n, x_n = np.nonzero(x)
    trainX_pool[y].append(x[np.min(y_n):np.max(y_n), np.min(x_n):np.max(x_n)])

print('Reading Multidigit dataset files...')
data0 = np.load('./dataset/data0.npy')
lab0 = np.load('./dataset/lab0.npy')

data1 = np.load('./dataset/data1.npy')
lab1 = np.load('./dataset/lab1.npy')

data2 = np.load('./dataset/data2.npy')
lab2 = np.load('./dataset/lab2.npy')

x = np.concatenate((data0, data1, data2), axis=0)
y = np.concatenate((lab0, lab1, lab2), axis=0)

x,y = shuffle(x,y)

x_test = x[:2000]
y_test = y[:2000]
x_train = x[2000:]
y_train = y[2000:]


def generateMultidigitDataset(sampleCount=100, x=[], y=[]):
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    for m in range(sampleCount):
                        error = True
                        while error:
                            try:
                                t1 = np.zeros((40, 168))
                                X_MIN = random.choice(range(5,15))
                                Y_MIN = random.choice(range(5,15))
                                img1 = trainX_pool[i][random.choice(range(len(trainX_pool[i])))].copy()
                                img2 = trainX_pool[j][random.choice(range(len(trainX_pool[j])))].copy()
                                img3 = trainX_pool[k][random.choice(range(len(trainX_pool[k])))].copy()
                                img4 = trainX_pool[l][random.choice(range(len(trainX_pool[l])))].copy()

                                t1[Y_MIN: Y_MIN + img1.shape[0], X_MIN: X_MIN + img1.shape[1]] = t1[Y_MIN: Y_MIN + img1.shape[0], X_MIN: X_MIN + img1.shape[1]] + img1

                                Y_MIN = random.choice(range(5,15))
                                x_min = X_MIN + img1.shape[1] + random.choice(range(30))
                                x_max = x_min + img2.shape[1]
                                t1[Y_MIN:Y_MIN + img2.shape[0], x_min: x_max] = t1[Y_MIN:Y_MIN + img2.shape[0], x_min: x_max] + img2

                                Y_MIN = random.choice(range(5,15))
                                x_min = x_max + random.choice(range(30))
                                x_max = x_min + img3.shape[1]
                                t1[Y_MIN:Y_MIN + img3.shape[0], x_min: x_max] = t1[Y_MIN:Y_MIN + img3.shape[0], x_min: x_max] + img3
                                
                                
                                Y_MIN = random.choice(range(5,15))
                                x_min = x_max + random.choice(range(30))
                                x_max = x_min + img4.shape[1]
                                t1[Y_MIN:Y_MIN + img4.shape[0], x_min: x_max] = t1[Y_MIN:Y_MIN + img4.shape[0], x_min: x_max] + img4

                                x.append(t1)
                                y.append(i+j+k+l)
                                error = False
                            except:
                                error = True


    return x,y

print('Generating Multidigit dataset...')
multiDigitTrainX, multiDigitTrainY = generateMultidigitDataset(3, [], [])

multiDigitTrainX = np.concatenate((multiDigitTrainX, x_train), axis=0)
multiDigitTrainY = np.concatenate((multiDigitTrainY, y_train), axis=0)

multiDigitTrainX = np.asarray(multiDigitTrainX).reshape(len(multiDigitTrainX), 40, 168, 1)
multiDigitTrainY = np.asarray(multiDigitTrainY).reshape(len(multiDigitTrainY), 1)

multiDigitTestX = np.asarray(x_test).reshape(x_test.shape[0], 40, 168, 1)
multiDigitTestY = np.asarray(y_test).reshape(y_test.shape[0], 1)

print('------------------')
print('multiDigitTrainX: {} - multiDigitTrainY: {}'.format(multiDigitTrainX.shape, multiDigitTrainY.shape))
print('multiDigitTestX: {} - multiDigitTestY: {}'.format(multiDigitTestX.shape, multiDigitTestY.shape))
print('------------------')

print('Saving dataset...')
with open('./customDataset/trainX.npy', 'wb') as f:
    np.save(f, multiDigitTrainX)
with open('./customDataset/trainY.npy', 'wb') as f:
    np.save(f, multiDigitTrainY)
with open('./customDataset/testX.npy', 'wb') as f:
    np.save(f, multiDigitTestX)
with open('./customDataset/testY.npy', 'wb') as f:
    np.save(f, multiDigitTestY)

print('Dataset generated successfully')

