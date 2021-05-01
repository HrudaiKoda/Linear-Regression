import csv
import numpy as np
import matplotlib.pyplot as plt
def load_dataset():
    dataset = list()
    with open('data.txt') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset.astype("float")
    return dataset

def normalise(x):
    mean = np.mean(x , axis=0)
    std = np.std(x , axis = 0)
    h = (x - mean)/std
    l = np.ones((x.shape[0],1))
    h= np.concatenate((l, h),axis=1)
    return h
def cost(x,y,t):
    #y = y.reshape(y.shape[0],1)
    diff = x.dot(t)-y
    p = diff.T.dot(diff)/(2*x.shape[0])
    return p
def train(x,y):
    theta = np.zeros((3,1))
    alpha = 0.1
    epochs =400
    m = x.shape[0]
    loss = list()
    y = y.reshape(y.shape[0],1)
    for i in range(epochs):
        loss.append(cost(x,y,theta))
        #print(theta.shape)
        #print(x.dot(theta)-y)
        dif = x.dot(theta)-y
        k = dif.T.dot(x)*alpha/x.shape[0]
        theta = theta - k.reshape(3,1)
    loss =np.array(loss)
    loss = loss.reshape(loss.shape[0])
    print(theta)
    #print(loss.shape)
    #plt.plot(loss)
    #plt.show()
    return theta
m = load_dataset()
nor = normalise(m[:,0:2])
lab = m[:,-1]
the = train(nor,lab)
act = m[:,-1]
pre = nor.dot(the)
y = np.arange(1,48)
org = plt.scatter(y, act ,marker="x", color="red" )
pred = plt.scatter(y,pre , marker="o",color="blue")
plt.legend((org,pred),("Original","Prediction"))
plt.show()
plt.savefig("Multi.png")
