import numpy as np 
import matplotlib.pyplot as plt 

def square(x):
    f = []
    y = np.sin(x)
    for ys in y:
        if ys >= 0:
            f.append(1)
        else:
            f.append(-1)
    return f

def sine(x):
    y = np.sin(x)
    return y

def RBF(x, mean, sigma):
    kernel = np.exp((-(x-mean)**2)/(2*(sigma**2)))
    return kernel

def P(x,hidden, sigma, mean):
    Phi = np.zeros((x.shape[0], mean.shape[0]))
    for i, xs in enumerate(x):
        for j, mu in enumerate(mean):
            Phi[i][j] = RBF(xs, mu, sigma)
    return Phi


def leastSquares(Phi, y):
    W = np.linalg.pinv((Phi.T).dot(Phi)).dot(Phi.T).dot(y)
    return W

def delta(x,y,learning,epoch, mean,hidden):
    W = np.random.randn(h, xTrain.shape[1])

    for i in range(epoch):
        for j,xs in enumerate(x):
            Phi = P(xs,hidden, sigma, mean)
            W += learning * (y[j] - (Phi).dot(W)) * (Phi.T)
    return W

xTrain = np.arange(0,2*np.pi,0.1).reshape(-1,1)
xTest = np.arange(0.05,2*np.pi, 0.1).reshape(-1,1)

yTrain = np.sin(2*xTrain)
yTest = np.sin(2*xTest)

#yTrain = square(2*xTrain)
#yTest = square(2*xTest)

hidden = np.arange(1,102,25)
#h = 100
sigma = 0.1

e = []

#Training
for h in hidden:
    print("Hidden units: ",h)

    arg = np.random.choice(len(xTrain),h)
    mean = xTrain[arg]

    #Phi = P(xTrain, h, sigma, mean)
    W = delta(xTrain,yTrain, 0.01, 1000, mean, h)
    #W = leastSquares(Phi,yTrain)
    #Prediction
    Phi2 = P(xTest, h, sigma, mean)
    y = Phi2.dot(W)

    error = np.sum(np.abs(y - yTest))/len(y)
    e.append(error)
    plt.plot(xTest, y, label = '%i units'%h)
    print(e)


plt.plot(xTest, yTrain,'b',label = 'Real')      
plt.legend()
plt.title("RBF")
plt.show()
plt.plot(hidden,e,label='Error')
plt.plot(hidden, 0.1*np.ones(len(e)),'--', label='0.1 threshold')
plt.plot(hidden, 0.01*np.ones(len(e)),'--', label='0.01 threshold')
plt.plot(hidden, 0.001*np.ones(len(e)),'--', label='0.001 threshold')
plt.xlabel('Hidden Nodes')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()