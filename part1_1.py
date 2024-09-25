import numpy as np
import matplotlib.pyplot as plt
import math
class RBFNetwork:

    def __init__(self, num_centers, X, sigma=.9, hta = 0.03):
        self.num_centers = num_centers
        self.sigma = sigma
        self.hta = hta
        self.W = np.random.normal(0,0.5,self.num_centers)
        random_indices = np.random.choice(X.shape[0], self.num_centers, replace=False)
        self.C = X[random_indices]

    def least_squares(self, X, Y):
            phi = self.phi(X)
            #print(phi.shape)
            #print(np.linalg.inv(phi.T@phi.T).shape)
            #print((np.linalg.inv(phi@phi.T)@phi.T).shape)
            #print(Y.shape)
            #print((np.linalg.inv(phi@phi.T)@phi).T.shape)
            #print(self.W.shape)
            print((phi.T@phi).shape)
            self.W = (np.linalg.pinv(phi.T@phi)@phi.T)@Y
            
        

    def update_center(self, sample, neighbors = 1):
        hta = self.hta
        distances = (self.C-sample)**2
        for i in range(neighbors):
            index = np.argmin(distances)
            self.C[index] += hta*(sample-self.C[index])
            hta /= 2
            distances[index] = math.inf
        #print("\n\n____________________\n\n")

    def phi(self,X):
        phi = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, c in enumerate(self.C):
                phi[i,j] = np.exp(-(x-c)**2/(2*self.sigma**2))
        return phi
    
    
    def forward(self, X):
        #print(X.shape)
        #print(self.phi(X).shape)
        #print(self.W.shape)
        #print("________________")
        self.output = self.phi(X)@self.W
        return self.output
    def backward(self, X, y):
        #print(y.flatten().shape)
        #print((self.output).shape)
        #print((y.flatten()-self.output).reshape(1, len(y)).shape)
        #print(self.phi(X).shape)
        #print((((y.flatten()-self.output).reshape(1, len(y)))@self.phi(X)).flatten().shape)
        #print(self.W.shape)
        self.W += self.hta*(((y.flatten()-self.output).reshape(1, len(y)))@self.phi(X)).flatten()
    def predict(self, X):
        return self.phi(X).T@self.W
def square(x):
    f = []
    y = np.sin(x)
    for ys in y:
        if ys >= 0:
            f.append(1.)
        else:
            f.append(-1.)
    return f
epochs = 10000

X = np.arange(0, 2 * np.pi, 0.1).reshape(-1,1)
T = np.sin(2 * X).reshape(-1,1)  
#T += np.random.normal(0, 0.1, T.shape)
# Combine the x and y values into training samples
#training_samples = np.column_stack((X, y_values))
plt.plot(X, T, label='True function (sin(2x))', linewidth=2)
plt.title('RBF Network Approximation sin(2x)',fontsize=16)
#plt.figure(figsize=(8, 6))
X_test = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1,1)
T_test = np.sin(2 * X_test).reshape(-1,1)

plt.xlabel('x',fontsize=16)
plt.ylabel('sin(2x)',fontsize=16)
hidden = np.arange(1,51,10)

train_errors = []
test_errors = []

for hidden_size in hidden:

    nn = RBFNetwork(hidden_size,X)
    # Plot the results
    
    #T_test += np.random.normal(0, 0.1, T_test.shape)


    #for i in range(epochs):
    #    random_index = np.random.randint(0, X.shape[0])
    #    sample = X[random_index]
    #    nn.update_center(sample)

    nn.least_squares(X,T)
    #plt.clf()
    #plt.plot(X, T, label='True function (sin(2x))', color='blue', linewidth=2)
    #print("\n\n\n\n\n\n____________________________________\n\n\n\n\n\n")
    Y = nn.forward(X)
    Y_test = nn.forward(X_test)
    plt.plot(X_test, Y_test, label='centers :'+str(hidden_size), linewidth=2)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.draw()
    plt.show(block=False)
    train_error = np.mean(np.abs(Y-T))
    #Y_test = nn.forward(X_test)
    #print("T_test : ",T_test.shape)
    T_test = T_test.reshape(63,)
    Y_test = Y_test.reshape(63,)
    #print("Y_test : ",Y_test.flatten()[:10])
    #print("T_test : ",T_test.flatten()[:10])

    test_error = np.mean(np.abs(Y_test-T_test))
    print("hidden size : ",hidden_size,"  train error : ",train_error)
    print("hidden size : ",hidden_size,"  test error : ",test_error)
    train_errors.append(train_error)
    test_errors.append(test_error)
plt.savefig('./part_1_sin.pdf',dpi=200)

plt.pause(100)

plt.clf()
print(hidden)
print(train_errors)
print(test_errors)
plt.title('Error per hidden size',fontsize=16)

plt.plot(hidden, 0.1*np.ones(len(train_errors)),'--', label='0.1')
plt.plot(hidden, 0.01*np.ones(len(train_errors)),'--', label='0.01')
plt.plot(hidden, 0.001*np.ones(len(train_errors)),'--', label='0.001')
plt.plot(hidden, train_errors, label='Train error', linewidth=2)
plt.plot(hidden, test_errors, label='Test error', linewidth=2)
plt.xlabel('hidden size',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.legend()
plt.savefig('./Error_per_hidden size_sin.pdf',dpi=200)


plt.show()
