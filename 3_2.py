import numpy as np
import matplotlib.pyplot as plt

class RBFNetwork:
    def __init__(self, num_centers, X, sigma=.1, hta_init = 0.01, hta_final = 0.0001, epochs = 1000):
        self.current_epoch = 0
        self.hta_final = hta_final
        self.epochs = epochs
        self.hta_init = hta_init
        self.num_centers = num_centers
        self.sigma = sigma
        self.hta = hta_init
        self.W = np.random.normal(0,0.5,self.num_centers)
        random_indices = np.random.choice(X.shape[0], self.num_centers)
        self.C = X[random_indices]
    def update_sceduler(self):
        self.current_epoch += 1
        self.hta = (self.hta_init-self.hta_final)*(self.epochs-self.current_epoch)/(self.epochs)+self.hta_final

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

epochs = 1000
X = np.arange(0, 2 * np.pi, 0.1).reshape(-1,1)

T = np.sin(2 * X).reshape(-1,1)
T += np.random.normal(0, 0.1, T.shape)

# Combine the x and y values into training samples
#training_samples = np.column_stack((X, y_values))
#plt.figure(figsize=(8, 6))
X_test = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1,1)


T_test = np.sin(2 * X_test).reshape(-1,1)
T_test += np.random.normal(0, 0.1, T_test.shape)

plt.plot(X, T, label='True function (sin(2x))', color='blue', linewidth=2)
plt.title('RBF Network Approximation of sin(2x) (from scratch)')
plt.plot(X_test, T_test, label='shit : ', linewidth=2)

plt.xlabel('x')
plt.ylabel('sin(2x)')
hidden = np.arange(1,31,10)

for hidden_size in hidden:
    nn = RBFNetwork(hidden_size,X)
    # Plot the results
    for i in range(epochs):
        Y = nn.forward(X)
        nn.backward(X,T)
        Y= Y.T.flatten()
        T = T.flatten() 
        nn.update_sceduler()  
        
    #plt.clf()
    Y = nn.forward(X).flatten()
    Y_test =  nn.forward(X_test)
    plt.plot(X_test, Y_test.flatten(), label='RBF Network Approximation with hidden size : '+str(hidden_size), linewidth=2)

    plt.legend()
    plt.grid(True)
    
    plt.draw()
    plt.pause(0.00000001)
    plt.show(block=False)
    train_error = np.sum(np.abs(Y-T))/len(Y)
    #Y_test = nn.forward(X_test)
    T_test = T_test.reshape(63,)
    #print("Y_test : ",Y_test.flatten()[:10])
    test_error = np.mean(np.abs(Y_test-T_test))
    print("hidden size : ",hidden_size,"  train error : ",train_error)
    print("hidden size : ",hidden_size,"  test error : ",test_error)
plt.pause(100)