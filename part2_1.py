import numpy as np
import math
import feed_forword

def get_animal_data(fname):
    with open(fname, 'r') as f:
        return np.asarray(f.read().replace("\n", "").split(',')).reshape((32, 84)).astype(int)
        #return np.asarray(f.read().replace("\n", "").split(',')).reshape((32, 84))#.astype(int)


def get_animal_names(fname):
    with open(fname, 'r') as f:
        return f.read().replace("'", "").split()

import numpy as np
import matplotlib.pyplot as plt

class RBFNetwork:

    def __init__(self, num_centers, sigma=1.0, hta = 0.2):
        self.num_centers = num_centers
        self.sigma = sigma
        self.hta = hta
        #self.W = np.random.normal(0,0.5,self.num_centers)
        self.C = np.random.rand(100,84)

    def least_squares(self, X, Y):
            phi = self.phi(X)
            print((phi.T@phi).shape)
            self.W = (np.linalg.pinv(phi.T@phi)@phi.T)@Y
            
    def update_center(self, sample,i,epochs):
        k_neighbors = int(70*(1-i/epochs))
        distances = np.sum((self.C-sample)**2,axis=1)
        winner = np.argmin(distances)
        for i in range(max(0,winner-int(k_neighbors/2)),min(winner+int(k_neighbors/2),self.C.shape[0])):
            self.C[i] += self.hta*(sample-self.C[i])
        

    def phi(self,X):
        phi = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, c in enumerate(self.C):
                phi[i,j] = np.exp(-(x-c)**2/(2*self.sigma**2))
        return phi
    
    
    def forward(self, X):
        self.output = self.phi(X)@self.W
        return self.output
    def backward(self, X, y):
        self.W += self.hta*(((y.flatten()-self.output).reshape(1, len(y)))@self.phi(X)).flatten()
    def predict(self, X):
        return self.phi(X).T@self.W

epochs = 20

nn = RBFNetwork(20)
# Plot the results
plt.figure(figsize=(8, 6))

data = get_animal_data('data/animals.dat')
names = get_animal_names('data/animalnames.txt')
print(len(names))
print(names)
for i in range(epochs):
    for j, sample in enumerate(data):
        #random_index = np.random.randint(0, data.shape[0])
        #sample = data[random_index]
        nn.update_center(sample,i,epochs)

print("data.shape : ",data.shape)
res = []
for i in range(data.shape[0]):
    distances = np.sum((nn.C-data[i])**2,axis=1)
    index = np.argmin(distances)
    res.append(index)

print(res)

Z = [x for _,x in sorted(zip(res,names))]

print(np.transpose(Z))