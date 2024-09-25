import numpy as np
import math
import feed_forword

def get_cities_data(fname):
    x = []
    y = []
    with open(fname, 'r') as f:
        possitions = f.read().replace(";", "").splitlines()
        print(possitions)
        for possition in possitions:
            print(possition.split(','))

            x_pos, y_pos = possition.split(',')
            print(x_pos, y_pos)
            x.append(float(x_pos))
            y.append(float(y_pos))
            #x.append(float(possitions[i]))
            #y.append(float(possitions[i+1]))

        print(x)
        print(y)
        city_positions_2d = np.column_stack((x, y))

        print(city_positions_2d)
        return city_positions_2d
            #return np.asarray(f.read().replace("\n", "").split(',')).reshape((32, 84)).astype(int)
            #return np.asarray(f.read().replace("\n", "").split(',')).reshape((32, 84))#.astype(int)


def get_animal_names(fname):
    with open(fname, 'r') as f:
        return f.read().replace(";", "").split()

import numpy as np
import matplotlib.pyplot as plt

class RBFNetwork:

    def __init__(self, num_centers, sigma=.7, hta = 0.4):
        self.num_centers = num_centers
        self.sigma = sigma
        self.hta = hta
        #self.W = np.random.normal(0,0.5,self.num_centers)
        self.C = np.random.rand(10,2)

    def least_squares(self, X, Y):
            phi = self.phi(X)
            print((phi.T@phi).shape)
            self.W = (np.linalg.pinv(phi.T@phi)@phi.T)@Y
            
    def update_center(self, sample,i,epochs):
        k_neighbors = max(int(5*(1-i/epochs)),1)
        self.hta = self.hta*(1-i/epochs)
        distances = np.sum((self.C-sample)**2,axis=1)
        winner = np.argmin(distances)
        for i in range(winner-int(k_neighbors/2),winner+int(k_neighbors/2)):
            index = i
            if index < 0:
                index = self.C.shape[0] - index
            if index >= self.C.shape[0]:
                print("here")
                index = index % self.C.shape[0]
            print(self.C.shape[0])
            print(index)
            print("__________________________")
            self.C[index] += self.hta*(sample-self.C[index])

        

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

epochs = 500

nn = RBFNetwork(10)
# Plot the results
plt.figure(figsize=(8, 6))

data = get_cities_data('data/cities.dat')
#names = get_animal_names('data/cities.txt')
#print(len(names))
#print(names)
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
print(data)
plt.plot(data[:,0], data[:,1], 'bo-', label='Cities')  # 'bo-' for blue circles connected by lines
plt.show()
print(res)
points = []
for i in range(data.shape[0]):
    points.append((data[i,0],data[i,1]))
print(points)
Z = [x for _,x in sorted(zip(res,points))]

print(np.transpose(Z)[0])
plt.plot(np.transpose(Z)[0], np.transpose(Z)[1], 'bo-', label='Cities')  # 'bo-' for blue circles connected by lines
plt.show()