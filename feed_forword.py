import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, alpha = 0.4, hta_init=0.1,hta_final=0.1,epochs =100):
        self.dw = np.zeros((input_size+ 1, hidden_size)) #TODO check the dimensions of matrix
        self.dv = np.zeros((hidden_size+1,output_size))
        self.w = np.random.normal(0,0.5,(input_size+ 1, hidden_size))
        self.v = np.random.normal(0,0.5,(hidden_size+1,output_size))
        self.hidden_size = hidden_size
        self.hta = hta_init
        self.hta_init = hta_init
        self.hta_final = hta_final
        self.epochs = epochs
        self.current_epoch = 0
        self.alpha = alpha
    def update_sceduler(self):
        self.current_epoch += 1
        self.hta = (self.hta_init-self.hta_final)*(self.epochs-self.current_epoch)/(self.epochs)+self.hta_final
        
    
    def add_bias(self, X):
        ones_column = np.ones((1,X.shape[0]))
        #print("ones_column.shape : ",ones_column.shape)
        #print("X : ",X.shape)
        X = np.hstack((X, ones_column.T))
        return X

    def phi(self, Χ):
        return 2/(1+np.exp(-Χ))-1
    def d_phi(self, phi):
        return (1+phi)*(1-phi)/2

    def forward(self, X):
        X  = self.add_bias(X)
        self.hin = X @ self.w ;
        self.hidden_layer_output = self.phi(self.hin)
        self.hout =  self.add_bias(self.hidden_layer_output)
        self.oin = self.hout @ self.v
        self.out = self.phi(self.oin)
        self.X = X
        return self.out

    def backward(self, T):
        delta_o = (self.out-T)*self.d_phi(self.out)

        delta_h = (delta_o@self.v.T)*self.d_phi(self.hout)
        delta_h = delta_h[:,0:self.hidden_size]

        self.dw = self.alpha*self.dw- (1-self.alpha)*(self.X.T@delta_h)
        self.dv = self.alpha*self.dv- (1-self.alpha)*(self.hout.T@delta_o)
       
        self.w = self.w + self.hta*self.dw
        self.v = self.v + self.hta*self.dv


    def train(self, epochs=10000, batch_size = 40):
        X_train = np.arange(0, 2 * np.pi, 0.1).reshape(-1,1)
        T_train = np.sin(2 * X_train).reshape(-1,1)

        # Combine the x and y values into training samples
        #training_samples = np.column_stack((X, y_values))
        #plt.figure(figsize=(8, 6))
        X_test = np.arange(0.05, 2 * np.pi, 0.1).reshape(-1,1)
        T_test = np.sin(2 * X_test).reshape(-1,1)

        
        
        
        train_error = []
        test_error = []
        train_samples, dim = X_train.shape
        train_data = X_train
        train_labels = T_train


        test_samples, dim = X_test.shape
        test_data = X_test
        test_labels = T_test

        for i in range(epochs):
            epoch_train_error = 0
            for i in range(0,train_samples,batch_size):
                #print(train_data.shape)
                #print(train_labels.shape)
                X = train_data[i:i+batch_size]
                T = train_labels[i:i+batch_size]
                Y = self.forward(X)
                T = T.reshape(-1,1)
                self.backward(T)
                Y= Y.T.flatten()

                epoch_train_error += np.sum(np.abs(Y-T.ravel()))
            epoch_test_error = 0
            for i in range(0,test_samples,batch_size):
                X = test_data[i:i+batch_size]
                T = test_labels[i:i+batch_size]
                Y = self.forward(X)
                Y= Y.T.flatten()

                epoch_test_error += np.sum(np.abs(Y-T.ravel()))
            # Compute loss
            train_error.append(epoch_train_error/train_samples)
            test_error.append(epoch_test_error/test_samples)

            print(f'Epoch {i}, train Loss: {epoch_train_error/train_samples}, test Loss: {epoch_test_error/test_samples}')
        #print("here")

        #x_min, x_max = -4, 4
        #y_min, y_max = -4, 4
        resolution = 0.01
        #plt.plot(X_train, T_train, label='True function (sin(2x))', color='blue', linewidth=2)
        #plt.title('RBF Network Approximation of sin(2x) (from scratch)')
        #plt.plot(X_test, T_test, label='shit : ', linewidth=2)
        plt.plot(X_test, self.forward(X_test), label='ff Network', linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.show()
# Example usage
if __name__ == "__main__":
    # Example dataset: XOR problem (just for illustration, you can replace it with your dataset)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
    y = np.array([0, 1, 1, 0])  # Target labels (binary classification)

    # One-hot encode the labels for softmax cross-entropy
    # y_encoded = np.eye(2)[y]

    # Initialize the network: 2 input features, 2 hidden units, 2 output units (binary classification)
    nn = NeuralNetwork(input_size=1, hidden_size=40, output_size=1, hta_init=0.01,hta_final=0.0001)

    # Train the model
    nn.train( epochs=10000)