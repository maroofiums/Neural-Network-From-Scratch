import numpy as np

class NeuralNetwork:
    def __init__(self,input_size, hidden_size, output_size, learning_rate = 0.01):
        np.random.seed(42)

        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1,hidden_size))

        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.zeros((1,output_size))

        self.learning_rate = learning_rate

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)
    
    def forward(self,X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)

        return self.output
    
    def loss(self,y, output):
        return np.mean((y - output) ** 2)
    
    def backward(self,X,y):
        m = X.shape[0]

        output_error = (self.output - y) * self.sigmoid_derivative(self.output)

        dW2 = np.dot(self.a1.T, output_error) / m
        db2 = np.sum(output_error, axis=0, keepdims=True) / m

        hidden_error = np.dot(output_error, self.W2.T) * self.sigmoid_derivative(self.a1)

        dw1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    
    def train(self,X,y, epochs):
        for epoch in range(epochs+1):
            outputs = self.forward(X)

            loss_value = self.loss(y, outputs)

            self.backward(X,y)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss_value}')

    def predict(self,X):
        outputs = self.forward(X)
        return (outputs > 0.5).astype(int)
    

if __name__ == "__main__":
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    nn = NeuralNetwork(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.1
    )

    nn.train(X, y, epochs=100000)

    print(nn.predict(X))