# Neural Network From Scratch (XOR Problem)

This project implements a simple **feedforward neural network from scratch using NumPy** to solve the classic **XOR problem**.

It demonstrates:

- Forward propagation  
- Sigmoid activation  
- Backpropagation  
- Gradient descent  
- Loss calculation  
- Binary prediction  

The goal is to understand how neural networks work internally before using frameworks like PyTorch or TensorFlow.

---

## Problem Statement

The XOR truth table:

| Input 1 | Input 2 | Output |
|----------|----------|----------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR is **not linearly separable**, meaning a single perceptron cannot solve it.

A hidden layer allows the network to learn nonlinear decision boundaries.

---

## Architecture

### Input Layer
- 2 neurons

### Hidden Layer
- 4 neurons
- Sigmoid activation

### Output Layer
- 1 neuron
- Sigmoid activation

Architecture:

```text
2 → 4 → 1
```

---

## Features

### Weight Initialization

Weights are initialized randomly:

```python
self.W1 = np.random.rand(input_size, hidden_size)
self.W2 = np.random.rand(hidden_size, output_size)
```

Biases are initialized as zeros.

---

## Forward Propagation

Hidden layer computation:

$$
z_1 = XW_1 + b_1
$$

$$
a_1 = \sigma(z_1)
$$

Output layer computation:

$$
z_2 = a_1W_2 + b_2
$$

$$
\hat{y} = \sigma(z_2)
$$

---

## Sigmoid Activation Function

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

This function maps outputs between:

```text
0 → 1
```

---

## Loss Function

This project uses **Mean Squared Error (MSE)**:

$$
Loss = \frac{1}{n}\sum (y-\hat{y})^2
$$

---

## Backpropagation

Output layer error:

$$
\delta_{output}=(\hat{y}-y)\cdot\sigma'(\hat{y})
$$

Hidden layer error:

$$
\delta_{hidden}=\delta_{output}W_2^T \cdot \sigma'(a_1)
$$

Weight updates:

$$
W = W - \eta \nabla W
$$

Where:

- $\eta$ = learning rate  
- $\nabla W$ = gradient  

---

## Training

Train the network using:

```python
nn.train(X, y, epochs=100000)
```

Training output example:

```text
Epoch 0, Loss: 0.30
Epoch 1000, Loss: 0.24
Epoch 2000, Loss: 0.18
...
```

---

## Prediction

After training:

```python
print(nn.predict(X))
```

Expected output:

```text
[[0]
 [1]
 [1]
 [0]]
```

---

## Project Structure

```text
neural-network/
│
├── main.py
├── README.md
└── requirements.txt
```

---

## How to Run

Install Dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

## Learning Goals

This project helps understand:

- Matrix multiplication  
- Activation functions  
- Gradient descent  
- Backpropagation  
- Loss optimization  
- Why hidden layers matter  

---

## Future Improvements

- Replace MSE with Binary Cross Entropy  
- Add ReLU activation  
- Add multiple hidden layers  
- Implement mini-batch gradient descent  
- Visualize loss curves using Matplotlib  
- Build the same model using PyTorch for comparison  

---

## Why Build This?

Libraries like TensorFlow and PyTorch abstract away the math.

Building a neural network from scratch helps you truly understand:

> how forward propagation, backpropagation, and gradient updates work under the hood.

That foundation becomes extremely valuable when moving into deep learning, research, or building custom architectures.