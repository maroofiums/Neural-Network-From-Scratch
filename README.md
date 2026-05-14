# Neural Network From Scratch (XOR Problem)

This project implements a simple **feedforward neural network from scratch using NumPy** to solve the classic **XOR problem**.

It demonstrates:

- Forward propagation  
- Sigmoid activation  
- Backpropagation  
- Gradient descent  
- Loss calculation  
- Binary prediction  
- Decision boundary visualization using Matplotlib  

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

XOR is **not linearly separable**, meaning a single-layer perceptron cannot solve it.

A hidden layer enables the network to learn nonlinear decision boundaries.

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

- Custom neural network implementation from scratch  
- Random weight initialization  
- Forward propagation  
- Backpropagation  
- Mean Squared Error loss  
- Gradient descent optimization  
- XOR prediction  
- Decision boundary visualization  

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

---

## Loss Function

This project uses Mean Squared Error:

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

Weight update rule:

$$
W = W - \eta \nabla W
$$

Where:

- $\eta$ = learning rate  
- $\nabla W$ = gradient  

---

## Decision Boundary Visualization

This project now visualizes how the neural network separates XOR classes using **Matplotlib**.

After training:

```python
nn.plot_decision_boundary(X, y)
```

This generates a plot showing:

- learned classification regions  
- XOR data points  
- nonlinear separation boundary  

This helps visualize why hidden layers are necessary for solving XOR.

---

## Training

Train the network:

```python
nn.train(X, y, epochs=100000)
```

Example output:

```text
Epoch 0, Loss: 0.30
Epoch 1000, Loss: 0.24
Epoch 2000, Loss: 0.18
...
```

---

## Prediction

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

## Installation

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Run Project

```bash
python main.py
```

---

## Output Visualization

![Decision Boundary](https://github.com/maroofiums/Neural-Network-From-Scratch/blob/main/images/decision-boundary.png)

--

## Learning Outcomes

This project helps understand:

- Matrix operations in neural networks  
- Activation functions  
- Gradient descent  
- Backpropagation  
- Nonlinear classification  
- Decision boundaries  
- Why XOR is a foundational ML problem  

---

## Future Improvements

- Replace MSE with Binary Cross Entropy  
- Add ReLU activation  
- Add multiple hidden layers  
- Implement mini-batch gradient descent  
- Add training loss graph  
- Rebuild using PyTorch/TensorFlow  

---

## Why Build This?

Libraries like PyTorch and TensorFlow abstract away the math.

Building neural networks from scratch helps develop deeper intuition for:

> forward propagation, gradients, optimization, and nonlinear learning

This foundation is extremely valuable for machine learning engineering and deep learning research.
