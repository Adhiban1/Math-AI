import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Matrices
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype='float32')

W = np.random.randn(3, 1)
previous_W = W.copy()

Y = np.array([[14], [32], [50]])

# Loss function [MSE]
def loss(Y, A, W):
    return ((Y - A @ W)**2).mean()

# Grad
def dW(Y, A, W, h):
    dw = []
    for i in range(len(W)):
        w_plus = W.copy()
        w_minus = W.copy()

        w_plus[i][0] += h
        w_minus[i][0] -= h
        dw.append((loss(Y, A, w_plus) - loss(Y, A, w_minus)) / 2*h)
    return np.array(dw).reshape(-1, 1)


print('Loss:', loss(Y, A, W))
losses = [loss(Y, A, W)]

# iteration
for i in range(1000):

    W = W - dW(Y, A, W, 0.1) # grad

    if i > 995 or i < 5:
        print(f'Loss ({i+1}):', loss(Y, A, W))
    elif i in [6,7,8]:
        print('            .')
    losses.append(loss(Y, A, W))

# Output
print(f'W: {previous_W.flatten()} -> {W.flatten()}')
print(f'Y: {Y.flatten()} ~= {(A @ W).flatten()}')

# Graph
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('graph.png')
plt.close()