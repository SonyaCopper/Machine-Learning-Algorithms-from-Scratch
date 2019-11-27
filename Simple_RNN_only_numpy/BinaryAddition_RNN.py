import numpy as np
import matplotlib.pyplot as plt

""" Addition of two binary numbers using simple RNN"""

num_len = 8  # length of the vector that represent the binary number
i_dim = 2  # input dimension, we add two numbers
w1_dim = 16  # hidden layer dimension
out_dim = 1  # output (one digit)

max_number = (2 ** num_len) / 2  # Largest number to be used as input, to insure output of the same size after addition
iterations = 10000
learning_rate = 0.5

w1_t0 = np.random.normal(0, 1, (i_dim, w1_dim))  # first layer weights
w2_t0 = np.random.normal(0, 1, (w1_dim, out_dim))  # second layer weights
u_t0 = np.random.normal(0, 1, (w1_dim, w1_dim))  # Memory layer weights


def sigmoid(z):  # Sigmoid activation function, we have binary output
    f = 1. / (1 + np.exp(-z))
    return f


def div_sigmoid(z):  # Sigmoid derivative
    f = 1. / (1 + np.exp(-z))
    df = f * (1 - f)
    return df


def get_number(digits):  # Convert the binary number to decimal
    sum = 0
    for i in range(len(digits)):
        sum += digits[i] * (2 ** i)
    return np.array([sum])


def rnn(iterations, w1, w2, u):
    loss_vector = np.zeros(iterations)
    for t in range(iterations):  # Creating two random binary numbers for each iteration
        a = np.random.randint(max_number, size=1, dtype=np.uint8)
        n_1 = np.unpackbits(a)[::-1]
        b = np.random.randint(max_number, size=1, dtype=np.uint8)
        n_2 = np.unpackbits(b)[::-1]
        answer = a + b  # Ground truth
        target = np.unpackbits(answer)[::-1]  # Binary ground truth
        memory = [np.zeros((1, w1_dim))]
        number = []
        delta_h2 = []
        grad_w2 = np.zeros_like(w2)
        grad_w1 = np.zeros_like(w1)
        grad_u = np.zeros_like(u)
        loss = 0

        for i in range(num_len):  # Forward propagation
            input_layer = np.array([[n_1[i], n_2[i]]])
            h1 = sigmoid(input_layer.dot(w1) + memory[i].dot(u))
            memory.append(h1)
            h2 = sigmoid(h1.dot(w2))
            number.append(int(np.around(h2)))
            delta_h2.append((div_sigmoid(h1.dot(w2)) * (target[i] - h2)))
            loss += 0.5 * ((target[i] - h2) ** 2)
        future_delta_h1 = np.zeros(w1_dim)

        for j in range(num_len, 0, -1):  # Back propagation
            delta_h1 = (future_delta_h1.dot(u.T) + delta_h2[j - 1].dot(w2.T)) * (memory[j] * (1 - memory[j]))
            grad_w2 += delta_h2[j - 1].dot(memory[j]).T
            grad_u += memory[j - 1].T.dot(delta_h1)
            grad_w1 += (np.array([[n_1[j - 1], n_2[j - 1]]])).T.dot(delta_h1)
            future_delta_h1 = delta_h1

        w1 += learning_rate * grad_w1
        u += learning_rate * grad_u
        w2 += learning_rate * grad_w2
        y = np.ravel(number)
        num = get_number(y)
        print(f"Predicted number is {num} \nThe real number is {answer}")
        loss_vector[t] = loss
    return loss_vector


total_loss = rnn(iterations, w1_t0, w2_t0, u_t0)

x = np.array(range(iterations))
plt.scatter(x, total_loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
