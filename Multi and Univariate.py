from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def univar():


    # Reading the data's file.
    data = pd.read_csv('data1.csv', names=['population', 'profit'])

    print(data)
    X_df = pd.DataFrame(data.population)
    y_df = pd.DataFrame(data.profit)

    # Length, or number of observations, in our data
    m = len(y_df)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_df, y_df)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
    theta = np.array([1, 1])

    iterations = 1500
    alpha = 0.02
    X_df['intercept'] = 1

    # Transform to Numpy arrays for easier matrix math and start theta at 0
    X = np.array(X_df)
    y = np.array(y_df).flatten()

    def cost_function(X, y, theta):
        """
        cost_function(X, y, theta) computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y
        """

        ## Calculate the cost with the given parameters
        J = np.sum((X.dot(theta) - y) ** 2) / 2 / m

        return J

    def gradient_descent(X, y, theta, alpha, iterations):
        """
        This performs gradient descent to learn theta
        theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
        taking iterations gradient steps with learning rate alpha
        """
        cost_history = [0] * iterations

        for iteration in range(iterations):
            hypothesis = X.dot(theta)
            loss = hypothesis - y
            gradient = X.T.dot(loss) / m
            theta = theta - alpha * gradient
            cost = cost_function(X, y, theta)
            cost_history[iteration] = cost
            print(cost)
        return theta, cost_history

    (t, c) = gradient_descent(X, y, theta, alpha, iterations)
    print(t)
    print(np.array([3.5, 1]).dot(t))
    print(np.array([7, 1]).dot(t))
    best_fit_x = np.linspace(0, 25, 20)
    best_fit_y = [t[1] + t[0] * xx for xx in best_fit_x]

    plt.figure(figsize=(10, 6))
    plt.plot(X_df.population, y_df, '.')
    plt.plot(best_fit_x, best_fit_y, '-')
    plt.axis([0, 25, -5, 25])
    plt.xlabel('Population ')
    plt.ylabel('Profit in ')
    plt.title('Profit vs. Population with Linear Regression Line')
    plt.show()
    plt.plot(c)
    plt.axis([0, 1500, 4.2, 6])
    plt.xlabel('Iterations ')
    plt.ylabel('Cost ')
    plt.title('Cost Function wrt Iterations')
    plt.show()

    xlist = np.linspace(-3.0, 3.0, 3)
    ylist = np.linspace(-3.0, 3.0, 4)
    X, Y = np.meshgrid(xlist, ylist)

    Z = np.sqrt(X ** 2 + Y ** 2)
    print(Z)
    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    Z = np.sqrt(X ** 2 + Y ** 2)
    cp = ax.contour(X, Y, Z)
    ax.clabel(cp, inline=True,
              fontsize=10)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()


def multivari():


    data = pd.read_csv('ex2data2.csv')
    data.shape
    data = normalize(data, axis=0)

    X = data[:, 0:2]
    Y = data[:, 2:]

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = X[:, 0]
    ys = X[:, 1]
    zs = Y
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('size')
    ax.set_ylabel('bedroom')
    ax.set_zlabel('price')

    plt.show()

    learning_rate = 0.03
    max_iteration = 500

    theta = np.zeros((data.shape[1], 1))

    def h(theta, X):
        tempX = np.ones((X.shape[0], X.shape[1] + 1))
        tempX[:, 1:] = X
        return np.matmul(tempX, theta)

    def loss(theta, X, Y):
        return np.average(np.square(Y - h(theta, X))) / 2

    def gradient(theta, X, Y):
        tempX = np.ones((X.shape[0], X.shape[1] + 1))
        tempX[:, 1:] = X
        d_theta = - np.average((Y - h(theta, X)) * tempX, axis=0)
        d_theta = d_theta.reshape((d_theta.shape[0], 1))
        return d_theta

    def gradient_descent(theta, X, Y, learning_rate, max_iteration, gap):
        cost = np.zeros(max_iteration)
        for i in range(max_iteration):
            d_theta = gradient(theta, X, Y)
            theta = theta - learning_rate * d_theta
            cost[i] = loss(theta, X, Y)
            if i % gap == 0:
                print('iteration : ', i, ' loss : ', loss(theta, X, Y))
        return theta, cost

    theta, cost = gradient_descent(theta, X, Y, learning_rate, max_iteration, 100)

    print('Final value of theta is ')
    print(theta)

    # plot the cost
    fig, ax = plt.subplots()
    ax.plot(np.arange(max_iteration), cost, 'r')
    ax.legend(loc='upper right',
              labels=['batch gradient descent', 'stochastic gradient descent', 'mini-batch gradient descent'])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost function wrt iterations')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = X[:, 0]
    ys = X[:, 1]
    zs = Y
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('size')
    ax.set_ylabel('bedroom')
    ax.set_zlabel('price')

    x = y = np.arange(0, 0.3, 0.05)
    xp, yp = np.meshgrid(x, y)
    z = np.array([h(theta, np.array([[x, y]]))[0, 0] for x, y in zip(np.ravel(xp), np.ravel(yp))])
    zp = z.reshape(xp.shape)

    ax.plot_surface(xp, yp, zp, alpha=0.7)

    plt.show()

#univar()

multivari()