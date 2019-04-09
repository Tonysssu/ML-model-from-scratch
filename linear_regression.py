import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

 # a=[[1][2][3]] b=[[4][5][6]]
 # a*b -> element-wise product = [[4] [10] [18]]
 # a.transpose()*b  broadcast, element-wise product = [[ 4  8 12] [ 5 10 15] [ 6 12 18]]

 # a.dot(b) -> matrxi product
def plot_line(x, y, y_hat,line_color='blue'):
    plt.scatter(x, y,  color='black')
    plt.plot(x, y_hat, color=line_color,
             linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def linear_cost_func(X_train, Y_train, theta):
    # return np.sum(np.square(linear_val_func(theta, X_train)-Y_train))/Y_train.shape[0]
    return np.dot(linear_val_func(theta, X_train).T, Y_train)

# error.T.dot(each colum of X) -> derivite correspond to each theta
def linear_grad_func(X_train, Y_train, theta):
    return np.dot((linear_val_func(theta, X_train) - Y_train).T, np.c_[np.ones(X_train.shape[0]), X_train])/X_train.shape[0]

def linear_val_func(theta, x):
    # forwarding
    return np.dot(np.c_[np.ones(x.shape[0]), x], theta.T)

def linear_grad_desc(X_train, Y_train, theta, lr = .01, max_iter = 50000, converge_change = .001):
    cost_iter = []
    cost = linear_cost_func(X_train, Y_train, theta)
    cost_iter.append([0, cost])
    cost_change = 1
    i = 1
    while cost_change > converge_change and i < max_iter:
        pre_cost = cost
        grad = linear_grad_func(X_train, Y_train, theta)
        theta -= lr* grad

        cost = linear_cost_func(X_train, Y_train ,theta)
        cost_iter.append([i, cost])
        cost_change = np.abs(cost-pre_cost)
        i += 1
    return theta, cost_iter

def linear_regression():
    dataset = datasets.load_diabetes()

    X = dataset.data[:, 2]
    Y = dataset.target

    X_train = X[:-20, None]
    X_test = X[-20:, None]

    Y_train = Y[:-20, None]
    Y_test = Y[-20:, None]

    theta = np.random.rand(1, X_train.shape[1]+1)
    fitted_theta, cost_iter = linear_grad_desc(X_train, Y_train, theta)
    print('Coefficients: {}'.format(fitted_theta[0,-1]))
    print('Intercept: {}'.format(fitted_theta[0,-2]))
    print('MSE: {}'.format(np.sum((linear_val_func(fitted_theta, X_test) - Y_test)**2) / Y_test.shape[0]))

    plot_line(X_test, Y_test, linear_val_func(fitted_theta, X_test))

def sklearn_linear_regression():
    dataset = datasets.load_diabetes()
    X = dataset.data[:, 2]
    Y = dataset.target

    X_train = X[:-20, None]
    X_test = X[-20:, None]

    Y_train = Y[:-20, None]
    Y_test = Y[-20:, None]

    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, Y_train)
    print('Coefficients: {}'.format(regressor.coef_))
    print('Intercept: {}'.format(regressor.intercept_))
    print('MSE:{}'.format(np.mean((regressor.predict(X_test) - Y_test) ** 2)))

    plot_line(X_test, Y_test, regressor.predict(X_test),line_color='red')


def main():
    print('My Linear Regression')
    linear_regression()

    print ('')

    print('sklearn Linear Regression')
    sklearn_linear_regression()


if __name__ == "__main__":
    main()
