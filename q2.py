import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        #Update parameters using GD with momentum and return

        
        self.vel = -self.lr*grad + self.beta * self.vel
        params += self.vel
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        sum = 0
        for i in range(len(X)):
            sum = sum + max((1-y[i]*np.dot(self.w,X[i])),0)
        ## Implement hinge loss
        return sum/len(X)
       

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        sum_term = np.zeros(X.shape[1])
        for i in range(len(X)):
            if y[i]*np.dot(self.w,X[i]) >= 1:
                add_term = 0
            else:
                add_term = (-y[i]*X[i])
            sum_term = sum_term + add_term
        
        w_panal = np.append(0,self.w[1:])
        grad = w_panal + self.c/len(X)*sum_term 
        return grad
        

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        y_hat = np.dot(X,self.w)
        return np.sign(y_hat)

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w,func_grad(w))
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    batch_generator = BatchSampler(train_data,train_targets,batchsize)
    svm = SVM(penalty,train_data.shape[1])

    for _ in range(iters):
        X_batch, y_batch = batch_generator.get_batch(batchsize)
        svm.w = optimizer.update_params(svm.w, svm.grad(X_batch,y_batch))
    return svm
        


if __name__ == '__main__':
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.insert(train_data, 0, values = np.ones(len(train_data)), axis=1)
    test_data = np.insert(test_data, 0, values = np.ones(len(test_data)), axis=1)
    gd1 = GDOptimizer(0.05, 0.0)
    model1 = optimize_svm(train_data, train_targets, penalty = 1.0, optimizer = gd1, batchsize = 100, iters = 500)
    gd2 = GDOptimizer(0.05, 0.1)
    model2 = optimize_svm(train_data, train_targets, penalty = 1.0, optimizer = gd2, batchsize = 100, iters = 500)
    hinge_loss_train1 = model1. hinge_loss(train_data, train_targets)
    print("The average hinge loss of train data with momentum of 0 is", hinge_loss_train1)
    hinge_loss_train2 = model2. hinge_loss(train_data, train_targets)
    print("The average hinge loss of train data with momentum of 0.1 is", hinge_loss_train2)
    hinge_loss_test1 = model1. hinge_loss(test_data, test_targets)
    print("The average hinge loss of test data with momentum of 0 is", hinge_loss_test1)
    hinge_loss_test2 = model2. hinge_loss(test_data, test_targets)
    print("The average hinge loss of test data with momentum of 0.1 is", hinge_loss_test2)
    print()
    pred_train1 = model1.classify(train_data)
    print("The classification accuracy on the training set with momentum of 0 is", sum(pred_train1 == train_targets)/len(pred_train1))
    pred_train2 = model2.classify(train_data)
    print("The classification accuracy on the training set with momentum of 0.1 is", sum(pred_train2 == train_targets)/len(pred_train2))
    pred_test1 = model1.classify(test_data)
    print("The classification accuracy on the test set with momentum of 0 is", sum(pred_test1 == test_targets)/len(pred_test1))
    pred_test2 = model2.classify(test_data)
    print("The classification accuracy on the test set with momentum of 0.1 is", sum(pred_test2 == test_targets)/len(pred_test2))


##plots
    
    plt.figure(0)
    w3 = optimize_test_function(GDOptimizer(1))
    w4 = optimize_test_function(GDOptimizer(1,beta=0.9))
    x_axes = [i for i in range(201)]
    plt.plot(x_axes,w3,color='red',label = "beta = 0.0")
    plt.plot(x_axes,w4,color='blue',label = "beta = 0.1")
    plt.legend(loc='lower right')
    plt.show()

    

    plt.figure(1)
    w1 = model1.w
    w1 = w1[1:].reshape((28,28))
    plt.imshow(w1)
    plt.show()
    
    plt.figure(2)
    w2 = model1.w
    w2 = w2[1:].reshape((28,28))
    plt.imshow(w2)
    plt.show()
    

