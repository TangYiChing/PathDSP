"""
Feedforward model construct
    number of hidden layers:5

    neural units of hidden layers: [2000, 1000, 800, 500, 100]
    activation function: elu
"""

import torch as tch

class BinaryClassifier(tch.nn.Module):
    """
    MLP Classifier with 2 layers
    Activate function:
    Reference:
    https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
    https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
    """
    def __init__(self, n_features):
        # call constructors from superclass
        super(BinaryClassifier, self).__init__()
        # define layers
        self.hidden1 = tch.nn.Linear(n_features, 500)
        self.hidden2 = tch.nn.Linear(500, 200)
        self.output = tch.nn.Linear(200, 1)
        # define activation
        self.act1 = tch.nn.ReLU()
        self.act2 = tch.nn.ReLU()
        self.act3 = tch.nn.Sigmoid()
        # define initial weights
        tch.nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        tch.nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        tch.nn.init.xavier_uniform_(self.output.weight)
        # define regularization
        self.dropout = tch.nn.Dropout(p=0.2)
        self.batchnorm1 = tch.nn.BatchNorm1d(500)
        self.batchnorm2 = tch.nn.BatchNorm1d(200)
        
    #def forward(self, n_inputs):
    #    x = self.relu(self.hidden1(n_inputs))
    #    x = self.batchnorm1(x)
    #    x = self.relu(self.hidden2(x))
    #    x = self.batchnorm2(x)
    #    x = self.dropout(x)
    #    x = self.output(x)
    #    return x
    def forward(self, n_inputs):
        # inputs to first hidden layer
        X = self.hidden1(n_inputs)
        X = self.act1(X)
        X = self.batchnorm1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.batchnorm2(X)
        # output layer
        X = self.dropout(X)
        X = self.output(X)
        X = self.act3(X)
        return X
        
class MultiClassClassifier(tch.nn.Module):
    """
    MLP with 2 layers
        Activate Function:
        Optimization Function:
        Weight Initialization:
        Loss Function:
    Reference:
    https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
    https://towardsdatascience.com/pytorch-vision-multiclass-image-classification-531025193aa
    """
    def __init__(self, n_features, n_classes):
        # call constructors from superclass
        super(MultiClassClassifier, self).__init__()
        # define layers
        self.hidden1 = tch.nn.Linear(n_features, int(n_features/2))
        self.output = tch.nn.Linear(int(n_features/2), n_classes)
        # define activation
        self.act1 = tch.nn.ReLU()
        # define initial weights
        tch.nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        tch.nn.init.xavier_uniform_(self.output.weight)
        # define regularization
        self.dropout = tch.nn.Dropout(p=0.2)
        self.batchnorm1 = tch.nn.BatchNorm1d(int(n_features/2))

    def forward(self, n_inputs):
        # inputs to first hidden layer
        X = self.hidden1(n_inputs)
        X = self.batchnorm1(X)
        X = self.act1(X)
        # output layer
        X = self.output(X)
        return X


class depre_MultiClassClassifier(tch.nn.Module):
    """
    MLP with 2 layers
        Activate Function:
        Optimization Function:
        Weight Initialization:
        Loss Function:
    Reference:
    https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
    https://towardsdatascience.com/pytorch-vision-multiclass-image-classification-531025193aa
    """
    def __init__(self, n_features, n_classes):
        # call constructors from superclass
        super(MultiClassClassifier, self).__init__()
        # define layers
        self.hidden1 = tch.nn.Linear(n_features, 500)
        self.hidden2 = tch.nn.Linear(500, 200)
        self.output = tch.nn.Linear(200, n_classes)
        # define activation 
        self.act1 = tch.nn.ReLU()
        self.act2 = tch.nn.ReLU()
        #self.act3 = tch.nn.Sigmoid()
        # define initial weights
        tch.nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        tch.nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        tch.nn.init.xavier_uniform_(self.output.weight)
        # define regularization
        self.dropout = tch.nn.Dropout(p=0.2)
        self.batchnorm1 = tch.nn.BatchNorm1d(500)
        self.batchnorm2 = tch.nn.BatchNorm1d(200)

    def forward(self, n_inputs):
        # inputs to first hidden layer
        X = self.hidden1(n_inputs)
        X = self.batchnorm1(X)
        X = self.act1(X) 
        # second hidden layer
        X = self.hidden2(X)
        X = self.batchnorm2(X)
        X = self.act2(X)
        X = self.dropout(X)
        # output layer
        X = self.output(X)
        #X = self.act3(X)
        return X

class Feedforward(tch.nn.Module):
    """
    Simple Neural Network with two hidden layers.
    Activate function: tanh

    """
    def __init__(self, n_inputs):
        # call constructors from superclass
        super(Feedforward, self).__init__()
        # define network layers
        self.hidden1 = tch.nn.Linear(n_inputs, 1000)
        self.hidden2 = tch.nn.Linear(1000, 800)
        self.hidden3 = tch.nn.Linear(800, 500)
        self.hidden4 = tch.nn.Linear(500, 100)
        self.output = tch.nn.Linear(100, 1)

    def forward(self, x):
        x = tch.nn.tanh(self.hidden1(x))
        x = tch.nn.tanh(self.hidden2(x))
        x = tch.nn.tanh(self.hidden3(x))
        x = tch.nn.tanh(self.hidden4(x))
        x = self.output(x)
        return x

class Feedforward_bn(tch.nn.Module):
    def __init__(self, n_inputs):
        # call constructors from superclass
        super(Feedforward_bn, self).__init__()

        # define network layers
        self.hidden1 = tch.nn.Linear(n_inputs, 1000)
        self.bn1 = tch.nn.BatchNorm1d(1000)
        self.hidden2 = tch.nn.Linear(1000, 800)
        self.bn2 = tch.nn.BatchNorm1d(800)
        self.hidden3 = tch.nn.Linear(800, 500)
        self.bn3 = tch.nn.BatchNorm1d(500)
        self.hidden4 = tch.nn.Linear(500, 100)
        self.bn4 = tch.nn.BatchNorm1d(100)
        self.output = tch.nn.Linear(100, 1)
        # xavier initialization
        tch.nn.init.xavier_uniform_(self.hidden1.weight)
        tch.nn.init.xavier_uniform_(self.hidden2.weight)
        tch.nn.init.xavier_uniform_(self.hidden3.weight)
        tch.nn.init.xavier_uniform_(self.hidden4.weight)
        tch.nn.init.xavier_uniform_(self.output.weight)
        # activate
        self.fnn = tch.nn.Sequential(self.hidden1, self.bn1, tch.nn.ELU(),
                                     self.hidden2, self.bn2, tch.nn.ELU(),
                                     self.hidden3, self.bn3, tch.nn.ELU(),
                                     self.hidden4, self.bn4, tch.nn.ELU(),
                                     self.output)
    def forward(self, x):
        return self.fnn(x)

class Feedforward_simple(tch.nn.Module):
    def __init__(self, n_inputs):
        # call constructors from superclass
        super(Feedforward_simple, self).__init__()

        # define network layers
        self.hidden1 = tch.nn.Linear(n_inputs, 800)
        self.hidden2 = tch.nn.Linear(800, 200)
        self.hidden3 = tch.nn.Linear(200, 50)
        self.output = tch.nn.Linear(50, 1)
        # xavier initialization
        tch.nn.init.xavier_uniform_(self.hidden1.weight)
        tch.nn.init.xavier_uniform_(self.hidden2.weight)
        tch.nn.init.xavier_uniform_(self.hidden3.weight)
        tch.nn.init.xavier_uniform_(self.output.weight)
        # activate
        self.fnn = tch.nn.Sequential(self.hidden1, tch.nn.SELU(),
                                     self.hidden2, tch.nn.SELU(),
                                     self.hidden3, tch.nn.SELU(),
                                     self.output)
    def forward(self, x):
        return self.fnn(x)

class Feedforward_seq2(tch.nn.Module):
    def __init__(self, n_inputs):
        # call constructors from superclass
        super(Feedforward_seq2, self).__init__()

        # define network layers
        self.hidden1 = tch.nn.Linear(n_inputs, 800)
        self.hidden2 = tch.nn.Linear(800, 400)
        self.hidden3 = tch.nn.Linear(400, 200)
        self.hidden4 = tch.nn.Linear(200, 100)
        #self.output = tch.nn.Linear(100, 1)
        self.output = tch.nn.Linear(n_inputs, 1)
        # xavier initialization
        tch.nn.init.xavier_uniform_(self.hidden1.weight)
        tch.nn.init.xavier_uniform_(self.hidden2.weight)
        tch.nn.init.xavier_uniform_(self.hidden3.weight)
        tch.nn.init.xavier_uniform_(self.hidden4.weight)
        tch.nn.init.xavier_uniform_(self.output.weight)
        # activate
        self.fnn = tch.nn.Sequential(self.hidden1, tch.nn.SELU(),
                                     self.hidden2, tch.nn.SELU(),
                                     self.hidden3, tch.nn.SELU(),
                                     self.hidden4, tch.nn.SELU(),
                                     self.output)
    def forward(self, x):
        return self.fnn(x)


class FNN(tch.nn.Module):
    def __init__(self, n_inputs):
        # call constructors from superclass
        super(FNN, self).__init__()
       
        # define network layers
        self.hidden1 = tch.nn.Linear(n_inputs, 1000)
        self.hidden2 = tch.nn.Linear(1000, 800)
        self.hidden3 = tch.nn.Linear(800, 500)
        self.hidden4 = tch.nn.Linear(500, 100)
        self.output = tch.nn.Linear(100, 1)
        # He initialization
        #tch.nn.init.kaiming_uniform_(self.hidden1.weight)
        #tch.nn.init.kaiming_uniform_(self.hidden2.weight)
        #tch.nn.init.kaiming_uniform_(self.hidden3.weight)
        #tch.nn.init.kaiming_uniform_(self.hidden4.weight)
        #tch.nn.init.kaiming_uniform_(self.output.weight)
        # dropout
        self.dropout = tch.nn.Dropout(p=0.1)
        # activate
        self.fnn = tch.nn.Sequential(self.hidden1, tch.nn.ELU(), self.dropout,
                                     self.hidden2, tch.nn.ELU(), self.dropout,
                                     self.hidden3, tch.nn.ELU(), self.dropout,
                                     self.hidden4, tch.nn.ELU(), self.dropout,
                                     self.output)
    def forward(self, x):
        return self.fnn(x)

if __name__ == "__main__":
    net = FNN(756) #Feedforward_bn(100)
    print('initiating an feed forward network....')
    print('    construct=\n    {:}'.format(net))
