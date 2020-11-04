"""
Feedforward model construct
    number of hidden layers:5

    neural units of hidden layers: [2000, 1000, 800, 500, 100]
    activation function: elu
"""

import torch as tch

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
