import torch
import torch.nn.functional as F

class MusicClassifier(torch.nn.Module):
    def __init__(self, n_features, n_labels, num_layers=10, neuron_incr=10, 
                dropout=0.5, batchnorm=False):
        super().__init__()
        convolutional_layers = self.get_convolutional_layers()
        linear_layers = self.get_linear_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm)
        
        layers = convolutional_layers + linear_layers
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, X):
        return self.layers(X)

    def get_convolutional_layers()
    
    def get_linear_layers(self, n_features, n_labels, num_layers, neuron_incr, dropout, batchnorm):
        current_neurons = n_features
        layers = []

        for layer in range(num_layers):
            if layer <= round(num_layers/2):
                next_neurons = current_neurons + neuron_incr
            else:
                next_neurons = current_neurons - neuron_incr

            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(current_neurons))

            layers.append(torch.nn.Linear(current_neurons, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            current_neurons = next_neurons
        
        print(current_neurons)

        if batchnorm:
            layers.append(torch.nn.BatchNorm1d(current_neurons))
        layers.append(torch.nn.Linear(current_neurons, n_labels))

        return layers

class CustomNetBiClassification(CustomNetRegression):
    def __init__(self, n_features, n_labels, num_layers=10, neuron_incr=10,
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels, num_layers=num_layers,
                neuron_incr=neuron_incr, dropout=dropout, batchnorm=batchnorm)
        self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm) + [torch.nn.Sigmoid()])

class CustomNetClassification(CustomNetRegression):
    def __init__(self, n_features=11, n_labels=16, num_layers=10, neuron_incr=10,
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels, num_layers=num_layers,
                neuron_incr=neuron_incr, dropout=dropout, batchnorm=batchnorm)
        self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm) + [torch.nn.Softmax(1)])

class CNNClassifier(LogisticRegression):
    def __init__(self):
        super().__init__(1, 1)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.Dropout(),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(50, 10),
            torch.nn.LogSoftmax(1)
        )