import torch
import torch.nn.functional as F

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

class MusicClassifier(torch.nn.Module):
    def __init__(self, n_channels, n_conv_layers, n_features, n_labels, n_linear_layers=10, neuron_incr=10, 
                dropout=0.5, batchnorm=False):
        super().__init__()
        convolutional_layers = self.get_convolutional_layers(n_channels, n_conv_layers, n_features)
        linear_layers = self.get_linear_layers(n_features, n_labels, n_linear_layers,
                                        neuron_incr, dropout, batchnorm)
        
        layers = convolutional_layers + linear_layers
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, X):
        return self.layers(X)

    def get_convolutional_layers(self, n_channels, n_conv_layers, n_features):
        current_channels = n_channels
        layers = []

        for layer in range(n_conv_layers):
            if layer <= round(n_conv_layers/2):
                next_neurons = current_channels + neuron_incr
            else:
                next_neurons = current_channels - neuron_incr

            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(current_channels))

            layers.append(torch.nn.Linear(current_channels, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            current_channels = next_neurons
        
        print(current_channels)

        if batchnorm:
            layers.append(torch.nn.BatchNorm1d(current_channels))
        layers.append(torch.nn.Linear(current_channels, n_labels))

        return layers
    
    def get_linear_layers(self, n_features, n_labels, n_linear_layers, neuron_incr, dropout, batchnorm):
        current_channels = n_features
        layers = []

        for layer in range(n_linear_layers):
            if layer <= round(n_linear_layers/2):
                next_neurons = current_channels + neuron_incr
            else:
                next_neurons = current_channels - neuron_incr

            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(current_channels))

            layers.append(torch.nn.Linear(current_channels, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            current_channels = next_neurons
        
        print(current_channels)

        if batchnorm:
            layers.append(torch.nn.BatchNorm1d(current_channels))
        layers.append(torch.nn.Linear(current_channels, n_labels))

        return layers

class CustomNetBiClassification(CustomNetRegression):
    def __init__(self, n_features, n_labels, n_linear_layers=10, neuron_incr=10,
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels, n_linear_layers=n_linear_layers,
                neuron_incr=neuron_incr, dropout=dropout, batchnorm=batchnorm)
        self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, n_linear_layers,
                                        neuron_incr, dropout, batchnorm) + [torch.nn.Sigmoid()])

class CustomNetClassification(CustomNetRegression):
    def __init__(self, n_features=11, n_labels=16, n_linear_layers=10, neuron_incr=10,
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels, n_linear_layers=n_linear_layers,
                neuron_incr=neuron_incr, dropout=dropout, batchnorm=batchnorm)
        self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, n_linear_layers,
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