import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, kernel, n_channels, channels_incr, n_conv_layers, stride, padding, dropout_conv, batchnorm_conv,
                n_features, n_labels, n_linear_layers=10, neuron_incr=10, dropout_lin=0.5, batchnorm_lin=False, lr=0.001):
        super().__init__()
        self.lr = lr
        convolutional_layers, neurons = self.get_convolutional_layers(kernel, n_channels, channels_incr, n_conv_layers,
                                                        stride, padding, dropout_conv, batchnorm_conv)
        linear_layers = self.get_linear_layers(neurons, n_labels, n_linear_layers,
                                        neuron_incr, dropout_lin, batchnorm_lin)
        
        layers = convolutional_layers + linear_layers
        self.layers = torch.nn.Sequential(*layers)
        
    def __call__(self, X):
        """
        Predicts the value of an output for each row of X
        using the Logistic Regression model.
        """
        return torch.argmax(self.forward(X), axis=1).reshape(-1, 1)
    
    def forward(self, X):
        return self.layers(X)

    def get_convolutional_layers(self, kernel, n_channels, channels_incr, n_conv_layers, stride, padding, dropout_conv, batchnorm_conv):
        current_channels = n_channels
        layers = []

        for layer in range(n_conv_layers):
            # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout_lin
            # -> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
            next_channels = current_channels*channels_incr
            layers.append(torch.nn.Conv1d(current_channels, next_channels, kernel_size=kernel, stride=stride, padding=padding))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_conv))

            if batchnorm_conv:
                layers.append(torch.nn.BatchNorm1d(next_channels))
            current_channels = next_channels

            # https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659
            # I - a size of input neuron,
            # K - kernel size,
            # P - padding,
            # S - stride.

            I = 140
            K = kernel
            P = padding
            S = stride
            neurons = ((I-K+2*P)/S + 1)
        
        print(current_channels)

        return layers, neurons
    
    def get_linear_layers(self, n_features, n_labels, n_linear_layers, neuron_incr, dropout_lin, batchnorm_lin):
        current_channels = n_features
        layers = []

        for layer in range(n_linear_layers):
            next_neurons = current_channels - neuron_incr
            layers.append(torch.nn.Linear(current_channels, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_lin))

            if batchnorm_lin:
                layers.append(torch.nn.BatchNorm1d(current_channels))

            current_channels = next_neurons
        
        print(current_channels)
        layers.append(torch.nn.Linear(current_channels, n_labels))

        return layers

    def fit(self, train_load, val_load=None, optimiser=None, epochs=1000,
            acceptable_error=0.001, return_loss=False):
        """
        Optimises the model parameters for the given data.

        INPUTS: train_load -> torch.utils.data.DataLoader object with the data.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """

        if optimiser==None:
            optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)

        writer = SummaryWriter()

        mean_train_loss = []
        mean_validation_loss = []

        for epoch in range(epochs):
            training_loss = []
            self.train()
            for X_train, y_train in train_load:
                optimiser.zero_grad()
                y_hat = self.forward(X_train)
                train_loss = self.get_loss(y_hat, y_train)
                training_loss.append(train_loss.item())
                train_loss.backward()
                optimiser.step()
            
            mean_train_loss.append(np.mean(training_loss))
            writer.add_scalar("./loss/train", mean_train_loss[-1], epoch)
            
            if val_load:
                validation_loss = []
                self.eval() # set model in inference mode (need this because of dropout)
                for X_val, y_val in val_load:
                    y_hat_val = self.forward(X_val)
                    val_loss = self.get_loss(y_hat_val, y_val)
                    validation_loss.append(val_loss.item())
                mean_validation_loss.append(np.mean(validation_loss))
                writer.add_scalar("./loss/validation", mean_validation_loss[-1], epoch)

                # if epoch > 2 and (
                #     (abs(mean_validation_loss[-2]- mean_validation_loss[-1])/mean_validation_loss[-1] < acceptable_error)
                #     or (mean_validation_loss[-1] > mean_validation_loss[-2])):
                #     print(f"Validation train_loss for epoch {epoch} is {mean_validation_loss[-1]}")
                #     break
        
        writer.close()
        if return_loss:
            return {'training': mean_train_loss,
                    'validation': mean_validation_loss}
        
    def predict(self, data_load, return_y=False):
        """
        Predicts the value of an output for each row of X
        using the fitted model.

        X is the data from data_load (DataLoader object).

        Returns the predictions.
        """
        self.eval()
        for idx, (X_val, y_val) in enumerate(data_load):
            if idx == 0:
                y_hat_val = self(X_val)
                y_label = y_val
            else:
                y_hat_val = torch.cat((y_hat_val, self(X_val)), dim=0)
                y_label = torch.cat((y_label, y_val), dim=0)
        
        if return_y:
            return y_label.reshape(-1, 1), y_hat_val
        else:
            return y_hat_val

class CustomNetBiClassification(CustomNetRegression):
    def __init__(self, n_features, n_labels, n_linear_layers=10, neuron_incr=10,
                dropout_lin=0.5, batchnorm_lin=False):
        super().__init__(n_features, n_labels, n_linear_layers=n_linear_layers,
                neuron_incr=neuron_incr, dropout_lin=dropout_lin, batchnorm_lin=batchnorm_lin)
        self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, n_linear_layers,
                                        neuron_incr, dropout_lin, batchnorm_lin) + [torch.nn.Sigmoid()])

class CustomNetClassification(CustomNetRegression):
    def __init__(self, n_features=11, n_labels=16, n_linear_layers=10, neuron_incr=10,
                dropout_lin=0.5, batchnorm_lin=False):
        super().__init__(n_features, n_labels, n_linear_layers=n_linear_layers,
                neuron_incr=neuron_incr, dropout_lin=dropout_lin, batchnorm_lin=batchnorm_lin)
        self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, n_linear_layers,
                                        neuron_incr, dropout_lin, batchnorm_lin) + [torch.nn.Softmax(1)])

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