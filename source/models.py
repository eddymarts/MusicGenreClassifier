import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class MusicClassifier(torch.nn.Module):
    def __init__(self, kernel=5, channels_incr=4, n_conv_layers=4, stride=1, padding=0, dropout_conv=0.5, batchnorm_conv=True,
                n_labels=16, n_linear_layers=4, neuron_incr=6, dropout_lin=0.5, batchnorm_lin=True, lr=0.001):
        super().__init__()
        self.lr = lr
        self.device = torch.device("cuda:0" if cuda_available else "cpu")
        convolutional_layers, neurons = self.get_convolutional_layers(kernel, channels_incr, n_conv_layers,
                                                        stride, padding, dropout_conv, batchnorm_conv)
        linear_layers = self.get_linear_layers(neurons, n_labels, n_linear_layers,
                                        neuron_incr, dropout_lin, batchnorm_lin)
        
        layers = convolutional_layers + [torch.nn.Flatten()] + linear_layers + [torch.nn.Softmax(1)]
        self.layers = torch.nn.Sequential(*layers)
        # self.layers = torch.nn.ModuleList(layers)

    def __call__(self, X):
        """
        Predicts the value of an output for each row of X
        using the Logistic Regression model.
        """
        return torch.argmax(self.forward(X), axis=1).reshape(-1, 1)
    
    def forward(self, X):
        torch.cuda.synchronize()
        print(X.device)
        return self.layers(X)
        # for idx, layer in enumerate(self.layers):
        #     print(f"Forward layer {idx}: Shape of data {X.shape}")
        #     X = layer(X)
        # return X
    
    def get_loss(self, y_hat, y):
        """
        Gets Cross Entropy between predictions (y_hat) and actual value (y).
        """
        return F.cross_entropy(y_hat, y.long().reshape(-1))

    def get_convolutional_layers(self, kernel, channels_incr, n_conv_layers, stride, padding, dropout_conv, batchnorm_conv):
        current_channels = 1
        layers = []

        I = 140
        K = kernel
        P = padding
        S = stride

        neurons = I

        for layer in range(n_conv_layers):
            # https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659
            # I - a size of input neuron,
            # K - kernel size,
            # P - padding,
            # S - stride.
            neurons = ((neurons-K+2*P)/S + 1)
            # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout_lin
            # -> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
            next_channels = current_channels*channels_incr
            print(f"Convolutional layer {layer}: neurons: {neurons} | in channels: {current_channels} | out channels: {next_channels}")
            layers.append(torch.nn.Conv1d(current_channels, next_channels, kernel_size=kernel, stride=stride, padding=padding))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_conv))

            if batchnorm_conv:
                layers.append(torch.nn.BatchNorm1d(next_channels))
            current_channels = next_channels

        return layers, neurons*current_channels
    
    def get_linear_layers(self, n_features, n_labels, n_linear_layers, neuron_incr, dropout_lin, batchnorm_lin):
        current_neurons = int(n_features)
        layers = []

        for layer in range(n_linear_layers):
            next_neurons = int(round(current_neurons/neuron_incr))
            print(f"Linear layer {layer}: in neurons: {current_neurons} | out neurons: {next_neurons}")
            layers.append(torch.nn.Linear(current_neurons, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_lin))

            if batchnorm_lin:
                layers.append(torch.nn.BatchNorm1d(next_neurons))

            current_neurons = int(next_neurons)

        layers.append(torch.nn.Linear(current_neurons, n_labels))

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
            for idx, (X_train, y_train) in enumerate(train_load):
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                optimiser.zero_grad()
                y_hat = self.forward(X_train)
                torch.cuda.synchronize()
                train_loss = self.get_loss(y_hat, y_train)
                training_loss.append(train_loss.item())
                torch.cuda.synchronize()
                train_loss.backward()
                torch.cuda.synchronize()
                optimiser.step()
                print(f"Model on cuda? {next(self.parameters()).is_cuda} | Data device {X_train.device} Epoch{epoch}, | Batch {idx}: Train batch loss: {train_loss.item()}")
            
            mean_train_loss.append(np.mean(training_loss))
            writer.add_scalar("./loss/train", mean_train_loss[-1], epoch)
            
            if val_load:
                validation_loss = []
                self.eval() # set model in inference mode (need this because of dropout)
                for idx, (X_val, y_val) in enumerate(val_load):
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    y_hat_val = self.forward(X_val)
                    val_loss = self.get_loss(y_hat_val, y_val)
                    validation_loss.append(val_loss.item())
                    print(f"Epoch{epoch}, Batch {idx}: Val batch loss: {val_loss.item()}")
                mean_validation_loss.append(np.mean(validation_loss))
                writer.add_scalar("./loss/validation", mean_validation_loss[-1], epoch)
                print(f"----Epoch: {epoch} | Train loss: {mean_train_loss[-1]} | Val loss: {mean_validation_loss[-1]}----")
            


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
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
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

# class AudioClassifier (nn.Module):
#     # ----------------------------
#     # Build the model architecture
#     # ----------------------------
#     def __init__(self):
#         super().__init__()
#         conv_layers = []

#         # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
#         self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#         self.relu1 = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(8)
#         init.kaiming_normal_(self.conv1.weight, a=0.1)
#         self.conv1.bias.data.zero_()
#         conv_layers += [self.conv1, self.relu1, self.bn1]

#         # Second Convolution Block
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu2 = nn.ReLU()
#         self.bn2 = nn.BatchNorm2d(16)
#         init.kaiming_normal_(self.conv2.weight, a=0.1)
#         self.conv2.bias.data.zero_()
#         conv_layers += [self.conv2, self.relu2, self.bn2]

#         # Second Convolution Block
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu3 = nn.ReLU()
#         self.bn3 = nn.BatchNorm2d(32)
#         init.kaiming_normal_(self.conv3.weight, a=0.1)
#         self.conv3.bias.data.zero_()
#         conv_layers += [self.conv3, self.relu3, self.bn3]

#         # Second Convolution Block
#         self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu4 = nn.ReLU()
#         self.bn4 = nn.BatchNorm2d(64)
#         init.kaiming_normal_(self.conv4.weight, a=0.1)
#         self.conv4.bias.data.zero_()
#         conv_layers += [self.conv4, self.relu4, self.bn4]

#         # Linear Classifier
#         self.ap = nn.AdaptiveAvgPool2d(output_size=1)
#         self.lin = nn.Linear(in_features=64, out_features=10)

#         # Wrap the Convolutional Blocks
#         self.conv = nn.Sequential(*conv_layers)
 
#     # ----------------------------
#     # Forward pass computations
#     # ----------------------------
#     def forward(self, x):
#         # Run the convolutional blocks
#         x = self.conv(x)

#         # Adaptive pool and flatten for input to linear layer
#         x = self.ap(x)
#         x = x.view(x.shape[0], -1)

#         # Linear layer
#         x = self.lin(x)

#         # Final output
#         return x


# class CustomNetBiClassification(CustomNetRegression):
#     def __init__(self, n_features, n_labels, n_linear_layers=10, neuron_incr=10,
#                 dropout_lin=0.5, batchnorm_lin=False):
#         super().__init__(n_features, n_labels, n_linear_layers=n_linear_layers,
#                 neuron_incr=neuron_incr, dropout_lin=dropout_lin, batchnorm_lin=batchnorm_lin)
#         self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, n_linear_layers,
#                                         neuron_incr, dropout_lin, batchnorm_lin) + [torch.nn.Sigmoid()])

# class CustomNetClassification(CustomNetRegression):
#     def __init__(self, n_features=11, n_labels=16, n_linear_layers=10, neuron_incr=10,
#                 dropout_lin=0.5, batchnorm_lin=False):
#         super().__init__(n_features, n_labels, n_linear_layers=n_linear_layers,
#                 neuron_incr=neuron_incr, dropout_lin=dropout_lin, batchnorm_lin=batchnorm_lin)
#         self.layers = torch.nn.ModuleList(self.get_linear_layers(n_features, n_labels, n_linear_layers,
#                                         neuron_incr, dropout_lin, batchnorm_lin) + [torch.nn.Softmax(1)])

# class CNNClassifier(LogisticRegression):
#     def __init__(self):
#         super().__init__(1, 1)
#         self.layers = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 10, kernel_size=5),
#             torch.nn.MaxPool2d(2),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(10, 20, kernel_size=5),
#             torch.nn.Dropout(),
#             torch.nn.MaxPool2d(2),
#             torch.nn.ReLU(),
#             torch.nn.Flatten(),
#             torch.nn.Linear(320, 50),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(),
#             torch.nn.Linear(50, 10),
#             torch.nn.LogSoftmax(1)
#         )

if __name__ == "__main__":
    from dataset import MusicData
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    cuda = True
    cuda_available = torch.cuda.is_available()

    if cuda:
        device = torch.device("cuda:0" if cuda_available else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Cuda? {cuda}. Selected device: {device}")

    music = MusicData(device=device)
    music_classifier = MusicClassifier()
    music_classifier.to(device)
    loss = music_classifier.fit(music.train_load, music.test_load, return_loss=True,
                                epochs=10, acceptable_error=0.0001)

    y_val, y_hat_val = music_classifier.predict(music.test_load, return_y=True)

    print(torch.cat((y_val, y_hat_val), dim=1)[0:10])
    print("R^2 score:", f1_score(y_hat_val.detach().numpy(), y_val.detach().numpy()))
    plt.plot(loss['training'], label="Training set loss")
    plt.plot(loss['validation'], label="Validation set loss")
    plt.xlabel(f"Epochs\nl={loss['validation'][-1]}")
    plt.ylabel("CE")
    plt.show()