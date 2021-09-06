import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from functools import partial
# import os
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

class MusicClassifier(torch.nn.Module):
    # def __init__(self, kernel=5, channels_incr=4, n_conv_layers=4, stride=1, padding=0, dropout_conv=0.5, batchnorm_conv=True,
    #             n_labels=16, n_linear_layers=10, neuron_incr=12, dropout_lin=0.5, batchnorm_lin=True, lr=0.1):
    def __init__(self, n_linear_layers=10, neuron_incr=12, dropout_lin=0.5, batchnorm_lin=True, lr=0.1):
        super().__init__()
        self.lr = lr
        self.device = torch.device("cuda:0" if cuda_available else "cpu")
        # convolutional_layers, neurons = self.get_convolutional_layers(kernel, channels_incr, n_conv_layers,
        #                                                 stride, padding, dropout_conv, batchnorm_conv)
        # linear_layers = self.get_linear_layers(neurons, n_labels, n_linear_layers,
        #                                 neuron_incr, dropout_lin, batchnorm_lin)
        
        # layers = convolutional_layers + [torch.nn.Flatten()] + linear_layers + [torch.nn.Softmax(1)]
        linear_layers = self.get_linear_layers(140, 16, n_linear_layers,
                                        neuron_incr, dropout_lin, batchnorm_lin)
        
        layers = linear_layers + [torch.nn.Softmax(1)]
        self.layers = torch.nn.Sequential(*layers)
        # self.layers = torch.nn.ModuleList(layers)

    def __call__(self, X):
        """
        Predicts the value of an output for each row of X
        using the Logistic Regression model.
        """
        return torch.argmax(self.forward(X), axis=1).reshape(-1, 1)
    
    def forward(self, X):
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
            next_neurons = int(round(current_neurons-neuron_incr))
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
        mean_train_f1 = []
        mean_val_f1 = []
        print("Learning rate = ", self.lr)

        for epoch in range(epochs):
            training_loss = []
            train_f1 = []
            self.train()
            for idx, (X_train, y_train) in enumerate(train_load):
                X_train, y_train = X_train.to(self.device, non_blocking=True), y_train.to(self.device, non_blocking=True)
                optimiser.zero_grad()
                y_hat = self.forward(X_train)
                train_loss = self.get_loss(y_hat, y_train)
                training_loss.append(train_loss.item())
                train_loss.backward()
                optimiser.step()
                y_hat = self(X_train)
                t_f1 = f1_score(y_hat.detach().cpu().clone().numpy(), y_train.detach().cpu().clone().numpy(), average='weighted')
                train_f1.append(t_f1)
                # print(f"Model on cuda? {next(self.parameters()).is_cuda} | Data device {X_train.device} Epoch{epoch}, | Batch {idx}: Train batch loss: {train_loss.item()}")
            
            mean_train_loss.append(np.mean(training_loss))
            mean_train_f1.append(np.mean(train_f1))
            writer.add_scalar("./loss/train", mean_train_loss[-1], epoch)
            
            if val_load:
                validation_loss = []
                val_f1 = []
                self.eval() # set model in inference mode (need this because of dropout)
                for idx, (X_val, y_val) in enumerate(val_load):
                    X_val, y_val = X_val.to(self.device, non_blocking=True), y_val.to(self.device, non_blocking=True)
                    y_hat_val = self.forward(X_val)
                    val_loss = self.get_loss(y_hat_val, y_val)
                    validation_loss.append(val_loss.item())
                    y_hat_val = self(X_val)
                    v_f1 = f1_score(y_hat_val.detach().cpu().clone().numpy(), y_val.detach().cpu().clone().numpy(), average='weighted')
                    val_f1.append(v_f1)
                    # print(f"Epoch{epoch}, Batch {idx}: Val batch loss: {val_loss.item()}")
                mean_validation_loss.append(np.mean(validation_loss))
                mean_val_f1.append(np.mean(val_f1))
                writer.add_scalar("./loss/validation", mean_validation_loss[-1], epoch)
                print(f"----Epoch: {epoch} | Train: loss={mean_train_loss[-1]}, F1={mean_train_f1[-1]} | Val: loss={mean_validation_loss[-1]}, F1={mean_val_f1[-1]}----")
            


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
            X_val, y_val = X_val.to(self.device, non_blocking=True), y_val.to(self.device, non_blocking=True)
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

    music = MusicData()
    music_classifier = MusicClassifier()
    music_classifier.to(device)
    loss = music_classifier.fit(music.train_load, music.test_load, return_loss=True,
                                epochs=100, acceptable_error=0.0001)

    y_val, y_hat_val = music_classifier.predict(music.test_load, return_y=True)

    print(torch.cat((y_val, y_hat_val), dim=1)[0:10])
    print("R^2 score:", f1_score(y_hat_val.detach().cpu().clone().numpy(), y_val.detach().cpu().clone().numpy(), average='weighted'))
    plt.plot(loss['training'], label="Training set loss")
    plt.plot(loss['validation'], label="Validation set loss")
    plt.xlabel(f"Epochs\nl={loss['validation'][-1]}")
    plt.ylabel("CE")
    plt.show()