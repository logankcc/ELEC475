from torch import nn
from torch.nn import functional

# Autoencoder -------------------------------------------------------------------------------------
#   This class inherits nn.Module and defines a four layer feedforward fully connected autoencoder.
#   Input and output image sizes are 28x28 pixels.
# -------------------------------------------------------------------------------------------------


class Autoencoder(nn.Module):

    def __init__(self, input_size=784, bottleneck_size=8, output_size=784):
        # Call the constructor of the superclass nn.Module
        super().__init__()
        # Define the layers of the autoencoder
        self.fully_connected_layer_1 = nn.Linear(input_size, input_size//2)
        self.fully_connected_layer_2 = nn.Linear(input_size//2, bottleneck_size)
        self.fully_connected_layer_3 = nn.Linear(bottleneck_size, output_size//2)
        self.fully_connected_layer_4 = nn.Linear(output_size//2, output_size)

    def encode(self, x):
        x = functional.relu(self.fully_connected_layer_1(x))
        x = functional.relu(self.fully_connected_layer_2(x))

        return x

    def decode(self, x):
        x = functional.relu(self.fully_connected_layer_3(x))
        x = functional.sigmoid(self.fully_connected_layer_4(x))

        return x

    def forward(self, x):
        return self.decode(self.encode(x))
