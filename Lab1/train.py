import argparse
import lab1
import matplotlib.pyplot as plt
import time
import torch
from model import Autoencoder
from torch import optim
from torch.nn import functional
from torch.nn import init


def plot_training_loss(training_loss_list, plot_file):
    epoch_list = list(range(1, len(training_loss_list) + 1))
    plt.plot(epoch_list, training_loss_list, label='Train')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(plot_file)


def init_weights(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            # Initialize the weights of the linear layer using Xavier uniform initialization
            init.xavier_uniform_(layer.weight)
            # Initialize the biases of the linear layer to 0
            init.constant_(layer.bias, 0)


def train(model, n_epochs, train_loader, device, loss_function, optimizer, save_file, plot_file):
    # Record the start time
    start_time = time.time()

    print(f'Training for {n_epochs} epochs...')

    # Initialize the weights
    init_weights(model)

    # Set the model to training mode
    model.train()
    training_loss_list = []

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch: {epoch}')
        training_loss = 0.0

        for data in train_loader:
            inputs = data[0][:]
            # Move the inputs to the cpu
            inputs = inputs.to(device)
            # Zero the gradients in the optimizer
            optimizer.zero_grad()
            # Pass the inputs through the model (i.e. call forward)
            outputs = model(inputs.view(-1, 784))
            # Calculate the loss between the inputs and outputs
            loss = loss_function(outputs, inputs.view(-1, 784))
            # Calculate the gradients with respect to loss
            loss.backward()
            # Update the model weights based on the gradients
            optimizer.step()
            training_loss += loss.item()

        # Calculate the avg. training loss for this epoch
        avg_training_loss = training_loss/len(train_loader)
        training_loss_list.append(avg_training_loss)
        print(f'Training loss: {avg_training_loss:.5f}')

    # Save the training parameters
    torch.save(model.state_dict(), save_file)
    # Plot and save the loss curve
    plot_training_loss(training_loss_list, plot_file)

    print('Training complete!')

    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Convert the elapsed time to minutes and seconds
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Elapsed time: {int(minutes)} minutes and {int(seconds)} seconds")


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', type=int, required=True)
    parser.add_argument('-e', type=int, required=True)
    parser.add_argument('-b', type=int, required=True)
    parser.add_argument('-s', type=str, required=True)
    parser.add_argument('-p', type=str, required=True)

    args = parser.parse_args()

    bottleneck_size = args.z
    n_epochs = args.e
    batch_size = args.b
    save_file = args.s
    plot_file = args.p

    # Instantiate the autoencoder
    model = Autoencoder(784, bottleneck_size, 784)
    # Specify the loss function
    loss_function = functional.mse_loss
    # Specify the device (i.e. cpu or gpu)
    device = 'cpu'
    # Specify the parameters (i.e. weights) and the learning rate for the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Download the training dataset
    train_set = lab1.download_mnist_train_dataset()
    # Create an iterable dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)

    # Train the model
    train(model, n_epochs, train_loader, device, loss_function, optimizer, save_file, plot_file)


# Train command: python train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
if __name__ == '__main__':
    main()
