import argparse
import datetime
import custom_dataset
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import utility
from torch import optim
from torch import nn
from torch.nn import init
from torchvision import transforms
from torchvision.models import ResNet18_Weights


def init_convolutional_weights(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            # Initialize the weights of convolutional layers using Kaiming normal initialization
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


def init_linear_weights(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            # Initialize the weights of linear layers using Xavier uniform initialization
            init.xavier_uniform_(layer.weight)


def init_model():
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )

    init_convolutional_weights(model)
    init_linear_weights(model)

    return model


def plot_training_loss(training_loss_list, validation_loss_list, plot_file):
    epoch_list = list(range(1, len(training_loss_list) + 1))
    plt.plot(epoch_list, training_loss_list, label='Train', color='blue')
    plt.plot(epoch_list, validation_loss_list, label='Validation', color='red')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(plot_file)


def train(model, num_epochs, loss_function, optimizer, scheduler, train_dataloader, validation_dataloader, device, weights, loss_plot):
    # Record the start time
    start_time = time.time()
    current_time = datetime.datetime.now()

    print(f'Time: {current_time.strftime("%I:%M:%S %p")}')
    print(f'Training for {num_epochs} epochs...')

    avg_train_loss_list = []
    avg_validation_loss_list = []

    for epoch in range(1, num_epochs + 1):
        # Training metrics
        total_train_loss = 0.0
        num_train_images = 0

        # Validation metrics
        total_validation_loss = 0
        num_validation_images = 0

        # Set the model to training mode
        model.train()

        for train_data in train_dataloader:
            train_inputs, train_labels = train_data
            # Move the inputs to the device
            train_inputs = train_inputs.to(device)
            # Move the labels to the device
            train_labels = train_labels.to(device)
            # Zero the gradients in the optimizer
            optimizer.zero_grad()
            # Pass the inputs through the model (i.e. call forward)
            train_outputs = model(train_inputs)
            train_outputs = train_outputs.view(-1)
            # Calculate training loss
            train_loss = loss_function(train_outputs, train_labels.float())
            # Calculate the gradients with respect to loss
            train_loss.backward()
            # Update the model weights based on the gradients
            optimizer.step()
            # Sum the total loss over the epoch
            total_train_loss += train_loss.item()
            # Sum the total number of images processed in the epoch
            num_train_images += train_labels.size(0)

        # Set the model to evaluation mode (i.e. for validation)
        model.eval()

        with torch.no_grad():
            for validation_data in validation_dataloader:
                validation_inputs, validation_labels = validation_data
                # Move the inputs to the device
                validation_inputs = validation_inputs.to(device)
                # Move the labels to the device
                validation_labels = validation_labels.to(device)
                # Pass the inputs through the model (i.e. call forward)
                validation_outputs = model(validation_inputs)
                # Calculate validation loss
                validation_loss = loss_function(validation_outputs, validation_labels)
                # Sum the total loss over the epoch
                total_validation_loss += validation_loss.item()
                # Sum the total number of images processed in the epoch
                num_validation_images += validation_labels.size(0)

        # Update the scheduler
        scheduler.step()

        # Calculate the avg. training loss for this epoch
        avg_train_loss = total_train_loss / num_train_images
        avg_train_loss_list.append(avg_train_loss)

        # Calculate the avg. validation loss for this epoch
        avg_validation_loss = total_validation_loss / num_validation_images
        avg_validation_loss_list.append(avg_validation_loss)

        current_time = datetime.datetime.now()
        print(f'Time: {current_time.strftime("%I:%M:%S %p")} Epoch: {epoch} Training Loss: {avg_train_loss:.5f} Validation Loss: {avg_validation_loss:.5f}')

    # Save the training parameters
    torch.save(model.state_dict(), weights)

    # Plot and save the loss curve
    plot_training_loss(avg_train_loss_list, avg_validation_loss_list, loss_plot)

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
    parser.add_argument('-train_dataset_directory', type=str, required=True)
    parser.add_argument('-train_label_file', type=str, required=True)
    parser.add_argument('-validation_dataset_directory', type=str, required=True)
    parser.add_argument('-validation_label_file', type=str, required=True)
    parser.add_argument('-epochs', type=int, required=True)
    parser.add_argument('-batch_size', type=int, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-loss_plot', type=str, required=True)
    parser.add_argument('-cuda', type=str, required=True)

    args = parser.parse_args()

    train_dataset_directory = args.train_dataset_directory
    train_label_file = args.train_label_file
    validation_dataset_directory = args.validation_dataset_directory
    validation_label_file = args.validation_label_file
    num_epochs = args.epochs
    batch_size = args.batch_size
    weights = args.weights
    loss_plot = args.loss_plot
    use_cuda = args.cuda

    # Initialize the model
    model = init_model()

    # Check if a CUDA-capable GPU is available and if it should be used for training
    device = utility.setup_device(model, use_cuda)

    # Specify the loss function
    loss_function = nn.BCELoss()

    # Specify the trainable parameters (i.e. weights) and the learning rate for the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)

    # Create a learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    train_dataset = custom_dataset.CustomDataset(train_dataset_directory, train_label_file, transform=transform)
    validation_dataset = custom_dataset.CustomDataset(validation_dataset_directory, validation_label_file, transform=transform)

    # Create iterable datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    train(model, num_epochs, loss_function, optimizer, scheduler, train_dataloader, validation_dataloader, device, weights, loss_plot)


if __name__ == '__main__':
    main()
