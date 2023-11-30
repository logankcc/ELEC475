import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torchvision
from torchvision import transforms
import argparse
import datetime
import matplotlib.pyplot as plt
from alex_custom_dataset import CustomDataset


def train(n_epochs, optimizer, model, scheduler, loss_fn, train_loader, valid_loader, device,
          save_model_path, save_plot_path):
    print("training...")

    avg_loss = []
    losses_valid = []
    epochs = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch', epoch)

        # Initialize a new list for this epoch
        loss_train = 0.00

        data_iter = iter(train_loader)

        model.train()  # Keep track of gradient for backtracking

        # Iterate through batches
        for batch in range(int(len(train_loader))):
            # print(batch)
            images, labels = next(data_iter)
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass through model
            outputs = model(images)
            outputs = outputs.view(-1)
            # Calculate loss
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # Calculate the average loss over batches for the entire epoch
        avg_loss += [loss_train / len(train_loader)]

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        loss_valid = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs = outputs.view(-1)
                loss = loss_fn(outputs, labels.float())
                loss_valid += loss.item()

        # Calculate the average validation loss for the entire epoch
        avg_valid_loss = loss_valid / len(valid_loader)
        losses_valid.append(avg_valid_loss)

        scheduler.step()

        # Build array for plotting loss
        epochs.append(epoch)

        print('{} Epoch {}, Training loss {}, Validation loss {}'.format(datetime.datetime.now(), epoch,
                                                                         loss_train / len(train_loader),
                                                                         loss_valid / len(valid_loader)))
        # Used this to save iteratively if it converged earlier than expected
        if epoch > 5 or epoch == 1:
            torch.save(model.state_dict(), save_model_path)

    # Plot training and validation loss over epochs
    plt.plot(epochs, avg_loss, label='Training Loss', color='blue')
    plt.plot(epochs, losses_valid, label='Validation Loss', color='red')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(save_plot_path)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # Set up terminal inputs
    parser.add_argument('-e', '--epochs', type=int, default=18)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-i', '--image-dir', type=str, default='./data/Kitti8ROIs/train')
    parser.add_argument('-t', '--text-file', type=str, default='./data/Kitti8ROIs/train/labels.txt')
    parser.add_argument('-s', '--save-model', type=str, default='YODA.pth')
    parser.add_argument('-p', '--save-plot', type=str, default='loss.YODA.png')

    args = parser.parse_args()

    n_epochs = args.epochs
    batch_size = args.batch_size
    img_dir = args.image_dir
    txt_file = args.text_file
    save_model_path = args.save_model
    save_plot_path = args.save_plot

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()  # Convert PIL images to tensors
    ])

    data_set = CustomDataset(img_dir, txt_file, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(data_set))
    valid_size = len(data_set) - train_size
    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model = torchvision.models.resnet18(pretrained=True)

    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    # Initialize modified model weights
    nn.init.xavier_uniform_(model.fc[0].weight)
    nn.init.constant_(model.fc[0].bias, 0)

    optimizer = optim.SGD(model.parameters(), lr=0.0005, weight_decay=0.0005, momentum=0.9)

    # Use binary cross entropy loss function
    loss_fn = nn.BCELoss()

    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

    # Check GPU availability, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Move model to the GPU if available
    model.to(device)

    # Calling the train method
    train(n_epochs, optimizer, model, scheduler, loss_fn, train_loader, valid_loader, device,
          save_model_path, save_plot_path)


if __name__ == "__main__":
    main()
