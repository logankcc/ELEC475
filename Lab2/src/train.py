import AdaIN_net
import argparse
import custom_dataset
import datetime
import matplotlib.pyplot as plt
import time
import torch
from torch import optim
from torchvision import transforms

# Constants
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 5e-5

def init_model(encoder_file):
    # Instantiate the encoder
    encoder = AdaIN_net.encoder_decoder.encoder
    # Load the saved encoder parameters
    encoder.load_state_dict(torch.load(encoder_file))
    # Instantiate the decoder
    decoder = AdaIN_net.encoder_decoder.decoder
    # Instantiate the model
    model = AdaIN_net.AdaIN_net(encoder, decoder)
    return model

# NOTE: adjust_learning_rate was copied from the adjust_learning_rate method found in https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py
def adjust_learning_rate(optimizer, iteration_count):
    adjusted_learning_rate = LEARNING_RATE / (1.0 + LEARNING_RATE_DECAY * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = adjusted_learning_rate

def plot_training_loss(training_loss_list, content_loss_list, style_loss_list, plot_file):
    epoch_list = list(range(1, len(training_loss_list) + 1))
    plt.plot(epoch_list, training_loss_list, label='Content+Style')
    plt.plot(epoch_list, content_loss_list, label='Content')
    plt.plot(epoch_list, style_loss_list, label='Style')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(plot_file)

def train(model, gamma, num_epochs, optimizer, content_dataloader, style_dataloader, device, decoder_file, loss_plot):
    # Record the start time
    start_time = time.time()
    current_time = datetime.datetime.now()
    print(f'Time: {current_time.strftime("%I:%M:%S %p")}')
    print(f'Training for {num_epochs} epochs...')

    # Set the model to training mode
    model.train()
    content_loss_list = []
    style_loss_list = []
    total_loss_list = []

    for epoch in range(1, num_epochs + 1):
        content_loss = 0.0
        style_loss = 0.0
        total_loss = 0.0

        batch_content_loss = 0.0
        batch_style_loss = 0.0
        batch_total_loss = 0.0

        num_batches = len(content_dataloader)

        for batch in range(num_batches):
            # Move the images to the device
            content_images = next(iter(content_dataloader)).to(device)
            style_images = next(iter(style_dataloader)).to(device)
            # Zero the gradients in the optimizer
            optimizer.zero_grad()
            # Pass the images through the model (i.e. call forward)
            loss_c, loss_s = model(content_images, style_images)
            # Calculate loss
            loss_s = gamma * loss_s
            loss = loss_c + loss_s
            # Calculate the gradients with respect to loss
            loss.backward()
            # Update the model weights based on the gradients
            optimizer.step()
            # Sum the loss of each batch
            batch_content_loss += loss_c.item()
            batch_style_loss += loss_s.item()
            batch_total_loss += loss.item()

        # NOTE: call to the adjust_learning_rate method is commented out as it resulted in slower convergence
        #adjust_learning_rate(optimizer, epoch)

        # Calculate the avg. training loss for this epoch
        content_loss = batch_style_loss/num_batches
        style_loss = batch_content_loss/num_batches
        total_loss = batch_total_loss/num_batches

        content_loss_list.append(content_loss)
        style_loss_list.append(style_loss)
        total_loss_list.append(total_loss)
        current_time = datetime.datetime.now()
        print(f'Time: {current_time.strftime("%I:%M:%S %p")} Epoch: {epoch} Content loss: {content_loss:.5f} Style loss: {style_loss:.5f} Total loss: {total_loss:.5f}') 

    # Save the training parameters
    torch.save(model.decoder.state_dict(), decoder_file)
    # Plot and save the loss curve
    plot_training_loss(total_loss_list, content_loss_list, style_loss_list, loss_plot)

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
    parser.add_argument('-content_dir', type=str, required=True)
    parser.add_argument('-style_dir', type=str, required=True)
    parser.add_argument('-gamma', type=float, required=True)
    parser.add_argument('-e', type=int, required=True)
    parser.add_argument('-b', type=int, required=True)
    parser.add_argument('-l', type=str, required=True)
    parser.add_argument('-s', type=str, required=True)
    parser.add_argument('-p', type=str, required=True)
    parser.add_argument('-cuda', type=str, required=True)

    args = parser.parse_args()

    content_directory_path = args.content_dir
    style_directory_path = args.style_dir
    gamma = args.gamma
    num_epochs = args.e
    batch_size = args.b
    encoder_file = args.l
    decoder_file = args.s
    loss_plot = args.p
    use_cuda = args.cuda.lower()

    # Instantiate the model
    model = init_model(encoder_file)

    # Define image transformations for the training data
    train_transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()])
        
    # Specify the parameters (i.e. weights) and the learning rate for the optimizer
    optimizer = optim.Adam(model.decoder.parameters(), LEARNING_RATE)

    # Create custom datasets for content and style images
    content_dataset = custom_dataset.custom_dataset(content_directory_path, train_transform)
    style_dataset = custom_dataset.custom_dataset(style_directory_path, train_transform)

    # Create data loaders for content and style datasets
    content_dataloader = torch.utils.data.DataLoader(content_dataset, batch_size, shuffle=True)
    style_dataloader = torch.utils.data.DataLoader(style_dataset, batch_size, shuffle=True)

    # Check if a CUDA-capable GPU is available and if it should be used for training
    device = None
    if torch.cuda.is_available() and use_cuda == 'y':
        print('Using CUDA-capable GPU for training...')
        device = torch.device('cuda')
        model.cuda()
    else:
        device = torch.device('cpu')
        print('Using CPU for training...')

    train(model, gamma, num_epochs, optimizer, content_dataloader, style_dataloader, device, decoder_file, loss_plot)

if __name__ == '__main__':
    main()