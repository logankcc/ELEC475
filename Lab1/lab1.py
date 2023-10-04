import argparse
import sys
import torch
import torchvision
from matplotlib import pyplot as plt
from model import Autoencoder
from torchvision import transforms


def download_mnist_train_dataset():
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    return train_set


def download_mnist_test_dataset():
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=train_transform)
    return test_set


def get_mnist_image_from_user(dataset):
    # Get an integer value from the user
    idx = input(f'Enter an integer value between 0 and {len(dataset) - 1}: ')
    idx = int(idx)

    # Check the user input
    if (idx < 0 or idx > len(dataset) - 1):
        print(f'Error: {idx} is not valid input.')
        sys.exit(1)

    # Get the MNIST image at index idx
    image = dataset.data[idx]

    # Normalize the image's pixel values between 0 and 1
    normalized_image = image.float() / 255.0

    return normalized_image


def load(save_file):
    # Instantiate the autoencoder
    model = Autoencoder()
    # Load the saved parameters
    model.load_state_dict(torch.load(save_file))
    # Set the model to evaluation mode (i.e. for inference)
    model.eval()

    return model


def add_uniform_noise(image, noise_level):
    # Generate random noise with the same shape as the image
    noise = torch.rand_like(image) * 2 * noise_level - noise_level

    # Add noise to the image
    noisy_image = image + noise

    return noisy_image


def plot_inference_comparison(input_image, output_image):
    figure = plt.figure()
    figure.add_subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    figure.add_subplot(1, 2, 2)
    plt.imshow(output_image, cmap='gray')
    plt.show()


def plot_denoising_comparison(input_image, noisy_image, output_image):
    figure = plt.figure()
    figure.add_subplot(1, 3, 1)
    plt.imshow(input_image, cmap='gray')
    figure.add_subplot(1, 3, 2)
    plt.imshow(noisy_image, cmap='gray')
    figure.add_subplot(1, 3, 3)
    plt.imshow(output_image, cmap='gray')
    plt.show()


def plot_interpolation(output_images):
    figure = plt.figure()
    for index, image in enumerate(output_images):
        figure.add_subplot(1, len(output_images), index + 1)
        plt.imshow(image, cmap='gray')
    plt.show()


def infer(save_file):
    # Disable gradient calculations
    with torch.no_grad():
        # Download the MNIST test dataset
        dataset = download_mnist_test_dataset()
        # Get a MNIST image index from the user
        input_image = get_mnist_image_from_user(dataset)
        # Move the input image to the cpu
        input_image = input_image.to('cpu')
        # Load the pre-trained model parameters
        model = load(save_file)
        # Pass the input through the model (i.e. call forward)
        output_image = model(input_image.view(1, 784))
        plot_inference_comparison(input_image.view(28, 28), output_image.view(28, 28))


def denoise(save_file):
    # Disable gradient calculations
    with torch.no_grad():
        # Download the MNIST test dataset
        dataset = download_mnist_test_dataset()
        # Get a MNIST image index from the user
        input_image = get_mnist_image_from_user(dataset)
        # Add uniform noise to the input image
        noisy_image = add_uniform_noise(input_image, 0.25)
        # Move the noisy image to the cpu
        input_image = input_image.to('cpu')
        # Load the pre-trained model parameters
        model = load(save_file)
        # Pass the noisy image through the model (i.e. call forward)
        output_image = model(noisy_image.view(1, 784))
        plot_denoising_comparison(input_image.view(28, 28), noisy_image.view(28, 28), output_image.view(28, 28))


def tensor_linear_interpolation(tensor_a, tensor_b, n_steps):
    # Generate a linear space between tensor_a and tensor_b
    interpolation_tensors = torch.linspace(0, 1, n_steps).unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Perform the linear interpolation
    interpolated_tensors = tensor_a + (tensor_b - tensor_a) * interpolation_tensors

    return interpolated_tensors


def interpolate(save_file):
    # Disable gradient calculations
    with torch.no_grad():
        # Download the MNIST test dataset
        dataset = download_mnist_test_dataset()
        # Get MNIST image indexes from the user
        image_a, image_b = get_mnist_image_from_user(dataset), get_mnist_image_from_user(dataset)
        # Move the images to the cpu
        image_a, image_b = image_a.to('cpu'), image_b.to('cpu')
        # Load the pre-trained model parameters
        model = load(save_file)
        # Encode the images
        encoded_image_a, encoded_image_b = model.encode(image_a.view(1, 784)), model.encode(image_b.view(1, 784))
        # Perform a linear transformation between tensor encoded_image_a and tensor encoded_image_b
        interpolated_tensors = tensor_linear_interpolation(encoded_image_a, encoded_image_b, 8)

        output_images = []

        # Add image_a to the output list
        output_images.append(image_a)

        # Decode each interpolated tensor and add it to the output list
        for tensor in interpolated_tensors:
            decoded_image = model.decode(tensor)
            output_images.append(decoded_image)

        # Add image_b to the output list
        output_images.append(image_b)

        # Resize the output images
        for index, image in enumerate(output_images):
            output_images[index] = image.view(28, 28)

        plot_interpolation(output_images)


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, required=True)
    args = parser.parse_args()
    save_file = args.l

    print('ELEC 475 Lab 1')
    print()

    print('Testing Step 4 - Test Your Autoencoder')
    infer(save_file)
    print()

    print('Testing Step 5 - Image Denoising')
    denoise(save_file)
    print()

    print('Testing Step 6 - Bottleneck Interpolation')
    interpolate(save_file)
    print()


# Test command: python lab1.py -l MLP.8.pth
if __name__ == '__main__':
    main()
