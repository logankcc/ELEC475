import argparse
import custom_dataset
#import sklearn.metrics
import torch
import torchvision
import utility
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights


def init_model(weights, use_cuda):
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )

    if torch.cuda.is_available() and use_cuda.lower() == 'y':
        model.load_state_dict(torch.load(weights, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    model.eval()

    return model


def test(model, test_dataloader, device):
    with torch.no_grad():
        num_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_labels = []

        for test_data in test_dataloader:
            test_inputs, test_labels = test_data
            # Move the inputs to the device
            test_inputs = test_inputs.to(device)
            # Move the labels to the device
            test_labels = test_labels.to(device)
            # Pass the inputs through the model (i.e. call forward)
            test_outputs = model(test_inputs)
            test_outputs = test_outputs.view(-1)
            # Sum the total number of samples
            num_samples += test_labels.size(0)
            # Determine the number of correct predictions
            predictions = torch.round(test_outputs)
            correct_predictions += (predictions == test_labels).sum().item()
            # Collect all predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

        # Calculate accuracy
        accuracy = (correct_predictions / num_samples) * 100

        # Generate a confusion matrix
        #confusion_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)
        #print('Confusion Matrix:')
        #print(confusion_matrix)

        print(f'Correct Predictions {correct_predictions}/{num_samples} Accuracy: {accuracy:.1f}%')


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_dataset_directory', type=str, required=True)
    parser.add_argument('-test_label_file', type=str, required=True)
    parser.add_argument('-batch_size', type=int, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-cuda', type=str, required=True)

    args = parser.parse_args()

    test_dataset_directory = args.test_dataset_directory
    test_label_file = args.test_label_file
    batch_size = args.batch_size
    weights = args.weights
    use_cuda = args.cuda

    # Instantiate the model
    model = init_model(weights, use_cuda)

    # Check if a CUDA-capable GPU is available and if it should be used for testing
    device = utility.setup_device(model, use_cuda)

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    test_dataset = custom_dataset.CustomDataset(test_dataset_directory, test_label_file, transform=transform)

    # Create an iterable dataset
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test(model, test_dataloader, device)


if __name__ == '__main__':
    main()
