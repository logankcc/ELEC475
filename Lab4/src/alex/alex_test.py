import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import argparse
import datetime
from alex_custom_dataset import CustomDataset
from sklearn.metrics import confusion_matrix


def test(model, test_loader, device, loss_fn):
    print("testing...")

    model.eval()  # Set the model to evaluation mode
    loss_test = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.view(-1)
            loss = loss_fn(outputs, labels.float())
            loss_test += loss.item()

            # Calculate accuracy
            predictions = torch.round(outputs)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Collect predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = loss_test / len(test_loader)
    accuracy = correct_predictions / total_samples

    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    print('{} Test Loss: {}, Accuracy: {:.2%}'.format(datetime.datetime.now(), avg_test_loss, accuracy))


def main():
    parser = argparse.ArgumentParser()

    # Set up terminal inputs
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-i', '--image-dir', type=str, default='./data/Kitti8ROIs/test')
    parser.add_argument('-t', '--text-file', type=str, default='./data/Kitti8ROIs/test/labels.txt')
    parser.add_argument('-m', '--model-path', type=str, default='./YODA18.pth')

    args = parser.parse_args()

    batch_size = args.batch_size
    img_dir = args.image_dir
    txt_file = args.text_file
    weights_path = args.model_path

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()  # Convert PIL images to tensors
    ])

    test_set = CustomDataset(img_dir, txt_file, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_model = torchvision.models.resnet18()

    # Modify the final fully connected layer for binary classification
    num_ftrs = test_model.fc.in_features
    test_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    # Use binary cross entropy loss function
    loss_fn = nn.BCELoss()

    # Check GPU availability, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    weights = torch.load(weights_path, map_location=torch.device(device))
    test_model.load_state_dict(weights)
    # Move model to the GPU if available
    test_model.to(device)

    test(test_model, test_loader, device, loss_fn)


if __name__ == "__main__":
    main()
