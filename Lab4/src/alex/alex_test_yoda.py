from KittiAnchors import Anchors
import argparse
import cv2
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import datetime
from PIL import Image


def strip_ROIs(class_ID, label_list):
    ROIs = []
    for i in range(len(label_list)):
        ROI = label_list[i]
        if ROI[1] == class_ID:
            pt1 = (int(ROI[3]), int(ROI[2]))
            pt2 = (int(ROI[5]), int(ROI[4]))
            ROIs += [(pt1, pt2)]
    return ROIs


def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU


def test(model, test_loader, device):
    print("testing...")

    model.eval()  # Set the model to evaluation mode
    loss_test = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.view(-1)

            # Calculate accuracy
            predictions = torch.round(outputs)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples

    print('{} Accuracy: {:.2%}'.format(datetime.datetime.now(), accuracy))

    return accuracy


def main():
    class_label = {'DontCare': 0, 'Misc': 1, 'Car': 2, 'Truck': 3, 'Van': 4, 'Tram': 5, 'Cyclist': 6, 'Pedestrian': 7,
                   'Person_sitting': 8}

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image-path', type=str, help='label path (./)',
                        default='./data/Kitti8/test/image/006388.png')
    parser.add_argument('-l', '--label-path', type=str, help='image path (./)',
                        default='./data/Kitti8/test/label/006388.txt')
    parser.add_argument('-m', '--model-path', type=str, help='input directory (./)',
                        default='./YODA18.pth')
    args = parser.parse_args()

    label_path = args.label_path
    img_path = args.image_path
    weights_path = args.model_path

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    with open(label_path) as label_file:
        labels_string = label_file.readlines()
    labels = []

    for i in range(len(labels_string)):
        lsplit = labels_string[i].split(' ')
        label = [lsplit[0], int(class_label[lsplit[0]]), float(lsplit[4]), float(lsplit[5]), float(lsplit[6]),
                 float(lsplit[7])]
        labels += [label]

    anchors = Anchors()
    anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

    ROI_images, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

    # print(len(ROIs[0][-1][0])) # image, row, column, RGB

    idx = class_label['Car']

    car_ROIs = strip_ROIs(class_ID=idx, label_list=labels)

    ROI_IoUs = []
    for idx in range(len(ROI_images)):
        ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

    print("Average IoU:", sum(ROI_IoUs)/len(ROI_IoUs))

    IoU_threshold = 0.02
    ROI_labels = []
    for k in range(len(boxes)):
        name_class = 0
        if ROI_IoUs[k] >= IoU_threshold:
            name_class = 1
        ROI_labels += [name_class]

    # Construct image and transform for each ROI array
    ROI_images = [transform(Image.fromarray(ROI)) for ROI in ROI_images]

    ROIs = [(image, label) for image, label in zip(ROI_images, ROI_labels)]

    test_loader = torch.utils.data.DataLoader(ROIs, batch_size=len(ROIs), shuffle=False)

    test_model = torchvision.models.resnet18()

    # Modify the final fully connected layer for binary classification
    num_ftrs = test_model.fc.in_features
    test_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    weights = torch.load(weights_path, map_location=torch.device(device))
    test_model.load_state_dict(weights)
    test_model.to(device)

    test_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            images = images.to(device)
            outputs = test_model(images)
            outputs = outputs.view(-1)

            predictions = torch.round(outputs)

            IoUs = []
            image_copy = image.copy()
            tint = 0
            for i in range(len(predictions)):
                if predictions[i] == 1:
                    tint += 30
                    # print(boxes[i])
                    # print(labels[i])
                    IoUs.append(calc_max_IoU(boxes[i], car_ROIs))
                    box = boxes[i]
                    pt1 = (box[0][1], box[0][0])
                    pt2 = (box[1][1], box[1][0])
                    cv2.rectangle(image_copy, pt1, pt2, color=(255, 255, 0))
            print("IoU score for each 'Car' ROI:", IoUs)
            cv2.imshow('boxes', image_copy)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break


if __name__ == "__main__":
    main()
