import os
import json
import torchvision.datasets as dsets
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from zlProject.model import efficientnet_b0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose([transforms.Resize(img_size[num_model]),
                                         transforms.CenterCrop(img_size[num_model]),
                                         transforms.ToTensor(),
                                         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    tran = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = ""
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(model)
    model.eval()
    testData = dsets.ImageFolder('', data_transform)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=10, shuffle=False)

    correct = 0
    total = 0
    list = []

    for images, labels in testLoader:
        images = images.cuda()
        outputs = model(images)
        print(outputs)
        _, predicted = torch.max(torch.sigmoid(outputs.data), 1)
        total += labels.size(0)
        print(labels.size(0))
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100 * correct / total))
        list.append(100 * correct / total)


if __name__ == '__main__':
    main()
