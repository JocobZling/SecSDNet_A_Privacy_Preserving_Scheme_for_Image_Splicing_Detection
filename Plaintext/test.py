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

    # # load image
    # img_path = "C:\\Users\\zl\\Desktop\\casia2.0\\1_real\\real2.png"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)
    #
    # # read class_indict
    # json_path = 'D:\\pythonStudy\\deepLearningForImageProcessing\\pytorch_classification\\Test9_efficientNet\\class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    #
    # json_file = open(json_path, "r")
    # class_indict = json.load(json_file)
    #
    # # create model
    # model = create_model(num_classes=2).to(device)
    # # load model weights
    # model_weight_path = "C:\\Users\\zl\\Desktop\\model-B0-1-0801-462.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # model.eval()
    # with torch.no_grad():
    #     # predict class
    #     output = torch.squeeze(model(img.to(device))).cpu()
    #     print(output)
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()
    #
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # print(print_res)
    # plt.show()
    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "C:\\Users\\zl\\Desktop\\model-B0-A-0808-442.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print(model)
    model.eval()
    testData = dsets.ImageFolder('C:\\Users\\zl\\Desktop\\casia1.0\\test\\', data_transform)
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
