import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from zlProject.test.modelUpdateEfficientNet1 import efficientnet_b0 as create_model
from zlProject.test.my_dataset import MyDataSet
from zlProject.test.utils import read_split_data, train_one_epoch, evaluate

savePath = '/data1/zl/first/model_efficientNet/weights-pretrain-coco-copy-move-B0/model-loss.txt'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights-pretrain-coco-copy-move-B0") is False:
        os.makedirs("./weights-pretrain-coco-copy-move-B0")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    # 处理图片大小
    # 随机长宽比裁剪 transforms.RandomResizedCrop
    # 依概率p水平翻转transforms.RandomHorizontalFlip
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=2).to(device)
    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.0004)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(args.epochs):
        # train
        mean_loss, train_sum = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch)
        train_acc = train_sum / len(train_data_set)
        optimizer.step()

        # validate
        val_loss, sum_num = evaluate(model=model,
                                     optimizer=optimizer,
                                     data_loader=val_loader,
                                     device=device)
        acc = sum_num / len(val_data_set)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

        # 记录 avg_loss 和 epoch acc
        with open(savePath, "a") as f:
            f.write("[E: %d] mean_loss: %f, val_loss: %f,  val_acc: %f , train_acc: %f" % (
                epoch, mean_loss, val_loss, round(acc, 3),
                round(train_acc, 3)))
            f.write("\n")

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights-pretrain-coco-copy-move-B0/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.0001)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data1/zl/first/preTrain/coco_copy_move")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    parser.add_argument('--weights', type=str,
                        default='/data1/zl/first/model_efficientNet/efficientnetb0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
