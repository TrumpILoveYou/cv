import os
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from my_model import resnet101, resnet152
from my_dataset import MyDataset
from torch.nn import functional as F
from timm import create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_dataset = MyDataset(file_list="../SplitAndIngreLabel/TR.txt", data_root="./ready_chinese_food", transform_type="train")
    train_num = len(train_dataset)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=nw)

    validate_dataset = MyDataset(file_list="../SplitAndIngreLabel/VAL.txt", data_root="./ready_chinese_food", transform_type="val")
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size, shuffle=False, num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    """
        net = resnet152(num_classes=172, include_top=True)
        # load pretrain weights
    
    
        model_weight_path = "./weights/resnet152.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    """

    # 加载预训练的 ResNet152
    net = create_model('resnet152', pretrained=False)
    # 手动加载权重
    checkpoint_path = '../weights/resnet152.pth'
    net.load_state_dict(torch.load(checkpoint_path))

    # 修改最后一层分类头
    net.fc = nn.Linear(net.num_features, 172)

    for param in net.parameters():
        param.requires_grad = False

    for param in net.layer4.parameters(): # 解冻layer4层
        param.requires_grad = True
    for layer in net.fc.parameters():  # 解冻fc层
        layer.requires_grad = True
    net.to(device)


    def smooth_crossentropy(pred, gold, smoothing=0.1):
        n_class = pred.size(1)

        one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
        log_prob = F.log_softmax(pred, dim=1)

        return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

    # define loss function
    loss_function = smooth_crossentropy


    # 定义优化器
    optimizer = torch.optim.Adam([
        {'params': net.layer4.parameters(), 'lr': 1e-4},  # layer4 使用较低学习率
        {'params': net.fc.parameters(), 'lr': 1e-3}  # fc 使用较高学习率
    ], weight_decay=1e-4)  # 设置权重衰减

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)


    epochs = 10
    best_acc = 0.0
    save_path = '../weights/resNet152-ld.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.long().to(device)).mean()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_running_loss = 0.0  # Accumulate validation loss
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                val_loss = loss_function(outputs, val_labels.long().to(device)).mean()
                val_running_loss += val_loss.item()

                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

        val_accurate = acc / val_num
        avg_val_loss = val_running_loss / len(validate_loader)
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, avg_val_loss, val_accurate))

        # Update the learning rate using the scheduler
        scheduler.step(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

def mytest():
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = MyDataset(file_list="../SplitAndIngreLabel/TE.txt", data_root="./ready_chinese_food", transform_type="val")
    test_num = len(test_dataset)
    validate_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,batch_size=batch_size,num_workers=nw)
    print("using {} images for testing.".format(test_num))


    # 加载预训练的 SE-ResNet50
    net = create_model('seresnet50', pretrained=False)
    # 修改最后一层分类头
    num_classes = 172  # 目标类别数
    net.fc = nn.Linear(net.num_features, num_classes)

    # load pretrain weights
    model_weight_path = "../weights/resnet152-ld.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    net.eval()
    acc = 0.0
    total_loss = 0.0  # 用于累积总损失
    with torch.no_grad():
        test_bar = tqdm(validate_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)  # 将数据移到相同的设备上
            outputs = net(test_images.to(device))
            loss = loss_function(outputs, test_labels.long())
            total_loss += loss.item()  # 累加损失
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            test_bar.desc = "Testing"

    test_accurate = acc / test_num
    avg_loss = total_loss / len(validate_loader)
    print('avg_loss: %.3f     test_accuracy: %.3f' % (avg_loss,test_accurate))


if __name__ == '__main__':
    mytest()