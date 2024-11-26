import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from my_model import resnet101

from src.my_dataset import  create_food_label_dict


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./ready_chinese_food/3/1_2.jpg" # 改成需要测的图片
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict

    class_indict = create_food_label_dict("../SplitAndIngreLabel/FoodList.txt")

    # create model
    model = resnet101(num_classes=172).to(device)

    # load model weights
    weights_path = "../weights/resNet101.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()
        print(predict_cla)

    print_res = "class: {}   Prob: {:.3}".format(class_indict[predict_cla],predict[predict_cla].numpy())
    print(print_res)
    plt.title(print_res)
    plot_top_10_predictions(predict, class_indict)
    for i in range(len(predict)):
        print("prob: {:.3}  class: {:10}".format(predict[i].numpy(),class_indict[i]))
    plt.show()


def plot_top_10_predictions(predict, class_indict):
    """
    绘制预测的前10个类别及其对应的概率的条形图

    Args:
        predict (tensor): 预测的概率分布，大小为 [num_classes]
        class_indict (dict): 标签编号到标签名称的映射字典
    """
    # 获取预测的概率值，并按概率值排序
    top10_prob, top10_idx = torch.topk(predict, 10)  # 返回前10个最大值的概率和索引
    top10_prob = top10_prob.cpu().numpy()  # 将概率值从 tensor 转换为 numpy 数组
    top10_idx = top10_idx.cpu().numpy()  # 获取对应的标签索引

    # 获取前10个类别的名称
    top10_classes = [class_indict[idx] for idx in top10_idx]

    # 绘制条形图
    plt.figure(figsize=(18, 6))
    plt.barh(top10_classes, top10_prob, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Top 10 Predicted Classes and their Probabilities')
    plt.gca().invert_yaxis()  # 反转y轴，确保最大概率在上面
    plt.show()

if __name__ == '__main__':
    main()