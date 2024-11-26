import os
import torch
import torch.nn as nn
from my_model import resnet101
from torchviz import make_dot
import hiddenlayer as hl
from torchsummary import summary
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    model_weight_path = "../weights/resnet101-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    net = resnet101()
    net.load_state_dict(torch.load(model_weight_path,weights_only=True))
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 172)
    net.to(device) # 先在cpu上把参数加载进模型再把模型放到gpu上

    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 替换为你的输入形状

    # 生成图
    output = net(dummy_input)
    dot = make_dot(output, params=dict(net.named_parameters()))

    # 保存图形
    dot.render("model_structure", format="png")
    # 构建图




if __name__ == '__main__':
    main()