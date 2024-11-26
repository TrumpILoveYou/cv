# 加载预训练的 SE-ResNet50
net = create_model('seresnet50', pretrained=True)
# 修改最后一层分类头
num_classes = 172  # 目标类别数
net.fc = nn.Linear(net.num_features, num_classes)