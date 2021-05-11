import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # in_channels参数代表输入特征矩阵的深度即channel，比如输入一张RGB彩色图像，那in_channels=3
            # out_channels参数代表卷积核的个数，使用n个卷积核输出的特征矩阵深度即channel就是n
            # kernel_size参数代表卷积核的尺寸，输入可以是int类型如3 代表卷积核的height=width=3，
            # 也可以是tuple类型如(3, 5)代表卷积核的height=3，width=5
            # stride参数代表卷积核的步距默认为1，和kernel_size一样输入可以是int类型，也可以是tuple类型
            # padding参数代表在输入特征矩阵四周补零的情况默认为0，同样输入可以为int型如1 代表上下方向各补一行0元素，
            # 左右方向各补一列0像素（即补一圈0），如果输入为tuple型如(2, 1) 代表在上方补两行下方补两行，左边补一列，右边补一列。
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            # inplace可以载入更大的模型  ReLU激活函数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            # 代表随机失活的比例
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # 输出类别
            nn.Linear(2048, num_classes),
        )
        # 初始化权重的选项 init_weights为true  进入初始化
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # torch.flatten展平处理
        x = torch.flatten(x, start_dim=1)
        # classifier输入到分类结构中
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # self.modules()遍历这个模块  迭代网络的每一层结构  判断m是否是nn.Conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal初始化变量方法  对m.weight权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果bias不为空的话  那么就用0对他初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # normal_通过正态分布对权重复制
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)