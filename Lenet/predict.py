import torch
import torchvision.transforms as transforms
from PIL import Image

from Lenet.model import LeNet


def main():
    # 预处理 将图像转化为32x32的大小 然后转化为tensor  transforms.Normalize标准化处理 因为训练的过程中做了标准化处理
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    # 载入我们保存的权重文件
    net.load_state_dict(torch.load('Lenet.pth'))
    # 载入图像
    im = Image.open('1.jpg')
    # 图像进行预处理
    im = transform(im)  # [C, H, W] 深度 高度 宽度
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W] unsqueeze()增加一个新的维度 dim=0代表在最前面添加   N 代表banch

    with torch.no_grad():  # 代表不需要求损失梯度
        outputs = net(im) # 将图片送到网络中得到输出
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()