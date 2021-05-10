import torch
import torchvision
import torch.nn as nn
from Lenet.model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # datasets导入我们的训练集 ，训练集的每个图像根据transform预处理
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)# transform函数会预处理
    # DataLoader   batch_size=36每一批36张图片进行训练  shuffle=True为True意思是是否是随机提取出来的
    # num_workers=0  载入数据的一个线程数  windous必须设置为0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)
    # 测试集
    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)
    # iter是一个迭代器  将val_loader转化为可迭代的迭代器
    val_data_iter = iter(val_loader)
    # next()就可以获取到一些数据，图像和标签
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # # 显示图片
    # imshow(torchvision.utils.make_grid(images))
    # # 打印图片标签
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



    net = LeNet() # 自己定义的模型
    # CrossEntropyLoss
    loss_function = nn.CrossEntropyLoss() #定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  #定义优化器  使用Adam   net.parameters()我们需要训练的参数

    # 进入训练过程
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0  #用来累计训练过程中的损失
        # 在通过一个循环，来遍历啊我们训练集的样本
        # 返回每一批的数据data 返回每一批data对应的步数  step从0开始
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # 将数据分离成  inputs 图像  labels 图像对应的标签

            # zero the parameter gradients
            # 历史损失函数清零  为什么要没计算一个batch,就要调用一次optimizero_grad()？
            # 如果不清除历史梯度，就会对计算的历史梯度进行累加（通过这个特性你能变相实现一个很大的batch数值训练
            #
            optimizer.zero_grad()
            # forward + backward + optimize
            # 将网络输入图片，得到输出
            outputs = net(inputs)
            # 通过损失函数计算损失  outputs网络预测的值  labels标签值
            loss = loss_function(outputs, labels)
            # 将loss进行反向传播
            loss.backward()
            # optimizer.step()进行参数的更新
            optimizer.step()

            # print statistics
            # 每次计算完loss 之后将他累加到running_loss
            running_loss += loss.item()
            # 每隔五百步打印一次信息
            if step % 500 == 499:  # print every 500 mini-batches2
                with torch.no_grad():  # 在接下来的计算中不要计算每个节点的误差损失梯度 with 是上下文管理器
                                        # 如果不这样 会占用更多的算力 存储每个节点的损失梯度  占用更多的内存资源
                    # 在这个地方，进行正向传播
                    outputs = net(val_image)  # [batch, 10]
                    # 输出最大的index在那个地方   dim=1在维度1上寻找最大值  [1]代表只需要index值 只需要在那个地方并不需要值
                    predict_y = torch.max(outputs, dim=1)[1]
                    # predict_y预测的标签类别 和真实值val_label进行比较  sum()求和  求得本次测试过程中预测对了多少样本
                    # 计算得到的是一个tensor 不是一个数值  item()拿到对应的数值
                    # / val_label.size(0) 得到之后除以测试样本的数目  得到测试的准确率
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # 打印信息  epoch第几轮训练  step + 1 每一轮的多少步 running_loss / 500平均训练误差
                    # accuracy 测试样本的准确率
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    # net.state_dict(),网络所有的参数
    # save_path 保存的路径
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()