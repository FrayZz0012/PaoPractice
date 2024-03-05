import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


# 处理测试数据
def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                             download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)

    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定训练所用的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将模型放到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征和标签放入设备中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            # 将模型设置为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的下标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，括号内为真，准确度加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)
    # 计算准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == "__main__":
    # 加载模型
    model = LeNet()
    # 将训练好的参数加载到模型中
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据集
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()

            print("预测值：", classes[result], "--------", "真实值", classes[label])


