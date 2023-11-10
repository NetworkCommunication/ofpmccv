import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

# reward
def plotResult():
    # 加载保存的CSV数据
    returns = np.loadtxt("data/Driving style/result/defensive/reward0820.csv", delimiter=',')

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.show()

# plotResult()

def plotResultSmooth():
    # 加载保存的CSV数据
    returns = np.loadtxt("data/Driving style/result/defensive/reward0820.csv", delimiter=',')
    returnsSmooth = scipy.signal.savgol_filter(returns, 30, 1)
    # 绘制折线图
    plt.plot(returnsSmooth)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.show()

# plotResultSmooth()

def plotCombinedResults():
    # 加载保存的CSV数据
    returns_not_front = np.loadtxt("result/other/reward0820.csv", delimiter=',')
    returns_front = np.loadtxt("result/main/reward0820.csv", delimiter=',')

    # 平滑处理
    returns_smooth_not_front = scipy.signal.savgol_filter(returns_not_front, 30, 1)
    returns_smooth_front = scipy.signal.savgol_filter(returns_front, 30, 1)

    # 绘制折线图
    plt.plot(returns_smooth_not_front, label='Not Front Traffic', color='blue')
    plt.plot(returns_smooth_front, label='Front Traffic', color='red')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward")

    # 添加图例
    plt.legend()

    plt.show()


# plotCombinedResults()

# laneChangeNumber
def plotResultLCN():
    # 加载保存的CSV数据
    returns = np.loadtxt("result/main/numberOfLaneChanges0812_1.csv", delimiter=',')

    # 计算平均数
    # 读取CSV文件
    # df = pd.read_csv('result/notFrontTraffice/numberOfLaneChanges0810.csv', header=None)
    # average = df.mean().values[0]
    # print("average: ", average)

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("laneChangeNumber")
    plt.title("laneChangeNumber vs Episode")
    plt.show()

# plotResultLCN()

def plotResultSmoothLCN():
    # 加载保存的CSV数据
    returns = np.loadtxt("result/main/numberOfLaneChanges0812_1.csv", delimiter=',')
    returnsSmooth = scipy.signal.savgol_filter(returns, 30, 1)
    # 绘制折线图
    plt.plot(returnsSmooth)
    plt.xlabel("Episode")
    plt.ylabel("laneChangeNumber")
    plt.title("laneChangeNumber")
    plt.show()

# plotResultSmoothLCN()

def plotCombinedLCNResults():
    # 加载保存的CSV数据
    returns_not_front = np.loadtxt("result/numberOfLaneChanges0820.csv", delimiter=',')
    returns_front = np.loadtxt("result/main/numberOfLaneChanges0820.csv", delimiter=',')

    # 平滑处理
    returns_smooth_not_front = scipy.signal.savgol_filter(returns_not_front, 30, 1)
    returns_smooth_front = scipy.signal.savgol_filter(returns_front,30, 1)

    # 绘制折线图
    plt.plot(returns_smooth_not_front, label='Not Front Traffic', color='blue')
    plt.plot(returns_smooth_front, label='Front Traffic', color='red')

    plt.xlabel("Episode")
    plt.ylabel("laneChangeNumber")
    plt.title("Lane Change Number vs Episode")

    # 添加图例
    plt.legend()

    plt.show()

# plotCombinedLCNResults()

# LC选取点
def plotLCNSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("result/numberOfLaneChanges0820.csv", delimiter=',')
    data_file2 = np.loadtxt("result/main/numberOfLaneChanges0820.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 250
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='notFrontTraffice Average', color='blue')
    plt.plot(average_values_file2, marker='o', label='normal Average', color='red')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotLCNSec()

# speed
def plotSpeedResult():
    # 加载保存的CSV数据
    returns = np.loadtxt("result/main/speedControl0812_1.csv", delimiter=',')

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("t")
    plt.ylabel("Last Speed")
    plt.title("Last Speed")
    plt.show()

# plotSpeedResult()

def plotSpeedSelResult():
    returns = np.loadtxt("result/main/speedControl0812_1.csv", delimiter=',')
    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 1
    num_groups = len(returns) // group_size
    grouped_data_file = np.array_split(returns, num_groups)

    # 计算每组数据的平均值
    average_values_file = [group.mean() for group in grouped_data_file]

    # 绘制折线图
    plt.plot(average_values_file, marker='o', label='notFrontTraffice Average', color='blue')

    plt.xlabel("Group")
    plt.ylabel("Average Speed")
    plt.title("Average Speed per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotSpeedSelResult()

# speed选取点
def plotSpeedSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("result/speedControl0820.csv", delimiter=',')
    data_file2 = np.loadtxt("result/main/speedControl0820.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 1
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='notFrontTraffice speed', color='blue')
    plt.plot(average_values_file2, marker='o', label='normal speed', color='red')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

plotSpeedSec()

# TTC选取点
def plotTtcSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("result/TTC0820.csv", delimiter=',')
    data_file2 = np.loadtxt("result/main/TTC0820.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 1
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='notFrontTraffice ttc', color='blue')
    plt.plot(average_values_file2, marker='o', label='normal ttc', color='red')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

plotTtcSec()


def plotMathStatisticalModel():
    # 定义sigmoid函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # 定义Tanh函数
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # 定义ReLU函数
    def relu(x):
        return np.maximum(0, x)
    def Softplus(x):
        return np.log(1 + np.exp(x))

    # 定义一个简单的三次多项式函数作为示例
    def polynomial_function(x):
        return 0.1 * x ** 3 + 2 * x ** 2 - 5 * x + 10


    # 创建x轴上的数据点
    x = np.linspace(-10, 10, 2000)

    # 计算对应的y值
    # y = relu(x)

    y = tanh(x)

    # 进行min-max归一化
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

    plt.subplot(1, 2, 2)
    plt.plot(x, y_normalized)
    plt.xlabel('x')
    plt.ylabel('Normalized y')
    plt.title('Normalized Function')
    plt.grid(True)

    plt.tight_layout()

    # 绘制图像
    # plt.plot(x, y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Function')
    plt.grid(True)
    plt.show()

# plotMathStatisticalModel()