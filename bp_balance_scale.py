'''
胡开智 222021335210071
date: 11/16/2022
IDE:Visual Studio Code
Python:3.9
'''
# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from pandas.plotting import radviz

# 1.初始化参数
def initialize_parameters(nx, nh, ny):
    # 初始化权重和偏置矩阵
    # 初始设定参数没有范围，直接采用随机数生成
    w1 = np.random.randn(nh, nx) * 0.01
    b1 = np.zeros(shape=(nh, 1))
    w2 = np.random.randn(ny, nh) * 0.01
    b2 = np.zeros(shape=(ny, 1))
    # 用字典存储
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters


# 2.前向传播
def forward_propagation(x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    # 前向传播的计算
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)  # 先使用tanh作为激活函数
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 再使用sigmoid作为激活函数
    # 用字典存储
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache


# 3.计算损失函数
def comcost(a2, y):
    # 得到样本总数
    m = y.shape[1]
    # 若利用均方误差损失函数，当起始输出值较大时，整个梯度更新幅度都比较小，收敛时间会很长
    # 而交叉熵损失只与输出值和真实值的差值成正比，此时收敛较快，故采用交叉熵
    entropy = np.multiply(np.log(a2), y) + np.multiply((1 - y), np.log(1 - a2))
    cost = - np.sum(entropy) / m
    return cost

# 4.反向传播
def backward_propagation(parameters, cache, x, y):
    m = y.shape[1]
    w2 = parameters['w2']
    a1 = cache['a1']
    a2 = cache['a2']
    # 反向传播的计算
    # 根据链式求导法则，可以求出输出单元以及隐藏单元的误差项
    dz2 = a2 - y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate=0.4):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']
    # 更新参数
    w1 -= dw1 * learning_rate
    b1 -= db1 * learning_rate
    w2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters


# 6.建立神经网络
def bp_model(x, y, n_h, n_input, n_output, iterations=10000, print_cost=False):
    np.random.seed(2)
    nx = n_input  # 输入层的节点数
    ny = n_output  # 输出层的节点数
    nh = n_h  # 中间一层隐藏层的节点数
    loss = []
    # 初始化参数
    parameters = initialize_parameters(nx, nh, ny)
    # 梯度下降过程
    for i in range(0, iterations):
        lossnum = 0
        # 向前传播
        a2, cache = forward_propagation(x, parameters)
        # 计算代价函数
        cost = comcost(a2, y)
        lossnum += cost
        # 反向传播
        grads = backward_propagation(parameters, cache, x, y)
        # 更新参数
        parameters = update_parameters(parameters, grads)
        # 每1000次迭代,输出一次代价函数
        if print_cost and i % 1000 == 0 and i != 0:
            loss.append(lossnum)
            print("train iterations[{}/{}]   cross-entropy loss:{:.3f}".
                  format(i, iterations, cost))
    # 绘制训练时的loss曲线
    plt.figure("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(loss, label="$Loss$")
    plt.legend()
    plt.show()
    return parameters


# 7.模型评估/预测
def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))
    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]
    # 存储预测值的结果
    output = np.empty(shape=(n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0
    return output


# 8.预测结果可视化
def visualization(x_test, y_test, result):
    cols = y_test.shape[1]
    # 将编码转换为类别
    y = []
    pre = []
    for i in range(cols):
        if y_test[0][i] == 0 and y_test[1][i] == 0 and y_test[2][i] == 1:
            y.append('B')
        elif y_test[0][i] == 0 and y_test[1][i] == 1 and y_test[2][i] == 0:
            y.append('R')
        elif y_test[0][i] == 1 and y_test[1][i] == 0 and y_test[2][i] == 0:
            y.append('L')
    for j in range(cols):
        if result[0][j] == 0 and result[1][j] == 0 and result[2][j] == 1:
            pre.append('B')
        elif result[0][j] == 0 and result[1][j] == 1 and result[2][j] == 0:
            pre.append('R')
        elif result[0][j] == 1 and result[1][j] == 0 and result[2][j] == 0:
            pre.append('L')
        else:
            pre.append('unknown')

    # 将特征值与类别拼接起来
    real = np.column_stack((x_test.T, y))
    prediction = np.column_stack((x_test.T, pre))
    # 将其转换为DataFrame类型, 并添加columns
    df_real = pd.DataFrame(real, index=None,
                           columns=['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance', 'Class-Name'])
    df_prediction = pd.DataFrame(prediction, index=None,
                                 columns=['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance',
                                          'Class-Name'])
    # 为了防止radviz报错, 将特征列转换为float类型
    df_real[['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']] = df_real[
        ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']].astype(float)
    df_prediction[['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']] = df_prediction[
        ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']].astype(float)
    # 绘图
    plt.figure("The true classification")
    radviz(df_real, 'Class-Name', color=['red', 'green', 'blue', 'black'])
    plt.figure('The predictive classification')
    radviz(df_prediction, 'Class-Name', color=['red', 'green', 'blue', 'black'])
    plt.show()


    # 混淆矩阵以及热度图
    sns.set()
    con_matrix = confusion_matrix(y, pre, labels=["B", "R", "L"])
    df = pd.DataFrame(con_matrix, index=["B", "R", "L"], columns=["B", "R", "L"])
    ax = sns.heatmap(df, annot=True)
    ax.set_xlabel("predict")
    ax.set_ylabel("true")
    plt.show()
    # 分类报告
    print(metrics.classification_report(y, pre, zero_division=True))


# 9.编写主函数
def main():
    # 读取数据
    data_set = pd.read_csv('balancescale.csv', header=None)
    x = data_set.iloc[:, 0:4].values
    y = data_set.iloc[:, 4:].values
    y = y.astype('uint8')
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # 训练模型
    # 输入4个节点，隐层10个节点，输出3个节点，迭代10000次
    parameters = bp_model(x_train.T, y_train.T, n_h=10, n_input=4, n_output=3, iterations=10000, print_cost=True)
    # 对模型进行测试
    result = predict(parameters, x_test.T, y_test.T)
    # 预测分类结果可视化
    visualization(x_test.T, y_test.T, result)


# 主函数
if __name__ == "__main__":
    main()
