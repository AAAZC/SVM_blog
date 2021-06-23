import numpy as np
import matplotlib.pyplot as plt

#导入
def loadDataSet(path):
    dataMet = []; labelMet = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMet.append([float(lineArr[0]), float(lineArr[1])])
        labelMet.append(float(lineArr[2]))
    return dataMet, labelMet

# 随机抽取第二个alpha
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j

# 确定alphaj的范围
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def show_(data, label, w1, w2, b, alpha_x):
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(len(label)):
        if label[i] == 1.0:
            x1.append(data[i][0])
            y1.append(data[i][1])
        elif label[i] == -1.:
            x2.append(data[i][0])
            y2.append(data[i][1])
    #作回归曲线
    x = range(10)
    y = []
    for i in range(10):
        temp = -(w1 / w2)*i - (b / w2)
        y.append(temp)
    plt.plot(x, y)
    # 绘出支持向量
    for i in range(len(alpha_x)):
        plt.scatter(alpha_x[i][0], alpha_x[i][1], s=100, c='black', marker='x')
    plt.scatter(x1, y1, s=30, c='r', marker='s', alpha=0.3)
    plt.scatter(x2, y2, s=30, c='b', alpha=0.3)
    plt.show()

# SMO
def smoSimple(data, label, c, toler, maxIter):# 数据、标签、松弛变量、容错率、迭代次数
    dataMatrix = np.mat(data); labelMat = np.mat(label).transpose()
    b = 0; m, n = np.shape(dataMatrix)
    # 初始化：假设所有点都满足约束条件
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 超平面方程ayx*x+b: 求是否在超平面上
            fXi = float(np.multiply(alphas, labelMat).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算误差
            ei = fXi - float(labelMat[i])
            # 取出满足误差条件和松弛范围的
            if (labelMat[i]*ei < -toler) and (alphas[i] < c) or \
                    (label[i]*ei > toler) and (alphas[i] > 0):
                # 随机找第二个支持向量
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T * \
                            (dataMatrix * dataMatrix[j, :].T)) + b
                ej = fxj - float(labelMat[j])
                # 记录优化前的
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alphai alphaj在0~c内
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = min(c, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                # eta是alphaj的最优修改量
                # 即：alphaj如果选的跟alphai一个位置或者比alphai更靠近约束条件，就说明不用再优化了
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - \
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 重新算一个alphaj
                alphas[j] -= labelMat[j] * (ei - ej) / eta
                # 判断区间
                alphas[j] = clipAlpha(alphas[j], H, L)
                # j改变的太少了，不算
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j is moving enough")
                    continue
                # 为了满足约束，一个增加一个减少
                alphas[i] += labelMat[j] * labelMat[i] *\
                             (alphaJold - alphas[j])
                b1 = b - ei - labelMat[i] * (alphas[i] - alphaIold)*\
                    dataMatrix[i, :]*dataMatrix[i, :].T -\
                    labelMat[j] * (alphas[j] - alphaJold)*\
                    dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - ej - labelMat[i] * (alphas[i] - alphaIold)*\
                    dataMatrix[i, :]*dataMatrix[j, :].T -\
                    labelMat[j] * (alphas[j] - alphaJold)*\
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.
                alphaPairsChanged += 1
                print("iter:{} i:{}, pairs change {}".format(iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number = {}".format(iter))
    return b, alphas

def handle_data(b, alpha, data, label):
    b = float(b[:1])
    alpha_label = []
    alpha_x = []
    new_alpha = []
    w1 = .0
    w2 = .0
    r = 0
    for i in range(100):
        if alpha[i] > 0:
            alpha_label.append(label[i])
            alpha_x.append(data[i])
            new_alpha.append(alpha[i])
            r += 1
            print(data[i], label[i], sep='  ')
    for i in range(r):
        w1 += alpha_label[i] * new_alpha[i] * alpha_x[i][0]
        w2 += alpha_label[i] * new_alpha[i] * alpha_x[i][1]
    w1 = float(w1[:1])
    w2 = float(w2[:1])
    return b, w1, w2, alpha_x

if __name__ == "__main__":
    data, label = loadDataSet(r'E:\MachineLeaning_file\SMO\testSet.txt')
    b, alpha = smoSimple(data, label, 0.6, 0.001, 40)
    print(b, alpha[alpha > 0], sep='\n')
    b, w1, w2, alpha_x = handle_data(b, alpha, data, label)
    show_(data, label, w1, w2, b, alpha_x)