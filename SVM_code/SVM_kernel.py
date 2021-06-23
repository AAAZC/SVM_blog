'''
核函数应用
其实加入核函数只是对原来再二维平面上的数据进行了一次优化
它与SMO的思维是一样的，我们还是需要求出支持向量，但是这次分类的算法变成了kernel*alpha*yi+b
这个kernel就是核函数，它是根据问题的不同而选择的变种，其核心在于我们还是在原来的维度上求出了w和b
'''
import matplotlib.pyplot as plt
import numpy as np
import SMO

# 绘图
def show_(data1, label1, data2, label2):
    #设置字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams['axes.unicode_minus'] = False
    fig, axis = plt.subplots(ncols=2, sharex=False, sharey=False, figsize=(10, 5))
    # 训练集数据
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(len(data1)):
        if label1[i] == 1.:
            x1.append(data1[i][0])
            y1.append(data1[i][1])
        elif label1[i] == -1.:
            x2.append(data1[i][0])
            y2.append(data1[i][1])
    axis[0].set_title("Train")
    axis[0].scatter(x1, y1, s=30, c='r', marker='s', alpha=0.3)
    axis[0].scatter(x2, y2, s=30, c='b', alpha=0.3)
    #测试集数据
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(len(data2)):
        if label2[i] == 1.:
            x1.append(data2[i][0])
            y1.append(data2[i][1])
        elif label2[i] == -1.:
            x2.append(data2[i][0])
            y2.append(data2[i][1])
    axis[1].set_title("Test")
    axis[1].scatter(x1, y1, s=30, c='r', marker='s', alpha=0.3)
    axis[1].scatter(x2, y2, s=30, c='b', alpha=0.3)

    plt.show()

# 结构化
class optStruct:
    # (data, label, 松弛变量, 迭代次数, 核函数元组)
    def __init__(self, data, label, c, toler, ktup):
        self.x = data
        self.labelMat = label
        self.c = c
        self.tol = toler
        self.m = np.shape(data)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.ecache = np.mat(np.zeros((self.m, 2)))
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernelTrans(self.x, self.x[i, :], ktup)

# 核函数选择，这里的lin、rbf只是一种选择，可以通过elif增加更多的核函数
def kernelTrans(x, a, ktup):# (data, data[i, :], 元组)
    m, n = np.shape(x)
    k = np.mat(np.zeros((m, 1)))
    # 元组存放了核类型
    if ktup[0] == 'lin':
        k = x * a.T
    elif ktup[0] == 'rbf':
        for j in range(m):
            deltaRow = x[j, :] - a
            k[j] = deltaRow * deltaRow.T
        # 高斯核函数
        k = np.exp(k / (-1*ktup[1]**2))
    else:
        # 报错
        raise NameError('Houston we have a problem -- That Kernerl is not recognized')
    return k

# 存误差
def calcEK(os, k):
    fxk = float(np.multiply(os.alphas, os.labelMat).T * os.k[:, k] + os.b)
    ek = fxk - float(os.labelMat[k])
    return ek

# 选出最大步长的alphaj
def selectJ(i, oS, ei):
    maxK = -1
    maxDeltaE = 0; ej = 0
    oS.ecache[i] = [1, ei]
    # nonzero以数组的形式返回序列非零元素的索引
    validEcacheList = np.nonzero(oS.ecache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            ek = calcEK(oS, k)
            deltaE = np.abs(ei- ek)
            if deltaE > maxDeltaE:
                maxK = k
                # 找最大步长
                maxDeltaE = deltaE
                ej = ek
        return maxK, ej
    else:
        j = SMO.selectJrand(i, oS.m)
        ej = calcEK(oS, j)
    return j, ej

# 更新误差
def updateEK(oS, k):
    ek = calcEK(oS, k)
    oS.ecache[k] = [1, ek]

# 对向量进行优化
def innerL(i, os):
    ei = calcEK(os, i)
    if ((os.labelMat[i]*ei < -os.tol) and (os.alphas[i] < os.c)) or\
            ((os.labelMat[i]*ei > os.tol) and (os.alphas[i] > 0)):
        j, ej = selectJ(i, os, ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        # 计算范围
        if os.labelMat[i] != os.alphas[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.c)
            H = min(os.c, os.alphas[j] + os.alphas[i])
        if L == H:
            print('L == H')
            return 0
        # 这个地方是与SMO_2中不同的
        eta = 2.0*os.k[i, j] - os.k[i, i] - os.k[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        os.alphas[j] -= os.labelMat[j]*(ei - ej) / eta
        os.alphas[j] = SMO.clipAlpha(os.alphas[j], H, L)
        updateEK(os, j)
        if abs(os.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough')
            return 0
        os.alphas[i] += os.labelMat[j]*os.labelMat[i]*\
                        (alphaJold - os.alphas[j])
        updateEK(os, i)
        # 这个地方是与SMO_2中不同的
        b1 = os.b - ei - os.labelMat[i]*(os.alphas[i] - alphaIold)*os.k[i, i]-\
            os.labelMat[j]*(os.alphas[j] - alphaJold)*os.k[i, j]
        b2 = os.b - ej - os.labelMat[i]*(os.alphas[i] - alphaIold)*os.k[i, j]-\
            os.labelMat[j]*(os.alphas[j] - alphaJold)*os.k[j, j]

        if (0 < os.alphas[i]) and (os.c > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.c > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

# 外循环程序
def smop(data, label, c, toler, maxIter, ktup=('lin', 0)):
    os = optStruct(np.mat(data), np.mat(label).transpose(), c, toler, ktup)
    iter = 0
    entireSet = True
    alphapairsChanged = 0
    # 迭代、设置检查点
    while (iter < maxIter) and ((alphapairsChanged > 0) or (entireSet)):
        alphapairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphapairsChanged += innerL(i, os)
            print('fullset, iter: {}, i: {}, paris changed: {}'.format(iter, i, alphapairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            for i in nonBoundIs:
                alphapairsChanged += innerL(i, os)
                print('non-bound, iter: {}, i: {}, pairs changed: {}'.format(iter, i, alphapairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphapairsChanged == 0):
            entireSet = True
        print('iteration number: {}'.format(iter))
    return os.b, os.alphas

# 设定sigma为1.3
def testRbf(k1 = 1.3):
    # 训练部分
    traindata, trainlabel = SMO.loadDataSet(path = r"E:\MachineLeaning_file\SVM_kernel\testSetRBF.txt")
    b, alphas = smop(traindata, trainlabel, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(traindata); labelMat = np.mat(trainlabel).transpose()
    # mat.A为矩阵转数组类型
    # 找支持向量
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are {} support vectors".format(np.shape(sVs)[0]))
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        # 核*alpha*yi + b
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(trainlabel[i]):
            errorCount += 1
    print('train error rate is: {}'.format(float(errorCount) / m))
    # 测试部分
    testdata, testlabel = SMO.loadDataSet(path = r"E:\MachineLeaning_file\SVM_kernel\testSetRBF2.txt")
    dataMat = np.mat(testdata); labelMat = np.mat(testlabel).transpose()
    # 绘图
    show_(traindata, trainlabel, testdata, testlabel)
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(testlabel[i]):
            errorCount += 1
    print('test error rate is: {}'.format(float(errorCount) / m))

if __name__ == '__main__':
    testRbf()