import numpy as np
import SMO
import matplotlib.pyplot as plt

class optStruct:
    def __init__(self, dataMatIn, classLabels, c, toloer):
        self.x = dataMatIn
        self.labelMat = classLabels
        self.c = c
        self.tol = toloer
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 缓存误差
        self.ecache = np.mat(np.zeros((self.m, 2)))

# 算误差
def calcEK(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*\
                (oS.x*oS.x[k, :].T)) + oS.b
    ek = fXk - float(oS.labelMat[k])
    return ek

# 选第二个向量
def selectJ(i, oS, ei):
    maxK = -1;
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

def updateEK(oS, k):
    ek = calcEK(oS, k)
    oS.ecache[k] = [1, ek]

# 检查alphaj是否在(0~c)上
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
        eta = 2.0*os.x[i, :]*os.x[j, :].T - os.x[i, :]*os.x[i, :].T -\
                os.x[j, :]*os.x[j, :].T
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
        b1 = os.b - ei - os.labelMat[i]*(os.alphas[i] - alphaIold)*\
            os.x[i, :]*os.x[i, :].T - os.labelMat[j]*\
             (os.alphas[j] - alphaJold)*os.x[i, :]*os.x[j, :].T
        b2 = os.b - ej - os.labelMat[i]*(os.alphas[i] - alphaIold)*\
            os.x[i, :]*os.x[j, :].T - os.labelMat[j]*\
             (os.alphas[j] - alphaJold)*os.x[j, :]*os.x[j, :].T
        if (0 < os.alphas[i]) and (os.c > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.c > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smop(data, label, c, toler, maxIter, ktup=('lin', 0)):
    os = optStruct(np.mat(data), np.mat(label).transpose(), c, toler)
    iter = 0
    entireSet = True
    alphapairsChanged = 0
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

# 求出w
def clacWs(alphas, dataArr, classLabels):
    x = np.mat(dataArr)
    label = np.mat(classLabels).transpose()
    m, n = np.shape(x)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*label[i], x[i, :].T)
    return w

# 绘出曲线
def show_(data, label, w, b):
    b = float(b[:1])
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
        temp = -(w.T[0][0] / w.T[0][1])*i - (b / w.T[0][1])
        y.append(temp)
    plt.plot(x, y)
    plt.scatter(x1, y1, s=30, c='r', marker='s', alpha=0.3)
    plt.scatter(x2, y2, s=30, c='b', alpha=0.3)
    plt.show()

if __name__ == '__main__':
    data, label = SMO.loadDataSet(r'E:\MachineLeaning_file\SMO\testSet.txt')
    b, alpha = smop(data, label, 0.6, 0.001, 40)
    w = clacWs(alpha, data, label)
    show_(data, label, w, b)