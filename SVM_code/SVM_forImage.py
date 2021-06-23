'''
基于SVM的手写数字识别
数据集：trainingData中包含约2000个32*32的数据，每个数据大约有200个
       testData中包含约900个
'''
import numpy as np
from os import listdir
import SVM_kernel

# 将数据格式化处理为一个向量
def img2vector(path):
    returnVect = np.zeros((1, 1024))
    fr = open(path)
    # 数据是32*32的，转化为1*1024的
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

# 下载数据
def loadImages(path):
    hwLabels = []
    trainingFileList = listdir(path)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 文件名格式为 数字_位数
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (path, fileNameStr))
    return trainingMat, hwLabels

def testDigits(ktup=('rbf', 10)):
    # 训练集
    trianData, trainLabel = loadImages(r'E:\MachineLeaning_file\SVM_kernel\digits\trainingDigits')
    b, alphas = SVM_kernel.smop(trianData, trainLabel, 200, 0.0001, 10000, ktup)
    dataMat = np.mat(trianData); labelMat = np.mat(trainLabel).transpose()
    # 支持向量
    svInd = np.nonzero(alphas.A > 0)[0]
    svs = dataMat[svInd]
    labelsv = labelMat[svInd]
    print('there are support vector: {}'.format(np.shape(svs)[0]))
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = SVM_kernel.kernelTrans(svs, dataMat[i, :], ktup)
        # 核*alpha*yi + b
        predict = kernelEval.T * np.multiply(labelsv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(trainLabel[i]):
            errorCount += 1
    print('train error rate is: {}'.format(float(errorCount) / m))

    # 测试集
    testData, testLabel = loadImages(r'E:\MachineLeaning_file\SVM_kernel\digits\testDigits')
    dataMat = np.mat(testData); labelMat = np.mat(testLabel).transpose()
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = SVM_kernel.kernelTrans(svs, dataMat[i, :], ktup)
        # 核*alpha*yi + b
        predict = kernelEval.T * np.multiply(labelsv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(testLabel[i]):
            errorCount += 1
    print('train error rate is: {}'.format(float(errorCount) / m))

if __name__ == '__main__':
    testDigits()