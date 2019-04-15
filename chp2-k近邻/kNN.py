from numpy import *
import operator
import os


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
2-1
classify0(input_X, train_data_X, train_data_label, k)通过k近邻计算后输出样本input_X的标记
input_X：测试样本的特征
train_data_X：训练样本的特征
train_data_label：训练样本的标记
k：考虑的近邻的个数
'''


def classify0(input_X, train_data_X, train_data_labels, k):
    # 距离计算
    datasetSize = train_data_X.shape[0]
    diffMat = tile(input_X, (datasetSize, 1)) - train_data_X
    diffMat_sq = diffMat ** 2
    diffMat_sq_sum = sum(diffMat_sq, axis=1)
    distance = diffMat_sq_sum ** 0.5

    # 选择距离最小的k个点
    distance_sorted_index = distance.argsort()
    classCount = {}
    for i in range(k):
        label = train_data_labels[distance_sorted_index[i]]
        classCount[label] = classCount.get(label, 0) + 1

    # 排序
    sorted_classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classCount[0][0]


'''
2-2
file2matrix(filename)将文本记录转换为numpy的train_data_feature和train_data_labels，并返回
'''


def file2matrix(filename):
    file = open(filename)
    filelines = file.readlines()
    train_data_num = len(filelines)

    train_data_feature = zeros((train_data_num, 3))  # 这里直接设为3是因为已知样本的特征为3维
    train_data_labels = []
    index = 0
    for line in filelines:
        values = line.strip().split('\t')
        train_data_feature[index, :] = values[0:3]
        train_data_labels.append(int(values[-1]))
        index += 1
    return train_data_feature, train_data_labels


'''
2-3
autoNorm(dataSet)对特征值进行归一化
返回归一化后的特征值矩阵，每个特征维度的原始取值范围差和最小值
'''


def autoNorm(dataSet):
    min_features = dataSet.min(0)  # 每列的最小值
    max_features = dataSet.max(0)
    ranges = max_features - min_features

    normDataset = zeros(dataSet.shape)
    n = dataSet.shape[0]  # 样本数量
    normDataset = dataSet - tile(min_features, (n, 1))
    normDataset = normDataset / tile(ranges, (n, 1))
    return normDataset, ranges, min_features


'''
2-4
datingClassTest() 分类器针对约会网站的测试代码
'''


def datingClassTest():
    dataset_X, dataset_labels = file2matrix('datingTestSet2.txt')
    dataset_X, dataset_X_ranges, dataset_X_min = autoNorm(dataset_X)
    testRatio = 0.1  # 测试集在整个数据集的占比
    dataset_num = dataset_X.shape[0]
    test_num = int(testRatio * dataset_num)
    error_count = 0
    for i in range(test_num):
        predict_label = classify0(dataset_X[i, :], dataset_X, dataset_labels, 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (predict_label, dataset_labels[i]))
        if predict_label != dataset_labels[i]:
            error_count += 1
    print("the total error rate is: %f" % (error_count / float(test_num)))


'''
2-5
classifyPerson()约会网站预测函数
'''


def classifyPerson():
    resuleList = ['not at all', 'in small doses', 'in large doses']
    dataset_X, dataset_labels = file2matrix('datingTestSet2.txt')
    dataset_X, dataset_X_ranges, dataset_X_min = autoNorm(dataset_X)
    percentGametime = float(input("percentage of time spent on playing video game?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    sample = array([percentGametime, ffMiles, iceCream])
    k = 3
    classifierResult = classify0((sample - dataset_X_min) / dataset_X_ranges, dataset_X, dataset_labels, k)
    print("You will probably like this person: ", resuleList[classifierResult - 1])


'''
img2vector(filename) 将32*32的矩阵转为1*1024的向量
'''


def img2vector(filename):
    file = open(filename)
    res = zeros((1, 1024))
    for i in range(32):
        fileline = file.readline()
        for j in range(32):
            res[0][32 * i + j] = int(fileline[j])
    return res


'''
手写体数字识别系统的测试代码
'''


def handwirtingClassTest():
    k = 3
    train_labels = []
    train_file = os.listdir('trainingDigits')
    train_num = len(train_file)
    train_features = zeros((train_num, 1024))
    for i in range(train_num):
        fileNameStr = train_file[i]
        fileNameStr_s = fileNameStr.split('.')[0]
        label_temp = int(fileNameStr_s.split('_')[0])
        train_labels.append(label_temp)
        train_features[i][:] = img2vector('trainingDigits/%s' % fileNameStr)

    test_file = os.listdir('testDigits')
    test_num = len(test_file)
    error_count = 0
    for i in range(test_num):
        fileNameStr = test_file[i]
        fileNameStr_s = fileNameStr.split('.')[0]
        real_label = int(fileNameStr_s.split('_')[0])
        img_test = img2vector('testDigits/%s' % fileNameStr)
        predict_label = classify0(img_test, train_features, train_labels, k)
        print("the classifier came back with: %d, the real label is: %d" %
              (predict_label, real_label))
        if predict_label != real_label:
            error_count += 1
    print("the total number of errors is: %d" % (error_count))
    print("the total error rate is: %f" % (float(error_count / test_num)))
