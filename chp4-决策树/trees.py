from math import log
import operator

'''
3-1
def calcShannonEnt(dataSet)计算数据集的信息熵
'''


def calcShannonEnt(dataSet):
    num = len(dataSet)

    # 为所有可能的分类创建字典
    label_count = {}
    for i in range(num):
        label_temp = dataSet[i][-1]
        if label_temp not in label_count.keys():
            label_count[label_temp] = 0
        label_count[label_temp] += 1

    # 以2为底求对数
    calcShannonEnt = 0.0
    for key in label_count:
        prob_temp = float(label_count[key]) / num
        calcShannonEnt -= prob_temp * log(prob_temp, 2)

    return calcShannonEnt


'''
def creatDataSet() 自制toy数据集
'''


def creatDataSet():
    dataSet = [[1, 1, 'yes'], ['1', '1', 'yes'], ['1', '0', 'no'], ['0', '1', 'no'], ['0', '1', 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


'''
3-2
splitDataSet(dataset,axis,value) 按照给定特征划分数据集
'''


def splitDataSet(dataset, axis, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''
3-3
chooseBestFeatureToSplit(dataset)选择最好的数据集划分方式
'''


def chooseBestFeatureToSplit(dataset):
    featureNum = len(dataset[0]) - 1
    dataNum = len(dataset)
    baseEnt = calcShannonEnt(dataset)
    bestEntGain = 0.0
    bestFeature = -1
    for f in range(featureNum):
        # 创建唯一的分类标签列表
        featureList = [featVec[f] for featVec in dataset]
        featureList = set(featureList)

        # 计算每种划分方式的信息熵
        subTotalEnt = 0.0
        for feature in featureList:
            subDataSet = splitDataSet(dataset, f, feature)
            prob = float(len(subDataSet)) / dataNum
            subTotalEnt += prob * calcShannonEnt(subDataSet)

        # 计算最好的信息增益
        gainEnt = baseEnt - subTotalEnt
        if gainEnt > bestEntGain:
            bestEntGain = gainEnt
            bestFeature = f

    return bestFeature


'''
majorityClass(classList)返回多数表决的结果
'''


def majorityClass(classList):
    classCount = {}
    for c in classList:
        if c not in classCount.keys():
            classCount[c] = 0
        classCount[c] += 1
    sortedClassCount = sorted(classCount, key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
3-4
createTree(dataset, featureLabels)创建决策树
'''


# 构建树
def createTree(dataset, featureLabels):

    #  当前数据集中样本的类别完全相同则停止划分
    labelsList = [example[-1] for example in dataset]
    if labelsList.count(labelsList[0]) == len(labelsList):
        return labelsList[0]

    #  遍历完所有特征时返回出现次数最多的类别，即多数表决的结果
    if len(dataset[0]) ==1:
        return majorityClass(labelsList)

    bestFeature = chooseBestFeatureToSplit(dataset)
    featLabel = featureLabels[bestFeature]
    # featureLabels.pop(bestFeature)
    del(featureLabels[bestFeature])
    myTree = {featLabel:{}}
    featureList = [example[bestFeature] for example in dataset]
    featureList = set(featureList)
    for feature in featureList:
        subDataSet = splitDataSet(dataset,bestFeature,feature)
        subFeatureLabels = featureLabels[:]
        myTree[featLabel][feature] = createTree(subDataSet,subFeatureLabels)

    return myTree
    # {'no surfacing': {'1': {'flippers': {'1': ['yes'], '0': ['no']}}, 1: ['yes'], '0': ['no', 'no']}}



'''
3-8
使用决策树的分类函数
'''
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict  =inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classlabel = classify(secondDict[key],featLabels,testVec)
            else:
                classlabel = secondDict[key]
            return classlabel


'''
3-9
使用pickle模块存储决策树
'''
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)