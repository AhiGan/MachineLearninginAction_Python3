import numpy as np
import random
'''
4-1
词表到向量的转换函数
'''


# 加载数据集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1-侮辱类，0-非侮辱类
    return postingList, classVec


# 基于数据集构造词汇表
def createVocabList(dataset):
    vocabList = set([])  # 空集
    for data in dataset:
        vocabList = vocabList | set(data)  # 两个集合的并集
    return list(vocabList)


# 将词条转换为词表向量
def setofWords2Vec(vocabList, inputSet):
    retVec = [0] * len(vocabList)
    for i in inputSet:
        if i in vocabList:
            retVec[vocabList.index(i)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % i)
    return retVec


'''
4-2
朴素贝叶斯分类器训练函数，返回两个类别的概率向量及属于侮辱性文档的概率
'''


# 基于极大似然估计
def trainNB0_0(trainMatrix, trainCategory):
    numTrainDocs = len(trainCategory)
    probAbusive = sum(trainCategory) / float(numTrainDocs)  # 属于侮辱性文档的概率,p1
    numWords = len(trainMatrix[0])
    p1NumVocab = np.zeros(numWords)
    p0NumVocab = np.zeros(numWords)
    p1NumTotal = 0.0
    p0NumTotal = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1NumVocab += trainMatrix[i]
            p1NumTotal += sum(trainMatrix[i])
        else:
            p0NumVocab += trainMatrix[i]
            p0NumTotal += sum(trainMatrix[i])
    p1ProbVec = p1NumVocab / p1NumTotal
    p0ProbVec = p0NumVocab / p0NumTotal

    return p1ProbVec, p0ProbVec, probAbusive


# 基于贝叶斯估计
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainCategory)
    probAbusive = sum(trainCategory) / float(numTrainDocs)  # 属于侮辱性文档的概率,p1
    numWords = len(trainMatrix[0])
    p1NumVocab = np.ones(numWords)
    p0NumVocab = np.ones(numWords)
    p1NumTotal = 2.0
    p0NumTotal = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1NumVocab += trainMatrix[i]
            p1NumTotal += sum(trainMatrix[i])
        else:
            p0NumVocab += trainMatrix[i]
            p0NumTotal += sum(trainMatrix[i])
    p1ProbVec = np.log(p1NumVocab / p1NumTotal)
    p0ProbVec = np.log(p0NumVocab / p0NumTotal)

    return p1ProbVec, p0ProbVec, probAbusive


'''
4-3
朴素贝叶斯分类函数
'''


# 取log以后，连乘变连加
def classifyNB(vec2Classify, p1vec, p0Vec, pClass1):
    p1 = sum(vec2Classify * p1vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setofWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setofWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))




'''
4-4
朴素贝叶斯词袋模型
'''
def bagOfWords2VecMN(wordsList, inputSet):
    retVec = [0] * len(wordsList)
    for word in inputSet:
        if word in wordsList:
            retVec[wordsList.index(word)] += 1

    return retVec

'''
4.6 示例
使用朴素贝叶斯过滤垃圾邮件,一次迭代
4-5 文件解析及完整的垃圾邮件测试函数
'''

# 文件解析函数,返回长度大于2的token的小写形式
def textParse(bigString):
    import re
    listToken = re.compile('\\W*').split(bigString)
    return [tok.lower() for tok in listToken if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    for i in range(1,26): # 正负样例各25
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainList = list(range(50))
    testList = []
    for i in range(10):
        index = int(random.uniform(0,len(trainList)))
        testList.append(trainList[index])
        trainList.pop(index)

    trainMat = []
    trainClass = []
    for j in trainList:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[j]))
        trainClass.append(classList[j])
    p1Vec, p0Vec, pSpam = trainNB0(np.array(trainMat),np.array(trainClass))
    errorCount = 0
    for j in testList:
        res = classifyNB(bagOfWords2VecMN(vocabList,docList[j]),p1Vec,p0Vec,pSpam)
        if res!= classList[j]:
            errorCount += 1
            print("classification error", docList[j])
    print('the error rate is: ', float(errorCount)/len(testList))



'''
4.7 示例
使用朴素贝叶斯分类器从个人广告中获取区域倾向
'''

'''
4-6
RSS源分类器及高频词去除函数
'''
# 返回最高频的前30个字典
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]


def localWords(feed1,feed0):
    docList = []
    classList = []
    minLen = min(len(feed0),len(feed1))

    for i in range(minLen):
        doc = textParse(feed1['entries'][i]['summary'])
        docList.append(doc)
        classList.append(1)
        doc = textParse(feed0['entries'][i]['summary'])
        docList.append(doc)
        classList.append(0)

    trainSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        index = int(random.uniform(0,len(trainSet)))
        testSet.append(index)
        trainSet.pop(index)

    # 获取词表,并去除高频词
    vocabList = createVocabList(docList)
    topFreq30 = calcMostFreq(vocabList,docList)
    for pair in topFreq30:
        if pair[0] in vocabList:
            vocabList.remove(pair[0])

    trainMat = []
    trainClasses = []
    for j in trainSet:
        wordVec = bagOfWords2VecMN(vocabList,docList[j])
        trainMat.append(wordVec)
        trainClasses.append(classList[j])

    p1Vec, p0Vec, pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for k in testSet:
        testVec = bagOfWords2VecMN(vocabList,docList[k])
        res = classifyNB(testVec,p1Vec,p0Vec,pSpam)
        if res != classList[k]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList,p0Vec,p1Vec


'''
4-7
最具表征性的词汇显示函数
返回大于阈值（-6.0）的所有词，元组按照它们的条件概率进行排序
'''
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])




