from numpy import *
import kNN
import matplotlib
import matplotlib.pyplot as plt

# if __name__=="__main__":
#     # 构造自制数据集
#     train_data_X, train_data_labels = kNN.createDataset()
#     print(kNN.classify0([0,0],train_data_X,train_data_labels,3))



# if __name__=="__main__":
#     # 导入约会数据集
#     dating_train_data_X, dating_train_data_labels = kNN.file2matrix('datingTestSet2.txt')
#     print(dating_train_data_X[0:20,:])
#     print(dating_train_data_labels[0:20])
#
#     # 分析数据，使用Matplotlib创建散点图
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     # scatter(x,y,size,color)
#     ax.scatter(dating_train_data_X[:,1],dating_train_data_X[:,2],
#                15.0 * array(dating_train_data_labels),15.0 * array(dating_train_data_labels))
#     plt.show()
#
#     # 输入数据归一化
#     normDataset, ranges, min_features = kNN.autoNorm(dating_train_data_X)
#     print(ranges)


# 从数据集中划分一部分作为测试集进行测试，统计错误率
# if __name__=="__main__":
#     kNN.datingClassTest()


# 对自己输入的样本进行测试
if __name__=="__main__":
    kNN.classifyPerson()
