import trees
import treePlotter

# if __name__=="__main__":
#     dataset, labels = trees.creatDataSet()
#     print(trees.calcShannonEnt(dataset))


# if __name__=="__main__":
#     dataset, labels = trees.creatDataSet()
#     print(trees.createTree(dataset,labels))


# if __name__=="__main__":
#     treePlotter.createPlot()


# if __name__=="__main__":
#     myTree= treePlotter.retrieveTree(0)
#     print(treePlotter.getNumLeafs(myTree))
#     print(treePlotter.getDepth(myTree))
#     treePlotter.createPlot(myTree)

if __name__=="__main__":
    dataset, labels = trees.creatDataSet()
    myTree= treePlotter.retrieveTree(0)
    print(trees.classify(myTree,labels,[1,0]))

    trees.storeTree(myTree,'classifierStorage.txt')
    print(trees.grabTree('classifierStorage.txt'))

