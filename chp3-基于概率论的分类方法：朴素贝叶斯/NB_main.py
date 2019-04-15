import bayes

# if __name__ == "__main__":
#     listOPosts, listClasses = bayes.loadDataSet()
#     myVocabList = bayes.createVocabList(listOPosts)
#     print(myVocabList)
#     trainMat = []
#     for postinDoc in listOPosts:
#         trainMat.append(bayes.setofWords2Vec(myVocabList,postinDoc))
#     p1V,p0V,pAb = bayes.trainNB0(trainMat,listClasses)
#     print(p1V)
#     print(p0V)
#     print(pAb)


# if __name__ == "__main__":
#     bayes.testingNB()

# if __name__ == "__main__":
#     bayes.spamTest()


import feedparser
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# print(ny)
# print(sf)
# vocabList,p0Vec,p1Vec = bayes.localWords(ny,sf)

# print(ny)
# print(len(ny['feed']))

if __name__== "__main__":
    # testingNB()
    #导入RSS数据源
    import operator

    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList, p0Vec, p1Vec = bayes.localWords(ny, sf)
    print(vocabList)
