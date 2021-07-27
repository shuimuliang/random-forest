#   -*- coding:utf-8 -*-
#   @Time   :   2020/09/01
#   @Author :   goldsunC
#   @Email  :   2428022854@qq.com
#   @Blog   :   https://blog.csdn.net/weixin_45634606
#   @公众号：goldsunC爱编程

#learn_url = '''https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.
#               html#sklearn.ensemble.RandomForestClassifier'''

from sklearn.datasets import load_iris
#iris 是鸢尾花数据集，包含 150 行数据，
# 分为 3 类：山鸢尾（Setosa）、杂色鸢尾（Versicolour）、维吉尼亚鸢尾（Virginica），
# 每类 50 行数据，每行数据包含 4 个属性：
# 花萼长度（sepal length）、
# 花萼宽度（sepal width）、
# 花瓣长度（petal length）、
# 花瓣宽度（petal width），
# 可通过这 4 个属性来预测鸢尾花属于 3 个种类中的哪一类。
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

iris = load_iris()

#print(iris.data)   #150*4数组
#print(iris.target) #每个花得类别
#print(iris.target_names) #3花类别种类
#print(iris.feature_names) #特征名称
#print(iris.DESCR) #备注说明

#设置Dataframe打印最大宽度和行数
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)


df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['is_train'] = np.random.uniform( low = 0 , high = 1, size = len(df)) <= .75

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


#print(df.head(10))

#3/4作为训练集，1/4作为测试集
train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]

clf = RandomForestClassifier()

y, _ = pd.factorize(train['species'])

#train[features]训练数据特征，y分类结果，有监督学习
clf.fit(train[features], y)


predicts = iris.target_names[clf.predict(test[features])]

#crosstab交叉表
print(pd.crosstab(test['species'], predicts, rownames=['actual'], colnames=['predicts']))
