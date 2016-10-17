
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
#https://ashokharnal.wordpress.com/tag/k-nearest-neighbor-classification-example-using-python/
#http://image.slidesharecdn.com/mlevaluationgladyscastillo-110224084718-phpapp02/95/lesson-3-evaluation-and-comparison-of-supervised-learning-algorithms-67-728.jpg?cb=1393107988

from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import radviz
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import datetime
import time
import glob


def radvizPlot(base, classe):
    plt.figure(figsize=(10,8))
    ax = radviz(base, classe)
    ax.legend(loc='center left', bbox_to_anchor=(0, 1), fancybox=True, ncol=2, fontsize='x-small')
    plt.ylim([-2,2])
    plt.show()

def coordenatesParallels(base, classe):
    plt.figure(figsize=(10,8))
    ax = parallel_coordinates(base, classe)
    ax.legend(loc='center left', bbox_to_anchor=(0, 1), fancybox=True, ncol=2, fontsize='x-small')
    plt.show()


def accuracy(base, classe, n, classificador):
    media = 0
    for i in range(n):
        x = base.drop([classe], axis=1)
        y = base[classe]
        x = StandardScaler().fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)
        classificador.fit(X_train, y_train)
        z = classificador.predict(X_test)
        media += accuracy_score(y_test, z) * 100
    return media/n

def classify(train, test, classe, n, classificador):
	media = 0
	for i in range(n):
		trainBase = train.drop([classe], axis=1)
		trainClasse = train[classe]
		testBase = test.drop([classe], axis=1)
		testClasse = test[classe]
		trainBase = StandardScaler().fit_transform(trainBase)
		testBase = StandardScaler().fit_transform(testBase)
		classificador.fit(trainBase, trainClasse)
		z = classificador.predict(testBase)
		media += accuracy_score(testClasse, z) * 100
	return media/n

def determineClassific(train, test, classe, classificador):
	trainBase = train.drop([classe], axis=1)
	trainClasse = train[classe]
	testBase = test.drop([classe], axis=1)
	testClasse = test[classe]
	trainBase = StandardScaler().fit_transform(trainBase)
	testBase = StandardScaler().fit_transform(testBase)
	classificador.fit(trainBase, trainClasse)
	return classificador.predict(testBase)


def confusionMatrix(base, classe, n, classificador):
    media = 0
    for i in range(n):
        x = base.drop([classe], axis=1)
        y = base[classe]
        x = StandardScaler().fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)
        classificador.fit(X_train, y_train)
        z = classificador.predict(X_test)
        media += confusion_matrix(y_test, z)
    return media/n

def categoricalConversion(data):
	return pd.Categorical.from_array(data).codes


def categorical(data, atributos):
	frame = pd.DataFrame()
	for col in atributos:
		frame[col] = pd.Categorical.from_array(data[col]).codes
	return frame

def categoricalRename(data, atributos):
	frame = pd.DataFrame()
	l = len(atributos)
	rename = np.arange(1,l)
	for col, name in zip(atributos, rename):
		frame[name] = pd.Categorical.from_array(data[col]).codes
	return frame

def monthConversion(x):
	if x == 'jan':
		return 1   
	if x == 'feb':
		return 2 
	if x == 'mar':
		return 3
	if x == 'apr':
		return 4 
	if x == 'may':
		return 5 
	if x == 'jun':
		return 6 
	if x == 'jul':
		return 7 
	if x == 'aug':
		return 8 
	if x == 'sep':
		return 9
	if x == 'oct':
		return 10 
	if x == 'nov':
		return 11
	if x == 'dec':
		return 12		

def dayConversion(x):
	if x == 'sun':
		return 1
	if x == 'mon':
		return 2 
	if x == 'tue':
		return 3
	if x == 'wed':
		return 4 
	if x == 'thu':
		return 5 
	if x == 'fri':
		return 6 
	if x == 'sat':
		return 7 



##################### Pesquisa do Perfil Empreendedor Database #####################

#pesquisa = pd.read_csv('Pesquisa - Potencial Empreendedor dos alunos de Sistemas da Informação da UFRN-CERES.csv',sep=',')
#classe = pesquisa['Qual sua intenção ao concluir o curso?']
#pesquisa = pesquisa.drop(["Trabalha? Se sim, em que?"], axis=1)
#year = datetime.date.today().year
#pesquisa['Data de nascimento'] = pesquisa['Data de nascimento'].apply(lambda x : year - time.strptime(x, "%d/%m/%Y").tm_year)
#pesquisa = pesquisa.drop(['Qual sua intenção ao concluir o curso?'], axis=1)
#categoric = categorical(pesquisa, list(pesquisa.columns.values))
#categoric['classe'] = classe

#print (accuracy(categoric, 'classe', 1000, RandomForestClassifier()))
#print (accuracy(categoric, 'classe', 1000, SVC()))
#print (accuracy(categoric, 'classe', 1000, GaussianNB()))
#print (accuracy(categoric, 'classe', 1000, KNeighborsClassifier()))
#print (accuracy(categoric, 'classe', 1000, DecisionTreeClassifier()))

#radvizPlot(categoric, 'classe')


####################### Character Database ############################

#learn = [pd.read_table(filename, header=None, delim_whitespace=True) for filename in glob.glob("CharacterDatabase/learn/*")]
#test = [pd.read_table(filename, header=None, delim_whitespace=True) for filename in glob.glob("CharacterDatabase/test/*")]
#testUser = [pd.read_table(filename, header=None, delim_whitespace=True) for filename in glob.glob("CharacterDatabase/testUser/*")]
#dataLearn = pd.concat(learn, axis=0, ignore_index=True)
#dataTest = pd.concat(test, axis=0, ignore_index=True)
#dataTestUser = pd.concat(testUser, axis=0, ignore_index=True)
#dataLearn.columns = ['classe','line', 'str', 'x1','y1','x2','y2', 'length', 'diagonal']
#dataTest.columns = ['classe','line', 'str', 'x1','y1','x2','y2', 'length', 'diagonal']
#dataTestUser.columns = ['classe','line', 'str', 'x1','y1','x2','y2', 'length', 'diagonal']
#dataLearn = dataLearn.drop('str', axis=1)
#dataTest = dataTest.drop('str', axis=1)
#dataTestUser = dataTestUser.drop('str', axis=1)

#radvizPlot(dataLearn, 'classe')
#radvizPLot(dataTest, 'classe')

#print (accuracy(dataLearn, 'classe', 1000, RandomForestClassifier()))
#print (accuracy(dataLearn, 'classe', 1000, SVC()))
#print (accuracy(dataLearn, 'classe', 1000, GaussianNB()))
#print (accuracy(dataLearn, 'classe', 1000, KNeighborsClassifier()))
#print (accuracy(dataLearn, 'classe', 1000, DecisionTreeClassifier()))

#print (classify(dataLearn, dataTest, 'classe', 10, SVC()))
#print (classify(dataLearn, dataTest, 'classe', 10, RandomForestClassifier()))
#print (classify(dataLearn, dataTest, 'classe', 10, GaussianNB()))
#print (classify(dataLearn, dataTest, 'classe', 10, KNeighborsClassifier()))
#print (classify(dataLearn, dataTest, 'classe', 10, DecisionTreeClassifier()))

#print (determineClassific(dataLearn, dataTestUser, 'classe', RandomForestClassifier()))


########################### Forestfires #########################

#forestfires = pd.read_csv('forestfires.csv',sep=',')
#forestfires['area'] = forestfires['area'] > 0
#forestfires = forestfires.drop('X', axis=1)
#forestfires = forestfires.drop('Y', axis=1)
#forestfires['month'] = forestfires['month'].apply(lambda x: monthConversion(x))
#forestfires['day'] = forestfires['day'].apply(lambda x: dayConversion(x))

#print (accuracy(forestfires, 'area', 1000, RandomForestClassifier()))
#print (accuracy(forestfires, 'area', 1000, SVC()))
#print (accuracy(forestfires, 'area', 1000, GaussianNB()))
#print (accuracy(forestfires, 'area', 1000, KNeighborsClassifier()))
#print (accuracy(forestfires, 'area', 1000, DecisionTreeClassifier()))

#radvizPlot(forestfires, 'area')


########################### parkinsons #############################

parkinsons = pd.read_csv('parkinsons.csv',sep=',')
parkinsons['status'] = parkinsons['status'] > 0
parkinsons['patient'] = categoricalConversion(parkinsons['name'].str.rsplit('_', expand=True, n=1)[0])
parkinsons['record'] = categoricalConversion(parkinsons['name'].str.rsplit('_', expand=True, n=1)[1])
parkinsons = parkinsons.drop('name', axis=1)

exemploParkinsons = pd.read_csv('exemploParkinsons.csv',sep=',')
exemploParkinsons['status'] = exemploParkinsons['status'] > 0
exemploParkinsons['patient'] = categoricalConversion(exemploParkinsons['name'].str.rsplit('_', expand=True, n=1)[0])
exemploParkinsons['record'] = categoricalConversion(exemploParkinsons['name'].str.rsplit('_', expand=True, n=1)[1])
exemploParkinsons = exemploParkinsons.drop('name', axis=1)

#print (accuracy(parkinsons, 'status', 1000, RandomForestClassifier()))
#print (accuracy(parkinsons, 'status', 1000, SVC()))
#print (accuracy(parkinsons, 'status', 1000, GaussianNB()))
#print (accuracy(parkinsons, 'status', 1000, KNeighborsClassifier()))
#print (accuracy(parkinsons, 'status', 1000, DecisionTreeClassifier()))

#print (determineClassific(parkinsons, exemploParkinsons, 'status', RandomForestClassifier()))
print (determineClassific(parkinsons, exemploParkinsons, 'status', DecisionTreeClassifier()))

#radvizPlot(parkinsons, 'status')
#coordenatesParallels(parkinsons, 'status')

#print (parkinsons)





