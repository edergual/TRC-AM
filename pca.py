# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:10:19 2018

@author: Éder Souza Gualberto
"""

import ExtractMsg
import glob
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import re
import nltk
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

import ExtractMsg
import glob
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
#from os import listdir
nltk.download('rslp')
from nltk.stem import RSLPStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint

def removerAcentosECaracteresEspeciais(palavra):

    # Unicode normalize transforma um caracter em seu equivalente em latin.
    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])

    # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
    return re.sub('[^a-zA-Z]', ' ', palavraSemAcento)


array = []

for message in glob.glob('phishing/português/*.msg'):
    #print 'Reading', message
    msg = ExtractMsg.Message(message)
    body = msg._getStringStream('__substg1.0_1000')
    sender = msg._getStringStream('__substg1.0_0C1F')
    array.append(body)
    #print (str(body))
msgarray = pd.DataFrame(array)
msgarray['Phishing'] = 1
msgarray.columns=["Message","Phishing"]

array2 = []
for message in glob.glob('ham/*.msg'):
    #print 'Reading', message
    msg = ExtractMsg.Message(message)
    body = msg._getStringStream('__substg1.0_1000')
    sender = msg._getStringStream('__substg1.0_0C1F')
    array2.append(body)
    #print (str(body))
msgarray2 = pd.DataFrame(array2)
msgarray2['Phishing'] = 0
msgarray2.columns=["Message","Phishing"]

msgarray = msgarray.append(msgarray2)

# Cleaning the texts

#palavras que nao adicionam significado, tais como artigos, preposicoes etc
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1155):
    #retira tudo o que n'ao for letras maisculas ou minusculas
    review = removerAcentosECaracteresEspeciais(str(msgarray['Message'].iloc[i]))
    #coloca tudo wm minusculo
    review = review.lower()
    #separa cada linha em tokens
    review = review.split()
    ps = RSLPStemmer()
    #exclui palavras que constam na lista stopwords, e as reduz ao radical minimo (morfema)
    #retirando prefixos e sufixos
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('portuguese'))]
    #junta todos os tokens incluindo espaco entre eles
    review = ' '.join(review)
    #junta essa linha ao corpus
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
#transforma cada token/palavra em corpus em uma coluna/feature
X = cv.fit_transform(corpus).toarray()
#X = StandardScaler().fit_transform(X)
#coloca em Y apenas a distincao entre bons e maus reviews


pca = PCA(n_components=100)


principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)


y = msgarray.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(principalDf, y, test_size = 0.30, random_state = 0)



from sklearn.svm import SVC

# Set the parameters by cross-validation
tuned_parameters_svc = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    #abordagem cross-validation 5x2
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=123456)

    clf = GridSearchCV(SVC(), tuned_parameters_svc, cv=cv,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred1 = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred1))
    print()
    
    
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_true, y_pred1)
print (cm1)
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.







# utiliza o algoritmo Naive Bayes no conjunto de treinamento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Faz a predição para o conjunto de teste
y_pred2 = classifier.predict(X_test)

# Cria a matriz de confusão
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
print (cm2)

# utiliza o algoritmo de regressão logística no conjunto de treinamento
from sklearn.linear_model import LogisticRegression


# Set the parameters by cross-validation
tuned_parameters_lr = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'penalty': ['l1', 'l2']}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=123456)

    clf = GridSearchCV(LogisticRegression(), tuned_parameters_lr, cv=cv,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred3 = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred3))
    print()
    
    
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_true, y_pred3)
print (cm3)
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.


# utiliza o algoritmo K-Neighbors no conjunto de treinamento
from sklearn.neighbors import KNeighborsClassifier




# Set the parameters by cross-validation
tuned_parameters_neighbors = [{
        'n_neighbors': [1, 3, 5, 10, 50, 100],
        'weights': ['uniform', 'distance']
    }]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=123456)

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters_neighbors, cv=cv,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred4 = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred4))
    print()
    
    
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_true, y_pred4)
print (cm4)






# utiliza o algoritmo árvores de decisão no conjunto de treinamento
from sklearn.tree import DecisionTreeClassifier

# Set the parameters by cross-validation
tuned_parameters_tree = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=123456)

    clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters_tree, cv=cv,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred5 = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred5))
    print()


from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_true, y_pred5)
print (cm5)




# utiliza o algoritmo random forest no conjunto de treinamento
from sklearn.ensemble import RandomForestClassifier
# Set the parameters by cross-validation
tuned_parameters_rf = {"max_depth": [3, None],
              "max_features": ['auto', 'sqrt', 'log2'],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=123456)

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters_rf, cv=cv,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred6 = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred6))
    print()
    
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_true, y_pred6)
print (cm6)

print("SVC")
print(classification_report(y_true, y_pred1))
print(cm1)
print("Naive Bayes Gaussian")
print(classification_report(y_true, y_pred2))
print(cm2)
print("Regressao logistica")
print(classification_report(y_true, y_pred3))
print(cm3)
print("K-Neighbors")
print(classification_report(y_true, y_pred4))
print(cm4)
print("Arvore de Decisoao")
print(classification_report(y_true, y_pred5))
print(cm5)
print("Random Forest")
print(classification_report(y_true, y_pred6))
print(cm6)