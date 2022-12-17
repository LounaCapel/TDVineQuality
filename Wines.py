#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:08:45 2022

@author: Capel
"""


import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

from modelIA import *
'''
# Lire les données.
wines = pandas.read_csv("Wines.csv")

WinesData = []
for i in range (len(wines["alcohol"])) :
    liste1 = [wines["fixed acidity"][i], 
              wines["volatile acidity"][i], 
              wines["citric acid"][i], 
              wines["residual sugar"][i],
              wines["chlorides"][i], 
              wines["free sulfur dioxide"][i],
              wines["total sulfur dioxide"][i],
              wines["density"][i],
              wines["pH"][i],
              wines["sulphates"][i],
              wines["alcohol"][i]]
    WinesData.append(liste1)

WinesData = np.array(WinesData)
WinesTarget = np.array(wines["quality"])



clf = DecisionTreeClassifier(criterion="entropy")
#Entrainement de l'abre de décision 
clf.fit(WinesData, WinesTarget)
#Affichage de l'abre de décision obtenu après entraînement


fig = plt.figure(figsize=(105,100))

plot_tree(clf, feature_names= ['fixedActivity', 'volatileAcidity','citricAcid','residualSugar','chlorides','freeSulfurDioxide','toalSulfurDioxide','density','ph','sulphates','alcohol'], class_names=["Qualité3","Qualité4","Qualité5","Qualité6","Qualité7","Qualité8"],filled=True)
plt.show()
fig.savefig("decistion_tree.png")

prediction : int = predictionNote( [[10.2,0.42,0.57,3.4,0.07,4.0,10.0,0.9971,03.04,0.63,9.6]], clf)

print(prediction)



fig = plt.figure(figsize=(105,100))
plot_tree(clf, feature_names= ['fixedActivity', 'volatileAcidity','citricAcid','residualSugar','chlorides','freeSulfurDioxide','toalSulfurDioxide','density','ph','sulphates','alcohol'], class_names=["Qualité3","Qualité4","Qualité5","Qualité6","Qualité7","Qualité8"],filled=True)
plt.show()
fig.savefig("decistion_tree.png")'''


wines = pandas.read_csv("Wines.csv")

WinesData = []
for i in range (len(wines["alcohol"])) :
    liste1 = [wines["fixed acidity"][i], 
              wines["volatile acidity"][i], 
              wines["citric acid"][i], 
              wines["residual sugar"][i],
              wines["chlorides"][i], 
              wines["free sulfur dioxide"][i],
              wines["total sulfur dioxide"][i],
              wines["density"][i],
              wines["pH"][i],
              wines["sulphates"][i],
              wines["alcohol"][i]]
    WinesData.append(liste1)

WinesData = np.array(WinesData)
WinesTarget = np.array(wines["quality"])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


X_train, X_test, y_train, y_test = train_test_split(WinesData, WinesTarget, train_size=0.7, random_state=0)

clf = RandomForestClassifier(
     n_estimators=100,
     criterion='entropy',
     max_depth=None,
     min_samples_split=2,
     min_samples_leaf=1,
     min_weight_fraction_leaf=0.0,
     max_features='auto',
     max_leaf_nodes=None,
     min_impurity_decrease=0.0,
     bootstrap=True,
     oob_score=False,
     n_jobs=None,
     random_state=None,
     verbose=0,
     warm_start=False,
     class_weight=None,
     ccp_alpha=0.0,
     max_samples=None,)

#print(clf.criterion)


clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
#print(f"Le pourcentage de bien classés est de : {accuracy_score(y_test, clf.predict(X_test))*100} %")


print(wines.corr()['quality'])



