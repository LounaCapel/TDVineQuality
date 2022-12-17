#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:08:45 2022

@author: Groupe Florent Kieffer & Louna Capel
"""

#####################################################################################################################
#################################################### Importation ####################################################
#####################################################################################################################
from pydantic import BaseModel
from keras import models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from csv import writer

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle




#####################################################################################################################
##################################################### Class Vin #####################################################
#####################################################################################################################
class Vin :
    fixedActivity : float
    volatileAcidity : float
    citricAcid : float
    residualSugar : float
    chlorides : float
    freeSulfurDioxide : float
    toalSulfurDioxide : float
    density : float
    ph : float
    sulphates : float
    alcohol : float
    quality : int
    identifiant : int


#####################################################################################################################
################################################### PredictionVin ###################################################
#####################################################################################################################
'''
Entrée : un vin à tester et un model utilise
Description : permet de renvoyer la prediction du niveau d'un vin en fonction de ses caractéristiques
Sortie : un entier definissant la qualite du vin
'''
def predictionNote(vin : Vin, RandomForest : models ) -> int:
    prediction : int = RandomForest.predict( [[vin.fixedActivity,
                                               vin.volatileAcidity,
                                               vin.citricAcid,
                                               vin.residualSugar,
                                               vin.chlorides,
                                               vin.freeSulfurDioxide,
                                               vin.toalSulfurDioxide,
                                               vin.density,
                                               vin.ph,
                                               vin.sulphates,
                                               vin.alcohol]])
    return (prediction)


######################################################################################################################
##################################################### VinParfait #####################################################
######################################################################################################################
'''
Entrée : un nom de fichier
Description : permet de renvoyer les caracteristiques d'un vin defini comme parfait avec une note de 10
Sortie : un vin
'''
def vinParfait(NomFichier : str) -> Vin:
    wines = pd.read_csv(NomFichier)
    WinesData : list = []
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
    
    
    
    return ("")


#####################################################################################################################
################################################## ModeleSerialise ##################################################
#####################################################################################################################
'''
Entrée : un model
Description : permet de serialise le model en un fichier en .pkl
Sortie : None
'''
def modeleSerialise(RandomForest : models) -> None :
    pickle.dump(RandomForest, open("RandomForestSerialise.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return(None)


#####################################################################################################################
################################################ LoadModeleSerialise ################################################
#####################################################################################################################
'''
Entrée : un nom de fichier
Description : permet de  load un model serialise
Sortie : un models
'''
def loadModeleSerialise(NomFichier : str) -> models :
    with open(NomFichier, 'rb') as f:
        data = pickle.load(f)
    return(data)


#####################################################################################################################
#################################################### Description ####################################################
#####################################################################################################################
'''
Entrée : un model
Description : permet d'obtenir la description d'un model
Sortie : une description, avec un texte, un dictionnaire et une accuracy
'''
def description(RandomForest : models, NomFichier : str) -> str :
    
    ###############################################
    ######### Description texte du model ##########
    ###############################################
    InfoTexteModel : str = 'Le modèle est une foret aléatoire qui effectue un apprentissage sur de multiples arbres de décision entraînés sur des sous-ensembles de données légèrement différents.'
    
    
    ###############################################
    ############## Accuracy du model ##############
    ###############################################
    wines = pd.read_csv(NomFichier)
    WinesData : list = []
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
    X_train, X_test, y_train, y_test = train_test_split(WinesData, WinesTarget, train_size=0.7, random_state=0)
    RandomForest.fit(X_train, y_train)
    accuracy : float = round(accuracy_score(y_test, RandomForest.predict(X_test))*100,2)


    ###############################################
    ########### HyperParametre du model ###########
    ###############################################
    DictHyperPara : dict = {}

    DictHyperPara["NbrArbre"] : int = RandomForest.n_estimators
    DictHyperPara["CitereDecision"] : str = RandomForest.criterion
    DictHyperPara["ProfondeurMax"] : int = RandomForest.max_depth
    DictHyperPara["EchantillonMinFeuilleSeparation"] : int = RandomForest.min_samples_split
    DictHyperPara["EchantillonMinCreationFeuille"] : int = RandomForest.min_samples_leaf
    DictHyperPara["FractionEchantillonMinCreationFeuille"] : float = RandomForest.min_weight_fraction_leaf
    DictHyperPara["NbrColonneParArbre"] : str = RandomForest.max_features
    DictHyperPara["NbrMaxFeuille"] : int = RandomForest.max_leaf_nodes
    DictHyperPara["CritereImpurete"] : float = RandomForest.min_impurity_decrease
    DictHyperPara["Bootstrap"] : bool = RandomForest.bootstrap
    DictHyperPara["NbrTraitement"] : int = RandomForest.n_jobs
    DictHyperPara["GraineAleatoire"] : int = RandomForest.random_state
    DictHyperPara["Verbose"] : int = RandomForest.verbose
    DictHyperPara["RepartitionApprentisage"] : bool = RandomForest.warm_start
    DictHyperPara["PoidsAssocieClasse"] : int = RandomForest.class_weight
    DictHyperPara["ReduireNbrObservation"] : int = RandomForest.max_samples

    
    return (InfoTexteModel,DictHyperPara,accuracy)


####################################################################################################################
##################################################### Enrichir #####################################################
####################################################################################################################
'''
Entrée : un vin a ajouter et un nom de fichier ou l'ajouter
Description : permet d'ajouter un vin dans la base d'entrainement
Sortie : None
'''
def enrichir(NewVin : Vin, NomFichier : str) -> None :
    list_data : list = []
    list_data.append(str(NewVin.fixedActivity))
    list_data.append(NewVin.volatileAcidity)
    list_data.append(NewVin.citricAcid)
    list_data.append(NewVin.residualSugar)
    list_data.append(NewVin.chlorides)
    list_data.append(NewVin.freeSulfurDioxide)
    list_data.append(NewVin.toalSulfurDioxide)
    list_data.append(NewVin.density)
    list_data.append(NewVin.ph)
    list_data.append(NewVin.sulphates)
    list_data.append(NewVin.alcohol)
    list_data.append(NewVin.quality)
    with open(NomFichier, 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
           
        #Pass the dictionary as an argument to the Writerow()
        writer_object.writerow(list_data)
        #Close the file object
        f_object.close()
    return (None)


#####################################################################################################################
##################################################### Entrainer #####################################################
#####################################################################################################################
'''
Entrée : un nom de fichier contenant les données d'entrainement
Description : permet d'obtenir un model prédictif en fonction des données d'entrée
Sortie : un model prédictif
'''
def entrainer(NomFichier : str ) -> models :
    wines = pd.read_csv(NomFichier)
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
    
    clf = RandomForestClassifier(
         n_estimators = 100,
         criterion = 'entropy',
         max_depth = None,
         min_samples_split = 2,
         min_samples_leaf = 1,
         min_weight_fraction_leaf = 0.0,
         max_features = 'auto',
         max_leaf_nodes = None,
         min_impurity_decrease = 0.0,
         bootstrap = True,
         oob_score = False,
         n_jobs = None,
         random_state = None,
         verbose = 0,
         warm_start = False,
         class_weight = None,
         ccp_alpha = 0.0,
         max_samples = None)
    
    #Entrainement de la forêt
    clf.fit(WinesData, WinesTarget)
    return (clf)



VinTest = Vin()
VinTest.fixedActivity : float = 12
VinTest.volatileAcidity : float = 12
VinTest.citricAcid : float = 12
VinTest.residualSugar : float = 12
VinTest.chlorides : float = 12
VinTest.freeSulfurDioxide : float = 12
VinTest.toalSulfurDioxide : float = 12
VinTest.density : float = 12
VinTest.ph : float = 12
VinTest.sulphates : float = 12
VinTest.alcohol : float = 12
VinTest.quality : float = 12


VinTest2 = Vin()
VinTest2.fixedActivity : float = 8.1
VinTest2.volatileAcidity : float = 0.38
VinTest2.citricAcid : float = 0.28
VinTest2.residualSugar : float = 2.1
VinTest2.chlorides : float = 0.066
VinTest2.freeSulfurDioxide : float = 13.0
VinTest2.toalSulfurDioxide : float = 30.0
VinTest2.density : float = 0.9968
VinTest2.ph : float = 3.23
VinTest2.sulphates : float = 0.73
VinTest2.alcohol : float = 9.7



foret = entrainer('Wines (copy).csv')
pred = predictionNote(VinTest2,foret)
print(pred)

#enrichir(VinTest,'Wines (copy).csv')


print(description(foret, 'Wines.csv'))

#modeleSerialise(foret)

modeleSerialise(foret)

foret2 = loadModeleSerialise("RandomForestSerialise.pkl")
pred = predictionNote(VinTest2,foret2)
print(pred)

#main()


