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
from typing import Optional
import pandas as pd
import numpy as np
from csv import writer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle




#####################################################################################################################
##################################################### Class Vin #####################################################
#####################################################################################################################
class Vin(BaseModel) :
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
    quality : Optional[int] = None
    identifiant : Optional[int] = None


#####################################################################################################################
################################################### PredictionVin ###################################################
#####################################################################################################################
'''
Entrée : un vin à tester et un model utilise
Description : permet de renvoyer la prediction du niveau d'un vin en fonction de ses caractéristiques
Sortie : un entier definissant la qualite du vin
'''
def predictionNote(vin : Vin, RandomForest : RandomForestClassifier ) -> int:
    prediction : int = int(RandomForest.predict([[vin.fixedActivity,
                                                vin.volatileAcidity,
                                                vin.citricAcid,
                                                vin.residualSugar,
                                                vin.chlorides,
                                                vin.freeSulfurDioxide,
                                                vin.toalSulfurDioxide,
                                                vin.density,
                                                vin.ph,
                                                vin.sulphates,
                                                vin.alcohol]]))
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
def modeleSerialise(RandomForest : RandomForestClassifier) -> None :
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
def loadModeleSerialise(NomFichier : str) -> RandomForestClassifier :
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
def description(RandomForest : RandomForestClassifier, NomFichier : str) -> str :
    
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

    DictHyperPara["InfoText"] : str = InfoTexteModel
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
    DictHyperPara["Precision"] : float = accuracy
    DictHyperPara["NbrVin"] : int = len(wines["alcohol"])

    
    return (DictHyperPara)


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
def entrainer(NomFichier : str ) -> RandomForestClassifier :
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
         max_features = 'sqrt',
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



# VinTest = Vin(
#     fixedActivity=12, 
#     volatileAcidity=12, 
#     citricAcid=12, 
#     residualSugar=12, 
#     chlorides=12,
#     freeSulfurDioxide=12,
#     toalSulfurDioxide=12,
#     density=12,
#     ph=12,
#     sulphates=12,
#     alcohol=12,
#     quality=12,
#     identifiant=1
#     )


# VinTest2 = Vin(
#     fixedActivity=8.1, 
#     volatileAcidity=0.38, 
#     citricAcid=0.28, 
#     residualSugar=2.1, 
#     chlorides=0.066,
#     freeSulfurDioxide=13.0,
#     toalSulfurDioxide=30.0,
#     density=0.9968,
#     ph=3.23,
#     sulphates=0.73,
#     alcohol=9.7,
#     quality=0,
#     identifiant=2
#     )



# foret = entrainer('Wines (copy).csv')
# pred = predictionNote(VinTest2,foret)
# print(pred)

# #enrichir(VinTest,'Wines (copy).csv')


# print(description(foret, 'Wines.csv'))

# modeleSerialise(foret)

# foret2 = loadModeleSerialise("RandomForestSerialise.pkl")
# pred = predictionNote(VinTest2,foret2)
# print(pred)

#main()
