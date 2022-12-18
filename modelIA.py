#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
"""
Cette fonction permet de renvoyer la prediction du niveau d'un vin en fonction de ses caractéristiques

Args :
    vin (Vin) : un vin à tester
    RandomForest (RandomForestClassifier) : un model

Returns : 
    prediction (int) : prediction de la qualité
"""
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
"""
Cette fonction permet de renvoyer les caracteristiques d'un vin defini comme parfait avec une note de 10

Args : 
    NomFichier (str) : un nom de fichier

Returns : 
    vinParfait (Vin) : un vin parfait
"""
def vinParfait(NomFichier : str) -> Vin:
    wines = pd.read_csv(NomFichier)
    fixedAcidity : list = []
    volatileAcidity : list = []
    citricAcid : list = []
    residualSugar : list = []
    chlorides : list = []
    freeSulfurDioxide : list = []
    totalSulfurDioxide : list = []
    density : list = []
    pH : list = []
    sulphates : list = []
    alcohol : list = []
    
    for i in range (len(wines["alcohol"])):
        
        fixedAcidity.append(wines["fixed acidity"][i])
        volatileAcidity.append(wines["volatile acidity"][i])
        citricAcid.append(wines["citric acid"][i])
        residualSugar.append(wines["residual sugar"][i])
        chlorides.append(wines["chlorides"][i])
        freeSulfurDioxide.append(wines["free sulfur dioxide"][i])
        totalSulfurDioxide.append(wines["total sulfur dioxide"][i])
        density.append(wines["density"][i])
        pH.append(wines["pH"][i])
        sulphates.append(wines["sulphates"][i])
        alcohol.append(wines["alcohol"][i])
        
    maxCitricAcid : float = max(citricAcid)
    maxSulphate : float = max(sulphates)
    maxAlcohol : float = max(alcohol)
    
    meanFixedAcidity : float = round((sum(fixedAcidity) / len(fixedAcidity)),2)
    meanVolatileAcidity : float = round((sum(volatileAcidity) / len(volatileAcidity)),2)
    meanResidualSugar : float = round((sum(residualSugar) / len(residualSugar)),2)
    meanChlorides : float = round((sum(chlorides) / len(chlorides)),2)
    meanFreeSulfurDioxide : float = round((sum(freeSulfurDioxide) / len(freeSulfurDioxide)),2)
    meanTotalSulfurDioxide : float = round((sum(totalSulfurDioxide) / len(totalSulfurDioxide)),2)
    meanDensity : float = round((sum(density) / len(density)),2)
    meanPH : float = round((sum(pH) / len(pH)),2)

    vinParfait = Vin(fixedActivity=meanFixedAcidity, 
                     volatileAcidity=meanVolatileAcidity, 
                     citricAcid=maxCitricAcid, 
                     residualSugar=meanResidualSugar, 
                     chlorides=meanChlorides,
                     freeSulfurDioxide=meanFreeSulfurDioxide,
                     toalSulfurDioxide=meanTotalSulfurDioxide,
                     density=meanDensity,
                     ph=meanPH,
                     sulphates=maxSulphate,
                     alcohol=maxAlcohol)
    
    return (vinParfait)


#####################################################################################################################
################################################## ModeleSerialise ##################################################
#####################################################################################################################
"""
Cette fonction permet de serialise le model en un fichier en .pkl

Args : 
    RandomForest (RandomForestClassifier) : un model

Returns : 
    None
"""
def modeleSerialise(RandomForest : RandomForestClassifier) -> None :
    pickle.dump(RandomForest, open("RandomForestSerialise.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return(None)


#####################################################################################################################
################################################ LoadModeleSerialise ################################################
#####################################################################################################################
"""
Cette fonction permet de load un model serialise

Args : 
    NomFichier (str) : un nom de fichier

Returns : 
    RandomForest (RandomForestClassifier) : un model
"""
def loadModeleSerialise(NomFichier : str) -> RandomForestClassifier :
    with open(NomFichier, 'rb') as f:
        data = pickle.load(f)
    return(data)


#####################################################################################################################
#################################################### Description ####################################################
#####################################################################################################################
"""
Cette fonction permet d'obtenir la description d'un model

Args : 
    RandomForest (RandomForestClassifier) : un model
    NomFichier (str) : un nom de fichier

Returns : 
    DictHyperPara (dict) : description du model
"""
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
"""
Cette fonction permet d'ajouter un vin dans la base d'entrainement

Args : 
    NewVin (Vin) : un vin à ajouter
    NomFichier (str) : un nom de fichier

Returns : 
    None
"""
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
"""
Cette fonction permet d'obtenir un model prédictif en fonction des données d'entrée

Args : 
    NomFichier (str) : un nom de fichier
 
Returns : 
    clf (RandomForestClassifier) : un model
"""
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