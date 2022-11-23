
#nom, para, retour
from pydantic import BaseModel
from keras import models

class Vin (BaseModel):
    fixedActivity : float

def PredictionNote (vin : Vin):
    return (0)

def VinParfait ():
    return ("")

def ModeleSerialise (modele : models):
    return()

def Description ():
    return ("information")

def Enrichir (modele : models,new : Vin ):
    return (modele)

def Entrainer (modele : models):
    return (modele)
