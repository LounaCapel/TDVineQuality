#nom, para, retour
from pydantic import BaseModel
from keras import models

class Vin (BaseModel):
    fixedActivity : float

def predictionNote(vin : Vin) -> int:
    return (0)

def vinParfait() -> Vin:
    return ("")

def modeleSerialise() -> models :
    return("")

def description() -> str :
    return ("information")

def enrichir(new : Vin) -> models :
    return ("")

def entrainer() -> models :
    return ("")


import pandas as pd
