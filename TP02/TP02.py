import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True, scaled=False).frame

data["sex"] = data["sex"].replace({1: "F", 2: "H"})

lignes_age = data.sample(n=25, random_state=1).index        
lignes_sex = data.sample(n=15, random_state=2).index          
lignes_s3 = data.sample(frac=0.45, random_state=3).index      

data.loc[lignes_age, "age"] = np.nan
data.loc[lignes_sex, "sex"] = np.nan
data.loc[lignes_s3, "s3"] = np.nan

data.to_csv("diabetes_manquants1.csv", index=False)
print("Fichier diabetes_manquants1.csv créé :", data.shape)

# PARTIE 1

data = pd.read_csv("diabetes_manquants1.csv")
print(data.head(10))
data.info()
print(data.isnull().sum())
print(data[data["age"].isnull()])
#Q1
print("Nombre de lignes :", data.shape[0])
print("Nombre de colonnes :", data.shape[1])
#Q2
print(data.isnull())
#Q3
print(data.dtypes)

# PARTIE 2-A

#1
data1=data.dropna()
print(data1)
#2
print("Nombre de lignes de data1 :", data1.shape[0])
print("Nombre de colonnes de data1 :", data1.shape[1])
#3
print(data.head(10))
print(data1.head(10))
#4
perdues=len(data)- len(data1)
print("les lignes perdues",perdues)
pourcentage =(perdues/len(data))*100
print("la pourcentage des  lignes perdues",round(pourcentage,2),"%")
# QUESTION 2 :
#2-1
ligne_reste1=len(data1)

print("le nombre des lignes reste est ",ligne_reste1)
#2-2
disparue=data.index.difference(data1.index)
print(" Index des  premières lignes disparues :",disparue)
#2-3
'''
La méthode dropna() supprime toutes les lignes qui contiennent au moins une valeur manquante 
donc La suppression de ces lignes réduit la taille du dataset et peut entraîner la perte d'informations utiles pour l'entraînement du modèle
'''
# PARTIE 2-B

data2=data.dropna(thresh=10)

print ("la taille de data ",data.shape)
print(data.head)

print ("la taille de data ",data2.shape)
print(data2.head)

#Q1
print("le nombre des lignes reste pour thresh=10 est ",len(data2))
#Q2
# Seuil 9
data3 = data.dropna(thresh=9)

# Seuil 10
data4 = data.dropna(thresh=10)

# Seuil 11
data5 = data.dropna(thresh=11)

# Comparaison
print("thresh=9 :", len(data3))
print("thresh=10 :", len(data4))
print("thresh=11 :", len(data5))
'''
Plus le seuil augmente, plus les conditions de conservation des lignes deviennent strictes. 
Le nombre de lignes conservées diminue ou reste identique.
'''
#Q3
'''data.dropna(thresh=11) equivalent de data.dropna()  '''

# PARTIE 3:
#3-1
pourcentage_VM=data.isnull().mean()*100
print(round(pourcentage_VM,2),"%")
#3-2
data_clean=data.copy()
#3-3
data_clean=data_clean.drop(["s3"],axis=1)
#3-4
print ("la taille de data ",data.shape)
print ("la taille de data_clean ",data_clean.shape)

# QUESTION 4:
#1
pourcentage_s3 = data["s3"].isnull().mean() * 100
print("Pourcentage de valeurs manquantes :", round(pourcentage_s3, 2), "%")
#2
'''La colonne `s3` contient 199 valeurs manquantes, ce qui représente environ 45,02 % des observations du dataset.

Compte tenu de cette proportion importante, nous avons décidé de supprimer cette colonne afin d'éviter de conserver une variable contenant trop de données manquantes.

Après la suppression, le dataset passe de 442 lignes et 11 colonnes à 442 lignes et 10 colonnes. Toutes les lignes sont conservées.'''

# PARTIE 4:
#age variable numerique:
#1
moyenne_age= data_clean["age"].median()
print("LA MOYENNE DE AGE EST :",round(moyenne_age,2))
#2
lignes_age =data_clean[ data_clean["age"].isnull()]
#3
print("Indices des lignes manquantes :",lignes_age.index)
print(lignes_age)
#4
data_clean["age"]=data_clean["age"].fillna(round(moyenne_age,2))
#5
print(data_clean.loc[lignes_age.index])
# B LE SEX: VARIABLE CATEGORIELLE:
#1
print(data_clean["sex"].value_counts())
#2
mode_sex=data_clean["sex"].mode()[0]
print ("la valurs le plus frequente",mode_sex)
#3 
lignes_sex=data_clean[data_clean["sex"].isnull()]
print("Indices des lignes manquantes :",lignes_sex.index)
print(lignes_sex)
#4
data_clean["sex"]=data_clean["sex"].fillna(mode_sex)
#5
print(data_clean.loc[lignes_sex.index])
# PARTIE 5:
print("1er data :",data.info)
print (" data final:" ,data_clean.info)
print(data_clean.isnull().sum())
data_clean.to_csv("diabetes_prepare.csv", index=False)

import os

print(os.path.exists("diabetes_prepare.csv"))
# QUESTION 7 : DIMENSIONS DU DATASET FINAL
print("Nombre de lignes :", data_clean.shape[0])
print("Nombre de colonnes :", data_clean.shape[1])
print("Dimensions finales :", data_clean.shape)















