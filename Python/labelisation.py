"""
* Labélisation des données



    
step3: -Créer un dossier
        -Dans le dossier génerer pour chaque signal un fichier csv qui contient les valeurs du signal
        -output : 25 fichiers csv dans un dossier
        
algorithme de décomposition :
*output step 2 : 25 signal de durée 65 min 
*tant que la durée du signal > 5 min : 
    dans la boucle dans chaque signal on retranche les premières 5 min : output 25 signal de durée 60 min 
    qui servent comme entrée au step3.
*output tant que : nombre de labels (folder) et dans chaque dossier on a 25 signals
"""
import matplotlib.pyplot as plt
import wfdb
import numpy as np
import pywt 
import cv2
from glob import glob  
import os
database_name  = 'PAF'
data = glob(database_name+'/p*.dat') #data is a list that contain 100 signals : 4 signals for each of 25 patients
N=3000 
"""
step1: read data =record 2 derivation 
 4 read : p01-p01c-p02-p02c
 output step1:array 25*2 p01 ; array 25*2 p01c ; array 25*2 p02 ; array 25*2 p02c
"""
records_name =[] #Define an empty list to add the name of records
records = []  #Define an empty list to add  records
for i in range(len(data)):
    records_name.append(data[i][4:-4]) # data[i]=PAF\\p01.dat==>[4;-4]
    record_name = data[i][4:-4]
    record = wfdb.rdsamp(database_name+'/'+record_name, sampto=N)
    records.append(record)



"""
Step2 (label1): concaténation des 4 signaux de chaque patients (p01-p01c-p02-p02c/patient: 1h 10 min)
    -Detection du dernier pic R du p01 et du premier pic R du p01c ;
    du dernier pic R du p01c et du premier pic R du p02
    -selection le signal p01 de 0 à l'indice du dernier pic R 
    -selection le signal p01c de l'indice du premier pic R à l'indice du dernier de pic R
    -selection le signal p02 de l'indice du premier pic R au dernier indice 
    -concaténation des 3 sélections (c.à.d des 3 listes): 
        boucle pour les 25 patients pour step 2
    -output : 25 signals de durée 65 min"""
patient=[]
patients = []

x = 0
try :
               os.mkdir("Base de données") 
except: 
               print("the folder exists") 
try :
               os.mkdir("Base de données/label0")               
except: 
               print("the folder exists") 
pathx="Base de données/label0"
while x < (len(records)):
    # print(x)
    #patient=np.concatenate([records[x],records[x+1],records[x+2],records[x+3]])
    #patient=np.concatenate([records.p_signals[x],records.p_signals[x+1],records.p_signals[0,x+2],records.p_signals[0,x+3]])
    #patients.append(patient)
     patients.append(records[x] and records[x+1] and records[x+2] and records[x+3])
     x+=4
     np.savetxt(pathx+patients+'_csv.csv',patients,delimiter=',')


print (len(patients))


    