import wfdb
import numpy as np
import pywt 
import cv2
from glob import glob  
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


#Nom de la base de donnée dans le fichier
database= 'PAF'
#Lire les fichiers .dat qui commenece par la lettre P ; / pour enter dans le doosier
#data is a list that contain 100 signals : 4 signals for each of 25 patients
data = glob(database+'/p*.dat') 
#Nombre des échantillons
N=3000 
"""
*Cette base contient 25 patients
Pour chaque patient on a 4 records p (signal)
Par exemple pour le patient 1 on 2:
    -p01 de durée 30 min : normal
    -p01c de durée 5 min : sans AF
    -p02 de durée 30 min : avant AF
    -p02c de durée 5 min : avec AF
Donc par total on a 100 signals.
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
Pour diviser ces signaux en 4 catégories on parcours 2 boucles.
*La première boucle de l'indice 0 à l'indice 99 pour séparer les signaux  px 
dont l'indice est paire et les signaux pxc dont l'indice est impaire.
La sortie de la première boucle sont 2 listes :
    -liste 1 : p01, p02,p03,p04, ... (50 signals de l'indice 0 à 49)
    -liste 2 : p01c, p02c,p03c,p04c, ... (50 signals de l'indice 0 à 49)
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    
*La deuxième boucle de l'indice 0 à l'indice 49 pour chacune les listes 1 et 2
pour séparer les signaux  impaire  dont l'indice est paire des signaux paires 
dont l'indice est impaire.
La sortie de la deuxième boucle sont 4 listes :
    -2 de la liste 1 :
        ► liste1_normale:p01,p03,p05,....(25 signals)
        ►liste1_avant AF:p02,p04,p06,....(25 signals)
    -2 de la liste 2 :
        ► liste2_sans:p01c,p03c,p05c,....(25 signals)
        ►liste2_avecAF:p02c,p04c,p06c,....(25 signals)
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
step1: read data =record 2 derivation 
 4 read : p01-p01c-p02-p02c
 output step1: array 25*2 p01; array 25*2 p01c; array 25*2 p02; array 25*2p02c
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
"""
#Initialiser deux listes vides : list_px pour les signaux px et list_pxc pour les signaux pxc
#Première boucle qui regroupe les 100 signals dans 2 listes dont chacune contient 50 signals selon l'indice 
list_px=[]
list_pxc=[]
for i in range (len(data)):
    if (i%2==0): 
        list_px.append(data[i])
    else:
        list_pxc.append(data[i])
#Sortie de la première boucle : 2 listes chacune contenant 50 signals. 
# Deuxième boucle qui décompose chacune des deux listes obtenues en deux autres listes 
p_normal,p_avantAF,p_sansAF,p_avecAF=[],[],[],[]
for j in range (len(list_px)):
    if (j%2==0):
        p_normal.append(list_px[j])
    else:
        p_avantAF.append(list_px[j])   
for k in range (len(list_pxc)):
    if (k%2==0):
       p_sansAF.append(list_pxc[k])
    else:
        p_avecAF.append(list_pxc[k])
#Sortie de la deuxième boucle : 4 listes chacune contenant 25 signals. 

    

def read_record(path):
    records_name,records,l_derv0,l_derv1=[],[],[],[]
    for i in range (len(path)):
        #Lecture d'un signal
        records_name.append(path[i][4:-4]) # data[i]=PAF\\p01.dat==>[4;-4]
        record_name = path[i][4:-4]
        #Read a WFDB record, and return the physical signals and a few important descriptor fields.
        record = wfdb.rdsamp(database+'/'+record_name, sampto=N)
        records.append(record)
        l0=record.p_signals.T[0]
        l_derv0.append(l0)
        l1=record.p_signals.T[1]
        l_derv1.append(l1)
    return l_derv0,l_derv1

l_derv0_p_normal,l_derv1_p_normal=read_record(p_normal)
l_derv0_p_avantAF,l_derv1_p_avantAF=read_record(p_avantAF)
l_derv0_p_sansAF,l_derv1_p_sansAF=read_record(p_sansAF) 
l_derv0_p_avecAF,l_derv1_p_avecAF=read_record(p_avecAF)       

"""
►L'entrée de l'étape 2 sont les 4 arrays pour chaque dérivation.
►On doit choisir l'une des deux dérivations pour faire l'étape 2

Step2 (label1): concaténation des 3 signaux de chaque patients (p01-p01c-p02/patient:1h5min)
    -Detection du dernier pic R du p01 et du premier pic R du p01c;
    du dernier pic R du p01c et du premier pic R du p02
    -selection le signal p01 de 0 à l'indice du dernier pic R 
    -selection le signal p01c de l'indice du premier pic R à l'indice du dernier de pic R
    -selection le signal p02 de l'indice du premier pic R au dernier indice 
    -concaténation des 3 sélections (c.à.d des 3 listes): 
        boucle pour les 25 patients pour step 2
    -output : 25 signaux de durée 65 min""" 
#x is a numpy array ; l_derv0_p_normal is a list ; list to nparray
def find_peak(array):
    index=[]
    for i in range(len(array)):
        x = np.array(array[i])[0:3000]
        peaks, _ = find_peaks(x, prominence=(None, 0.6))
        np.diff(peaks)
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.show()
        index.append(peaks)
    return(index)
iii=peak_normal=find_peak(l_derv0_p_normal)
#peak_sansAF=find_peak(l_derv0_p_sansAF)
#peak_avantAF=find_peak(l_derv0_p_avantAF)
#peak_avecAF=find_peak(l_derv0_p_avecAF)
