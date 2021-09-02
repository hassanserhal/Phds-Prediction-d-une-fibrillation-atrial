"""PAF: Data base disponible on physionnet
50 signals : n01--> n50: all normal
For each patient : p01 (30 min) --p01c (5 min) --- p02 (30 min):All normal 
--- p02c (5min) :With FA
For each signal there are 2 derivations : derivation 0 and 1
Total duration of signal p is 65 min
fe=128 Hz then the lenght of signal =65 *60*128 =499200
We interest on the signal p
"""
#import librairies
import matplotlib.pyplot as plt
"""A library of tools for reading, writing, and processing WFDB signals and annotations."""
import wfdb 
import numpy as np
import pywt 
import cv2
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split 

"""Glob is a general term used to define techniques to match specified patterns 
according to rules related to Unix shell"""
from glob import glob  
import os
#Import data from PAF
database_name  = 'PAF'
#/p*.dat :/ to enter inside the folder
#data = glob(database_name+'/p0*.dat')
data = glob(database_name+'/p*.dat')
""" data is a list that contain 100 signals : 4 signals for each of 25 patients"""
N=3000 #number of samples
"""##########################import data function##################"""
"""create a fonction to import data; input=path ; output=data"""
def import_data (data) :
    records_name =[] #Define an empty list to add the name of records
    records = []  #Define an empty list to add  records
    for i in range(len(data)):
        records_name.append(data[i][4:-4]) # data[i]=PAF\\p01.dat==>[4;-4]
        record_name = data[i][4:-4]
        record = wfdb.rdsamp(database_name+'/'+record_name, sampto=N)
        records.append(record)
    return records
"""########################################################################"""
"""Call the function import_data"""  #s=import_data(data) :to call a function it is necessary to assign it to the value    
"""divide data into normale (even p signal) and anormale (odd p signal) for each derivation"""    
list_test=[]; list_normale=[]; list_anormale=[];
#list_test contain all the signal pxc number =50;list_normale contain px+1c (normale) number=25
#list_anormale contain pxc (anormale) number=25
count=0
#Create a list thant contain the signal p0xc: index0:p01;index1:p01c;index2:p02;index3:p02c
# range (1=p01c , len(data)=100, 2 : step 2 : p02c --> p03c --> p04c....)
for i in range (1,len(data),2): 
    list_test.append(data[i])
    if count%2==0 : #signaux 
            list_normale.append(data[i])
    else:
            list_anormale.append(data[i])
    print(i)
    count+=1
list_normale_values=import_data(list_normale)
list_anormale_values=import_data(list_anormale)
#    list_normale_values=import_data(list_normale)
#    list_anormale_values=import_data(list_anormale)
"""Apply EMD  for each pair pxc on derv0 and drev1""" 
# EMD p01c-p02c
listderv0=[]; listderv1=[]
for i in range(min(len(list_normale),len(list_anormale))): 
        #min: to obtain the same lenght for each list
        print('{}/{}'.format(list_normale[i],list_anormale[i]))
        signal_normale_0=list_normale_values[i].p_signals.T[0] 
        #p_signals inside tle list_normale_values and contain 2 columns 0 and 1 ; each column contain 3000 samples
        #T :tranposet of p_signals (convert columns to rows)
        signal_normale_1=list_normale_values[i].p_signals.T[1]
        signal_anormale_0=list_anormale_values[i].p_signals.T[0]
        signal_anormale_1=list_anormale_values[i].p_signals.T[1]
        listderv0.append([signal_normale_0,signal_anormale_0])
        listderv1.append([signal_normale_1,signal_anormale_1])
"""##########################generate imf function##################"""
"""import all from: test_emd """
"""Create a function named imf_generate"""
from test_emd import *
if __name__ == "__main__":
    import pylab as plt
def imf_generate(signal,N) :
    # EMD options
        max_imf = -1
        DTYPE = np.float64
     # Signal options
        S = signal
        S = S.astype(DTYPE)
        print("Input S.dtype: " + str(S.dtype))
        tMin, tMax = 0, N
        #T = np.linspace(tMin, tMax, N, dtype=DTYPE)
        T = np.array([ii for ii in range(1,N+1)]).astype(DTYPE)
        # Prepare and run EMD
        emd = EMD()
        emd.FIXE_H = 5
        emd.nbsym = 2
        emd.spline_kind = 'cubic'
        emd.DTYPE = DTYPE
        imfs = emd.emd(S, T, max_imf)
        imfNo = imfs.shape[0]    
        # Plot results
#        c = 1
#        r = np.ceil((imfNo+1)/c)
#        plt.ioff()
#        plt.subplot(r, c, 1)
#        plt.plot(T, S, 'r')
#        plt.xlim((tMin, tMax))
#        plt.title("Original signal")   
#        for num in range(imfNo):
#            plt.subplot(r,c,num+2)
#            plt.plot(T, imfs[num], 'g')
#            plt.xlim((tMin, tMax))
#            plt.ylabel("Imf "+str(num+1))    
#        plt.tight_layout()
#        plt.show()
        return imfs
"""########################################################################"""
""" Les IMFs des signaux normaux et anormaux sur la dérivation0"""
"""Call the function imf_generate(); input(signal,N);output: imfs"""
imfs_globale_derv0=[]
for i in range(len(listderv0)) :
        signal_normal=listderv0[i][0]
        signal_anormal=listderv0[i][1]
        imfs_normale=imf_generate(signal_normal,N)
        imfs_anormale=imf_generate(signal_anormal,N)
        imfs_globale_derv0.append([imfs_normale,imfs_anormale])
        
imfs_globale_derv1=[]
for i in range(len(listderv1)) :
        signal_normal=listderv1[i][0]
        signal_anormal=listderv1[i][1]
        imfs_normale=imf_generate(signal_normal,N)
        imfs_anormale=imf_generate(signal_anormal,N)
        imfs_globale_derv1.append([imfs_normale,imfs_anormale])
"""LES IMFS significations : créer une fonction "significativity" qui calcul le
coefficient de variation qui est le rapport entre l'écart type et la moyenne;
si la moyenne est la même on travail sur la variance mais si la moyenne es différente
on travail sur le coefficient de variation qui est significatif s'il est >15%"""
# def : Calculate coefficient de corrélation variance between normal and abnormal
# A modifier en coefficient de variation=ecart type/moyenne( entre 12% et 15%)
#Si le coefficient > 15% ( hétérogénité)
def significativity (coeffiecient_variation):
    
        if ((coeffiecient_variation[0]>0.15 and coeffiecient_variation[1]>0.15) or (coeffiecient_variation[0]<0.15 and coeffiecient_variation[1]<0.15)):
            return False
        else :
            return True
def seuil (ecart):
    if ecart > 100 :
        return True
    else :
        return False
"""##########################################################################""" 
""" définir une fonction qui prend comme entrée imfs_globale_derv0  qui doit retourner les
IMFs significatives ainsi les compteurs des IMFs
imfs_globale_derv0 contient 2 arrays; on fait un boucle sur les 2 arrays (p01c-p02c)
On compare IMF[p01c] avec celle de p02c--Calculer le coefficient de variation puis on fait
appel de la fonction significativity -- initialiser un compteur à 0 et des compteurs pour tous les 
IMFs; A chaque itération si la IMF est significative on incrémente le compteur de l'IMf"""
"""IMF significatif in derv0"""
l_variance=[];l_standard_deviation=[];l_mean=[];
l_coeffiecient_variation=[];l_imfs_retenue_globale=[]
c0=c1=c2=c3=c4=c5=c6=c7=c8=c9=c10=c11=0
p0=p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=0
#var=l_imfs_retenue_globale[10]
#for i in range(len(l_imfs_retenue_globale)):
#    
#    if var[0] in l_imfs_retenue_globale[i] :
#        print("trouver")
l_vect=[]
max_lenght = 0
for i in range(len(imfs_globale_derv0)):
    signal=imfs_globale_derv0[i] 
    list_imfs_normaux=signal[0]
    list_imfs_anormaux=signal[1]
    imfs_retenue=[]
    l_temp_variance,l_temp_standard_deviation,l_temp_mean,l_temp_coeffiecient_variation = [],[],[],[]

    j = min(len(list_imfs_normaux),len(list_imfs_anormaux)) 
    vect = [t for t in range(0,j)]
    l_vect.append(vect) 
    if max_lenght<len(vect):
        max_lenght = len(vect) 
c0=c1=c2=c3=c4=c5=c6=c7=c8=c9=c10=c11=0 
for vect in l_vect:
    if 0 in  vect:
        c0+=1
    if 1 in vect:
        c1+=1
    if 2 in vect:
                   c2=c2+1
                   
    if 3 in vect:
                   c3=c3+1
                   
    if 4 in vect:
                   c4=c4+1 
                   
    if 5 in vect:
                   c5=c5+1 
                   
    if 6 in vect:
                   c6=c6+1 
                      
    if 7 in vect:
                   c7=c7+1 
                     
    if 8 in vect:
                   c8=c8+1 
                       
    if 9 in vect:
                   c9=c9+1 
    if 10 in vect:                  
                   c10=c10+1 
                      
    if 11 in vect:
                   c11=c11+1              
print("c0={},c1={},c2={},c3={},c4={},c5={},c6={},c7={},c8={},c9={},c10={},c11={}".format(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11))                
c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11=0,0,0,0,0,0,0,0,0,0,0,0  
ecart_0,ecart_1,ecart_2,ecart_3,ecart_4,ecart_5,ecart_6,ecart_7,ecart_8,ecart_9,ecart_10,ecart_11=0,0,0,0,0,0,0,0,0,0,0,0
for i in range(len(imfs_globale_derv0)):
       signal=imfs_globale_derv0[i]
       list_imfs_normaux=signal[0]
       list_imfs_anormaux=signal[1]
       imfs_retenue=[]
       l_temp_variance,l_temp_standard_deviation,l_temp_mean,l_temp_coeffiecient_variation,l_temp_ecart= [],[],[],[],[]
       for j in range(min(len(list_imfs_normaux),len(list_imfs_anormaux))):
           imf_normal=list_imfs_normaux[j]
           imf_anormal=list_imfs_anormaux[j]
           array_imf=np.array([imf_normal,imf_anormal])
           variance=np.var(array_imf,axis=1)
           l_temp_variance.append(variance)
           standard_deviation=np.std(array_imf,axis=1)
           l_temp_standard_deviation.append(standard_deviation)
           mean=np.mean(array_imf,axis=1)
           l_temp_mean.append(mean)
           coeffiecient_variation= standard_deviation/mean
           l_temp_coeffiecient_variation.append(coeffiecient_variation)
           ecart=int(((max(l_temp_variance[j])-min(l_temp_variance[j]))/min(l_temp_variance[j]))*100)
           l_temp_ecart.append(ecart)
           significativit=significativity(coeffiecient_variation)
           ecartbool=seuil(ecart)
           if (ecartbool==True):
               imfs_retenue.append([j,imf_normal,imf_anormal])
               if  j==0:
                   c_0+=1
#                   ecart_0=int(((max(l_temp_variance[0])-min(l_temp_variance[0]))/min(l_temp_variance[0]))*100)
               elif j==1:
                   c_1+=1
#                   ecart_1=int(((max(l_temp_variance[1])-min(l_temp_variance[1]))/min(l_temp_variance[1]))*100)
               elif j==2:
                   c_2+=1
#                   ecart_2=int(((max(l_temp_variance[2])-min(l_temp_variance[2]))/min(l_temp_variance[2]))*100)
               elif j==3:
                   c_3+=1
#                   ecart_3=int(((max(l_temp_variance[3])-min(l_temp_variance[3]))/min(l_temp_variance[3]))*100)
               elif j==4:
                   c_4+=1 
#                   ecart_4=int(((max(l_temp_variance[4])-min(l_temp_variance[4]))/min(l_temp_variance[4]))*100)
               elif j==5:
                   c_5+=1 
#                   ecart_5=int(((max(l_temp_variance[5])-min(l_temp_variance[5]))/min(l_temp_variance[5]))*100)
               elif j==6:
                   c_6+=1 
#                   ecart_6=int(((max(l_temp_variance[6])-min(l_temp_variance[6]))/min(l_temp_variance[6]))*100)  
               elif j==7:
                   c_7+=1 
#                   ecart_7=int(((max(l_temp_variance[7])-min(l_temp_variance[7]))/min(l_temp_variance[7]))*100)
               elif j==8:
                   c_8+=1 
#                   ecart_8=int(((max(l_temp_variance[8])-min(l_temp_variance[8]))/min(l_temp_variance[8]))*100)   
               elif j==9:
                   c_9+=1 
#                   ecart_9=int(((max(l_temp_variance[9])-min(l_temp_variance[9]))/min(l_temp_variance[9]))*100) 
               elif j==10:
                   c_10+=1 
#                   ecart_10=int(((max(l_temp_variance[10])-min(l_temp_variance[10]))/min(l_temp_variance[10]))*100) 
               elif j==11:
                   c_11+=1 
#                   ecart_11=int(((max(l_temp_variance[11])-min(l_temp_variance[11]))/min(l_temp_variance[11]))*100)
     #  print( variance[j])
                    
                      
       l_imfs_retenue_globale.append(imfs_retenue)
       l_variance.append(l_temp_variance)
       l_standard_deviation.append(l_temp_standard_deviation)
       l_mean.append(l_temp_mean)
       l_coeffiecient_variation.append(l_temp_coeffiecient_variation)
       
p0=int((c_0/c0)*100)
p1=int((c_1/c1)*100)
p2=int((c_2/c2)*100)
p3=int((c_3/c3)*100)
p4=int((c_4/c4)*100)
p5=int((c_5/c5)*100)
p6=int((c_6/c6)*100)
p7=int((c_7/c7)*100) 
p8=int((c_8/c8)*100)
p9=int((c_9/c9)*100) 
p10=int((c_10/c10)*100)
p11=int((c_11/c11)*100)
print("c_0={},c_1={},c_2={},c_3={},c_4={},c_5={},c_6={},c_7={},c_8={},c_9={},c_10={},c_11={}".format(c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11))  
print("p0={},\t\t p1={},\t\t p2={},\t\t p3={},\t\t p4={},\t\t p5={},\t\t p6={},\t\t p7={},\t\t p8={},\t\t p9={},\t\t p10={},\t\t p11={}".format(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11)) 
#print("ecart_0={},\t\t  ecart_1={},\t\t ecart_2={},\t\t ecart_3={},\t\t ecart_4={},\t\t ecart_5={},\t\t ecart_6={},\t\t ecart_7={},\t\t ecart_8={},\t\t ecart_9={},\t\t ecart_10={},\t\t ecart_11={}".format(ecart_0,ecart_1,ecart_2,ecart_3,ecart_4,ecart_5,ecart_6,ecart_7,ecart_8,ecart_9,ecart_10,ecart_11)) 
seuilpourcentage=[i for i in range(100,2100,100)]

#print(l_imfs_retenue_globale)
#count(imfs_globale_derv1)
""" Créer des définitions pour aplliquer CWT et pour sauvegarder des images"""
#def : Applying CWT          
def apply_CWT (imf,wavelet_family='mexh'):
    widths = np.arange(1, 31)
    cwtmatr, freqs = pywt.cwt(imf,widths, wavelet =wavelet_family)
    return cwtmatr,freqs
# def : save image
def save_image (path_image,name_IMF,cwtmatr):
    fig=plt.figure()
    plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
    plt.show()
    fig.savefig(path_image+''+name_IMF+'.png')
#Create folders for each CWT
def create_folder_CWt (CWT):
    for nb in range(len(imfs)):
        try :
            os.mkdir("CWT_IMF"+str(nb))
        except: 
            print("the folder exists") 
            path_imgs  = "CWT_IMF"+str(nb)
    return path_imgs
#Create foldres for csv file
def create_folder_csv ():
    for nb in range(len(imfs_retenue_globale)):
        try :
            os.mkdir("CSV"+str(nb))
        except: 
            print("the folder exists") 
            path_csv  = "CSV"+str(nb)
    return path_csv
#Variance entre p01c et p02c ,... 

#def read image
def proc_images(lowerIndex,upperIndex,imagePat,classZero,classOne,dim):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """
    x = []
    y = []
    WIDTH = dim
    HEIGHT = dim
    for img in imagePat[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y
#Apply CWT                           
#CWT
#Initialiser 2 listes vide
CWT_normaux=[]
CWT_anormaux=[]
path=""
try :
            os.mkdir("CWT_IMF")
except: 
            print("the folder exists") 
path= "CWT_IMF/" 
try :
            os.mkdir(path+'CWT_imf_deriv0')
except: 
            print("the folder exists") 
path1=path+'CWT_imf_deriv0/'

try :
            os.mkdir(path1+"imf_normale") 
except: 
            print("the folder exists")
try :
            os.mkdir(path1+"imf_anormale")
except: 
            print("the folder exists")
#for i in range(len(l_temp_ecart)):
for i in range(len(l_imfs_retenue_globale)):
     signal=l_imfs_retenue_globale[i] # imf p01c-p02c;p03c-p04c,...(normale;anormale)
     count=0
     for imf in signal :   #loop in the array signal contains imf normale and anormale and 3000 samples
         imf_normale=imf[0]   #First iterartion :first element (first IMF p01c) of the array signal
         imf_anormale=imf[1]  #First iterartionsecond element(first IMF p02c) of the array signal
         CWT0,freq0=apply_CWT(imf_normal)
         # Create folder for each signal
         CWT1,freq1=apply_CWT(imf_anormal)
         name_imf0='signal_'+str(i)+ '_imf_'+str(count)+"_normale"
         name_imf1='signal_'+str(i)+ '_imf_'+str(count)+"_anormale"
         path_normale=path1+"imf_normale/"
         path_anormale=path1+"imf_anormale/"
         save_image (path_normale,name_imf0,CWT0)
         save_image (path_anormale,name_imf1,CWT1)
         np.savetxt(path_normale+name_imf0+'_csv.csv',CWT0,delimiter=',')
         np.savetxt(path_anormale+name_imf1+'_csv.csv',CWT1,delimiter=',')
         count+=1
try :
            os.mkdir(path+'CWT_imf_deriv1')
except: 
            print("the folder exists") 
path1=path+'CWT_imf_deriv1/'

try :
            os.mkdir(path1+"imf_normale") 
except: 
            print("the folder exists")
try :
            os.mkdir(path1+"imf_anormale")
except: 
            print("the folder exists")
for i in range(len(l_imfs_retenue_globale)):
     signal=l_imfs_retenue_globale[i] # imf p01c-p02c;p03c-p04c,...(normale;anormale)
     count=0
     for imf in signal :   #loop in the array signal contains imf normale and anormale and 3000 samples
         imf_normale=imf[0]   #First iterartion :first element (first IMF p01c) of the array signal
         imf_anormale=imf[1]  #First iterartionsecond element(first IMF p02c) of the array signal
         CWT0,freq0=apply_CWT(imf_normal)
         # Create folder for each signal
         CWT1,freq1=apply_CWT(imf_anormal)
         name_imf0='signal_'+str(i)+ '_imf_'+str(count)+"_normale"
         name_imf1='signal_'+str(i)+ '_imf_'+str(count)+"_anormale"
         path_normale=path1+"imf_normale/"
         path_anormale=path1+"imf_anormale/"
         save_image (path_normale,name_imf0,CWT0)
         save_image (path_anormale,name_imf1,CWT1)
         np.savetxt(path_normale+name_imf0+'_csv.csv',CWT0,delimiter=',')
         np.savetxt(path_anormale+name_imf1+'_csv.csv',CWT1,delimiter=',')
         count+=1
"""Concaténation des signaux"""
#supeprosition des IMFs
imfts_normale=[]
imfts_anormale=[]
for i in range(len(l_imfs_retenue_globale)):
     signalf=l_imfs_retenue_globale[i] # imf p01c-p02c;p03c-p04c,...(normale;anormale)
     imft_normale=[]
     imft_anormale=[]
     for imf in signalf :   #loop in the array signal contains imf normale and anormale and 3000 samples
         imf_normale=imf[1]   #First iterartion :first element (first IMF p01c) of the array signal
         imf_anormale=imf[2]  #First iterartionsecond element(first IMF p02c) of the array signal
         imft_normale=np.concatenate([imft_normale,imf_normale])
         imft_anormale=np.concatenate([imft_anormale,imf_anormale])
     imfts_normale.append(imft_normale)
     imfts_anormale.append(imft_anormale)
"""

         #         path_normale=path1+"imf_normale/"
#         path_anormale=path1+"imf_anormale/"
#         
#         count+=1
#construire une base de données labelisée
#deux entrées :
#images = cwtmatr / labels = 0 pour cwtmatr nonAF et 1 pour cwtmatr AF
try :
            os.mkdir(path1+"csv_files") 
except: 
            print("the folder exists")
signal_entier=[]
for i in range(len(signal)):
    signal_entier=np.concatenate([signal_normale_0[i].T[0]],[signal_normale_0[i].T[1]],[signal_anormale_0[i].T[0]],[signal_anormale_1[i].T[1]])
    While len(signal_entier)>38400):
        signal[i]=76800-signal_entier[i]
        signal='csv'+str(j)
        path=path1+"signal/"
        np.savetxt(path+signal+'_csv.csv',delimiter=',')"""


# Construire un réseau de neurones profonds  CNN 
""" Application du modèle CNN"""   
"""Définition modèle de classification en vue de la prévision""" 
def classifier(shape,num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=shape,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))
    #Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    model.summary()
    return model 

#import data
img_normaux_path= glob(path_normale+'/*.png')
img_anormaux_path= glob(path_anormale+'/*.png')
imagePat=img_normaux_path +img_anormaux_path
classZero=img_normaux_path
classOne=img_anormaux_path
dim=64
lowerIndex=0 
upperIndex=100000
X,y=proc_images(lowerIndex,upperIndex,imagePat,classZero,classOne,dim)
X=np.array(X)
X=X/255
#split data : train , validation , test
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=0,test_size=0.2)
y_train_2digit=tf.keras.utils.to_categorical(y_train, num_classes=2, dtype="float32")
y_val_2digit=tf.keras.utils.to_categorical(y_val, num_classes=2, dtype="float32")
#classificator model
batch_size = 64
epochs = 20
num_classes =2
model=classifier(X_train[0].shape,2)
# training
model.fit(X_train,y_train_2digit,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val,y_val_2digit))
predict_train=model.predict(X_train)
#evaluation
test_eval = model.evaluate(X_train, y_train_2digit, verbose=0)
print('Train loss:', test_eval[0])
print('Train accuracy:', test_eval[1])
test_eval = model.evaluate(X_val, y_val_2digit, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
