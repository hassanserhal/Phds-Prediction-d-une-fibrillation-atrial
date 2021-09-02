import matplotlib.pyplot as plt
import wfdb
import numpy as np
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras
import tensorflow as tf
import pywt 
import cv2
from glob import glob
from sklearn.model_selection import train_test_split 
import os

database_name  = 'PAF'
from test_emd import *
if __name__ == "__main__":
    import pylab as plt
# import files in database folder

"""Déterminer les IMF les plus importantes pour la FA
step1:
    Importer p01c(5min sans FA) et p02c (5 min avec FA)
step2: appliquer l'approche EMD
step3:
    Une boucle qui parcours les IMFs résultants
    A chaque itération,calculer la variance entre IMFi de p01c et IMFi de p02c
        si la variance est significative on retient les IMFs
        Si non on passe à l'IMF suivant"""
## Import data
filenames = glob(database_name+'/p*.dat')
#def import data
def list_record(filenames):
    records_name =[]
    records = []
    for i in range(len(filenames)):
        records_name.append(filenames[i][4:-4])
        record_name = filenames[i][4:-4]
        record = wfdb.rdsamp(database_name+'/'+record_name, sampto=N)
        #annotation = wfdb.rdann('PAF/n01', 'atr', sampto=3000)
#        wfdb.plotrec(record,
#                 title='Record'+ record_name +'from '+database_name,
#                 timeunits = 'seconds', figsize = (10,4), ecggrids = 'all')
        records.append(record)
    return records

# def generate imf
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
        c = 1
        r = np.ceil((imfNo+1)/c)
        plt.ioff()
        plt.subplot(r, c, 1)
        plt.plot(T, S, 'r')
        plt.xlim((tMin, tMax))
        plt.title("Original signal")
    
        for num in range(imfNo):
            plt.subplot(r,c,num+2)
            plt.plot(T, imfs[num], 'g')
            plt.xlim((tMin, tMax))
            plt.ylabel("Imf "+str(num+1))
    
        plt.tight_layout()
        plt.show()
        return imfs

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
    for nb in range(len(imfs_retenue_globale1)):
        try :
            os.mkdir("CSV"+str(nb))
        except: 
            print("the folder exists") 
            path_csv  = "CSV"+str(nb)
    return path_csv
#Variance entre p01c et p02c ,... 
def significativity (coeffiecient_variation):
    
        if ((coeffiecient_variation[0]>0.15 and coeffiecient_variation[1]>0.15) or (coeffiecient_variation[0]<0.15 and coeffiecient_variation[1]<0.15)):
            return False
        else :
            return True
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
#def create model

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
######################################
N=3000
if database_name == 'PAF':   
                
    list_test=[]
    list_normale=[]
    list_anormale=[]
    listderv0=[]
    listderv1=[]
    count=0
    for i in range(1,len(filenames),2): 
        list_test.append(filenames[i])
        if count%2==0 :
            list_normale.append(filenames[i])
        else:
            list_anormale.append(filenames[i])
        print(i)
        count+=1
    list_normale_values=list_record(list_normale)
    list_anormale_values=list_record(list_anormale)
# EMD p01c-p02c
for i in range(min(len(list_normale),len(list_anormale))):
    print('{}/{}'.format(list_normale[i],list_anormale[i]))
    signal_normale_0=list_normale_values[i].p_signals.T[0]
    signal_normale_1=list_normale_values[i].p_signals.T[1]
    signal_anormale_0=list_anormale_values[i].p_signals.T[0]
    signal_anormale_1=list_anormale_values[i].p_signals.T[1]
    listderv0.append([signal_normale_0,signal_anormale_0])
    listderv1.append([signal_normale_1,signal_anormale_1])

imfs_globale_derv1=[]
for i in range(len(listderv1)) :
    signal_normal=listderv1[i][0]
    signal_anormal=listderv1[i][1]
    imfs_normale=imf_generate(signal_normal,N)
    imfs_anormale=imf_generate(signal_anormal,N)
    imfs_globale_derv1.append([imfs_normale,imfs_anormale])
  
list_variance1=[]
list_standard_deviation1=[]
list_mean1=[]
list_coeffiecient_variation1=[]
imfs_retenue_globale1=[]
for i in range(len(imfs_globale_derv1)) :
  
   signal1=imfs_globale_derv1[i]
   list_imfs_normaux1=signal1[0]
   list_imfs_anormaux1=signal1[1]
   imfs_retenue1=[]
   for j in range(min(len(list_imfs_normaux1),len(list_imfs_anormaux1))):
       imf_normal1=list_imfs_normaux1[j]
       imf_anormal1=list_imfs_anormaux1[j]
       array_imf1=np.array([imf_normal1,imf_anormal1])
       variance1=np.var(array_imf1,axis=0)
       list_variance1.append(variance1)
       standard_deviation1=np.std(array_imf1,axis=0)
       list_standard_deviation1.append(standard_deviation1)
       mean1=np.mean(array_imf1,axis=0)
       list_mean1.append(mean1)
       coeffiecient_variation1=standard_deviation1/mean1
       list_coeffiecient_variation1.append(coeffiecient_variation1)
       significativit1=significativity(coeffiecient_variation1)
       if significativit1== True :
           imfs_retenue1.append([j,imf_normal1,imf_anormal1])
         
   imfs_retenue_globale1.append(imfs_retenue1)       
                           
#CWT
CWT_normaux=[]
CWT_anormaux=[]
path=""

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
for i in range(len(imfs_retenue_globale1)):
     signal=imfs_retenue_globale1[i] # imf p01c-p02c;p03c-p04c,...(normale;anormale)
     count=0
     for imf in signal :   #loop in the array signal contains imf normale and anormale and 3000 samples
         imf_normale=imf[0]   #First iterartion :first element (first IMF p01c) of the array signal
         imf_anormale=imf[1]  #First iterartionsecond element(first IMF p02c) of the array signal
         CWT0,freq0=apply_CWT(imf_normale)
         # Create folder for each signal
         CWT1,freq1=apply_CWT(imf_anormale)
         name_imf0='signal_'+str(i)+ '_imf_'+str(count)+"_normale"
         name_imf1='signal_'+str(i)+ '_imf_'+str(count)+"_anormale"
         path_normale=path1+"imf_normale/"
         path_anormale=path1+"imf_anormale/"
         save_image (path_normale,name_imf0,CWT0)
         save_image (path_anormale,name_imf1,CWT1)
         np.savetxt(path_normale+name_imf0+'_csv.csv',CWT0,delimiter=',')
         np.savetxt(path_anormale+name_imf1+'_csv.csv',CWT1,delimiter=',')
         count+=1

#supeprosition des IMFs
imfts_normale=[]
imfts_anormale=[]
c0=c1=c2=c3=c4=c5=c6=c7=c8=c9=c10=c11=c12=0
p0=p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=0
for i in range(len(imfs_retenue_globale1)):
    if (imfs_retenue_globale1[i]==imfs_retenue_globale1[0]):
        c0=c0+1
        p0=(c0/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[1]):
        c1=c1+1
        p1=(c1/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[2]):
        c2=c2+1
        p2=(c2/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[3]):
        c3=c3+1
        p3=(c3/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[4]):
        c4=c4+1
        p4=(c4/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[5]):
        c5=c5+1
        p5=(c5/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[6]):
        c6=c6+1
        p6=(c6/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[7]):
        c7=c7+1
        p7=(c7/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[8]):
        c8=c8+1
        p8=(c8/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[9]):
        c9=c9+1
        p9=(c9/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[10]):
        c10=c10+1
        p10=(c10/len(imfs_retenue_globale1))*100
    elif (imfs_retenue_globale1[i]==imfs_retenue_globale1[11]):
        c11=c11+1
        p11=(c11/len(imfs_retenue_globale1))*100
    else: 
        c12=c12+1
        p12=(c12/len(imfs_retenue_globale1))*100
print(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12)
print(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12)
signalf=imfs_retenue_globale1[i] # imf p01c-p02c;p03c-p04c,...(normale;anormale)
imft_normale=[]
imft_anormale=[]
for imf in signalf :   #loop in the array signal contains imf normale and anormale and 3000 samples
     imf_normale=imf[0]   #First iterartion :first element (first IMF p01c) of the array signal
     imf_anormale=imf[1]  #First iterartionsecond element(first IMF p02c) of the array signal
     imft_normale=np.concatenate([imft_normale,imf_normale])
     imft_anormale=np.concatenate([imft_anormale,imf_anormale])
imfts_normale.append(imft_normale)
imfts_anormale.append(imft_anormale)
#CWT 
try :
            os.mkdir(path1+"imff_normale") 
except: 
            print("the folder exists")
try :
            os.mkdir(path1+"imff_anormale")
except: 
            print("the folder exists")
for i in range(len(imfts_normale)):
       cwtf0,freqf0=apply_CWT(imfts_normale[i],wavelet_family='mexh')
       name_imff0='signal_'+str(i)+ '_imff_'+str(i)+"_normale"
       path_normale=path1+"imff_normale/"
       save_image (path_normale,name_imff0,cwtf0)
       np.savetxt(path_normale+name_imff0+'_csv.csv',cwtf0,delimiter=',')
for j in range(len(imfts_anormale)):
       cwtf1,freqf1=apply_CWT(imfts_anormale[j],wavelet_family='mexh')
       name_imff1='signal_'+str(j)+ '_imff_'+str(j)+"_anormale"
       path_anormale=path1+"imff_anormale/"
       save_image (path_anormale,name_imff1,cwtf1)
       np.savetxt(path_anormale+name_imff1+'_csv.csv',cwtf1,delimiter=',')
# Apply model
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
train_eval = model.evaluate(X_train, y_train_2digit, verbose=0)
print('Train loss:', train_eval[0])
print('Train accuracy:', train_eval[1])

test_eval = model.evaluate(X_val, y_val_2digit, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])   

