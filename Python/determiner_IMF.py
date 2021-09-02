# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:07:41 2021

@author: hp
"""

"""Déterminer les IMF les plus importantes pour la FA
step1:
    Importer p01c(5min sans FA) et p02c (5 min avec FA)
step2: appliquer l'approche EMD
step3:
    Une boucle qui parcours les IMFs résultants
    A chaque itération,calculer la variance entre IMFi de p01c et IMFi de p02c
        si la variance est significative on retient les IMFs
        Si non on passe à l'IMF suivant"""

import wfdb
import matplotlib.pyplot as plt
from glob import glob 



def import_record (filenames,Nb_samples):
    
    records_name =[]
    records = []
    if database_name == 'PAF':
        for i in range(len(filenames)):
            records_name.append(filenames[i][4:-4])
            
            record_name = filenames[i][4:-4]
            record = wfdb.rdsamp(database_name+'/'+record_name, sampto=Nb_samples)
            #annotation = wfdb.rdann('PAF/n01', 'atr', sampto=3000)
            wfdb.plotrec(record,
                     title='Record'+ record_name +'from '+database_name,
                     timeunits = 'seconds', figsize = (10,4), ecggrids = 'all')
            records.append(record)
    return records
           

database_name  = 'PAF'
# import files in database folder
filenames = glob(database_name+'/p0*.dat')
Nb_samples=3000

records=import_record (filenames,Nb_samples)
from test_emd import *


if __name__ == "__main__":
    import pylab as plt

    # Logging options
    #logging.basicConfig(level=logging.DEBUG)
    listSignals=[derv1_AvantAF,derv1_AF]
    for i in listSignals :
        
        # EMD options
        max_imf = -1
        DTYPE = np.float64
    
        # Signal options
        
        S = i
        S = S.astype(DTYPE)
        print("Input S.dtype: " + str(S.dtype))
        tMin, tMax = 0, 3000
#        T = np.linspace(tMin, tMax, N, dtype=DTYPE)
        T = [ii for ii in range(1,N+1)]
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
#CWT 
import pywt        
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.cwt(imfs[2],widths, wavelet ='mexh')

plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP

plt.show()
#signal0 = records[100]
#derv1 = signal0.p_signals.T[0]
#derv2 = signal0.p_signals.T[1]
#    
#    
#    
#plt.plot(derv1)
#
##pour PAF : pas besoin de diviser en intervalle - 
##calcul d'indices 
##interval1 = [Normal]
#
##interval2 = [before-AF]
##interval3 = [AF]
##interval4 = [After- AF]
#
#
## import exemple
#signal0 = records[130]
#derv1_AvantAF = signal0.p_signals.T[0]
#derv2_AvantAF = signal0.p_signals.T[1]
#import matplotlib.pyplot as plt
#plt.plot(derv2_AvantAF)
#
## import exemple
#signal0 = records[131]
#derv1_AF = signal0.p_signals.T[0]
#derv2_AF = signal0.p_signals.T[1]
#import matplotlib.pyplot as plt
#plt.plot(derv2_AF)
#
## EMD 
#from test_EMD import *
#
#
#if __name__ == "__main__":
#    import pylab as plt
#
#    # Logging options
#    #logging.basicConfig(level=logging.DEBUG)
#    listSignals=[derv1_AvantAF,derv1_AF]
#    for i in listSignals :
#        
#        # EMD options
#        max_imf = -1
#        DTYPE = np.float64
#    
#        # Signal options
#        
#        S = i
#        S = S.astype(DTYPE)
#        print("Input S.dtype: " + str(S.dtype))
#        tMin, tMax = 0, 3000
##        T = np.linspace(tMin, tMax, N, dtype=DTYPE)
#        T = [ii for ii in range(1,N+1)]
#        # Prepare and run EMD
#        emd = EMD()
#        emd.FIXE_H = 5
#        emd.nbsym = 2
#        emd.spline_kind = 'cubic'
#        emd.DTYPE = DTYPE
#    
#        imfs = emd.emd(S, T, max_imf)
#        imfNo = imfs.shape[0]
#    
#        # Plot results
#        c = 1
#        r = np.ceil((imfNo+1)/c)
#    
#        plt.ioff()
#        plt.subplot(r, c, 1)
#        plt.plot(T, S, 'r')
#        plt.xlim((tMin, tMax))
#        plt.title("Original signal")
#    
#        for num in range(imfNo):
#            plt.subplot(r,c,num+2)
#            plt.plot(T, imfs[num], 'g')
#            plt.xlim((tMin, tMax))
#            plt.ylabel("Imf "+str(num+1))
#    
#        plt.tight_layout()
#        plt.show()
##CWT 
#import pywt        
#widths = np.arange(1, 31)
#cwtmatr, freqs = pywt.cwt(imfs[2],widths, wavelet ='mexh')
#
#plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
#
#plt.show()