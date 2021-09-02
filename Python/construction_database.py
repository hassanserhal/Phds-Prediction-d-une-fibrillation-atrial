# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:52:38 2021

@author: hp
"""

"""
Déterminer les IMF les plus importantes pour la FA
step1:
    Importer p01c(5min sans FA) et p02c (5 min avec FA)
step2: appliquer l'approche EMD
step3:
    Une boucle qui parcours les IMFs résultants
    A chaque itération,calculer la variance entre IMFi de p01c et IMFi de p02c
        si la variance est significative on retient les IMFs
        Si non on passe à l'IMF suivant
    


Création de la base de donnée
step1:
    load p01;p01c;p02;p02c
Une boucle qui va parcouruir tous les signaux de la base avec un pas de 4

step2:
    Concetenation  entre p01 , p01c et p02 (1h 5 min)
    
step 3:
Créer une boucle
Créer une liste :
     premier element : p01+p01c +p02 ; à chaque itération on retranche 10min(10*60*128Hz=76800)
     du début de la liste;
     On reste dans la boucle jusqu'à 5 min  avant la FA ( dernier 5 min de p02)
     résultats: on aura  7 labels  alors on obtient 7 bases de données
     