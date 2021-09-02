import wfdb
#from glob import glob 
#filenames = glob(database_name+'/*.dat')

#database_name  = 'PAF'
with open("C:/Users/hp/Desktop/Codes FA/PAF/n01.dat") as f:
    with open("n01.csv", "w") as f1:
        for line in f:
            f1.write(line)

