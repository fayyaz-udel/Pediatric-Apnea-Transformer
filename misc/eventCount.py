import glob
import os
import pandas as pd

path = r"D:\nchsdb\sleep_data\*.tsv"
files = glob.glob(path)
length = len(files)


for i in range(length):
    if i ==0:
        df = pd.read_csv(files[i], sep='\t')
    else:
        df = pd.concat([df, pd.read_csv(files[i], sep='\t')], axis=0)

item_counts = df["description"].value_counts()
print(item_counts)