import glob
import os
import pandas as pd

path = r"D:\nchsdb\sleep_data\*.tsv"
files = glob.glob(path)
length = len(files)


for i in range(length):
    df = pd.read_csv(files[i], sep='\t')
    ic = df["description"].value_counts()
    neg = ic.get("Sleep stage N2", 0) + ic.get("Sleep stage N3", 0) + ic.get("Sleep stage R", 0) + ic.get("Sleep stage ?", 0) + ic.get("Sleep stage N1", 0)

    pos = ic.get("Obstructive Hypopnea", 0) + ic.get("Obstructive Apnea", 0) + ic.get("Hypopnea", 0) + ic.get("Central Apnea", 0) + ic.get("Mixed Apnea", 0)
    print(str(files[i]), ", " +  str(neg), "," + str(pos))

