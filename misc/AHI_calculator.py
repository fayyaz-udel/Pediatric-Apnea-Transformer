import os
import sleep_study as ss
import numpy as np
import pandas as pd

def identity(df):
    return df

ss.__init__()
for i, name in enumerate(ss.data.study_list):
    raw = ss.data.load_study(name, identity)

    df = pd.read_csv(os.path.join(ss.data_dir, 'Sleep_Data', name+ '.tsv'), sep='\t')
    vcs = df['description'].value_counts()
    try:
        w = vcs.loc['Sleep stage W']
    except Exception:
        w = 0
    try:
        n1 = vcs.loc['Sleep stage N1']
    except Exception:
        n1 = 0
    try:
        n2 = vcs.loc['Sleep stage N2']
    except Exception:
        n2 = 0
    try:
        n3 = vcs.loc['Sleep stage N3']
    except Exception:
        n3 = 0
    try:
        r = vcs.loc['Sleep stage R']
    except Exception:
        r = 0
    try:
        n5 = vcs.loc['Sleep stage ?']
    except Exception:
        n5 = 0
    ####################################################################################################################
    h_count = 0
    try:
        h_count += vcs.loc["Obstructive Hypopnea"]
    except Exception:
        pass
    try:
        h_count += vcs.loc["Hypopnea"]
    except Exception:
        pass
    try:
        h_count += vcs.loc["hypopnea"]
    except Exception:
        pass
    try:
        h_count += vcs.loc["Mixed Hypopnea"]
    except Exception:
        pass
    try:
        h_count += vcs.loc["Central Hypopnea"]
    except Exception:
        pass
    ####################################################################################################################
    a_count = 0
    try:
        a_count += vcs.loc["Obstructive Apnea"]
    except Exception:
        pass
    try:
        a_count += vcs.loc["Central Apnea"]
    except Exception:
        pass
    try:
        a_count += vcs.loc["Mixed Apnea"]
    except Exception:
        pass
    try:
        a_count += vcs.loc["apnea"]
    except Exception:
        pass
    try:
        a_count += vcs.loc["obstructive apnea"]
    except Exception:
        pass
    try:
        a_count += vcs.loc["central apnea"]
    except Exception:
        pass
    try:
        a_count += vcs.loc["Apnea"]
    except Exception:
        pass

    sleep_length = (n1 + n2 + n3 + r + n5)/120
    e_count = a_count + h_count
    if sleep_length > 0:
        AHI = round(e_count/sleep_length, 2)

        print(name + ", " + str(round(sleep_length, 2)) + ", " + str(e_count) + ", " + str(AHI))