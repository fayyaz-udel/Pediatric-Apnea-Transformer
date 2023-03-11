import pandas as pd

ss = pd.read_csv(r"D:\nchsdb\health_data\SLEEP_STUDY.csv") #Sleep Study
demo = pd.read_csv(r"D:\nchsdb\health_data\DEMOGRAPHIC.csv")
m = pd.read_csv(r"D:\nchsdb\health_data\MEASUREMENT.csv") # Measurement

ss = ss.merge(demo, left_on='STUDY_PAT_ID', right_on='STUDY_PAT_ID')

ss['date'] = pd.to_datetime(ss["SLEEP_STUDY_START_DATETIME"])
ss["age"] = ss["AGE_AT_SLEEP_STUDY_DAYS"].div(30.41).round(1)

m = m[m["MEAS_TYPE"] == "BMIPCT"]
m['date'] = pd.to_datetime(m['MEAS_RECORDED_DATETIME'])

for index, row in ss.iterrows():
    tmp = m[m["STUDY_PAT_ID"] == row['STUDY_PAT_ID']]
    #tmp.loc[:,'diff'] = ((row['date']- tmp['date']).dt.days).abs()
    tmp = tmp.assign(diff=((row['date']- tmp['date']).dt.days).abs())
    tmp = tmp.sort_values(by=['diff'])

    if len(tmp) > 0:
        ss.at[index, 'bmi'] = tmp.iloc[0]['MEAS_VALUE_NUMBER']
        ss.at[index, 'diff'] = tmp.iloc[0]['diff']


ss['id'] = ss['STUDY_PAT_ID'].astype(str) + "_" + ss['SLEEP_STUDY_ID'].astype(str)


ss['PCORI_GENDER_CD'] = ss['PCORI_GENDER_CD'].map({'F': 0, 'M': 1, 'UN': -1})
ss = ss[['id', 'bmi', 'age', 'PCORI_GENDER_CD', "PCORI_RACE_CD", 'PCORI_HISPANIC_CD']]
ss = ss.join(pd.get_dummies(ss["PCORI_RACE_CD"]))
ss =ss.drop(columns=["PCORI_RACE_CD"])
ss['PCORI_HISPANIC_CD'] = ss['PCORI_HISPANIC_CD'].map({'N': 0, 'Y': 1, 'UN': -1, 'NI': -1})
ss=ss.fillna(0)
ss['bmi'] = ss['bmi']/100
ss['age'] = ss['age']/300


ss.to_csv("result.csv", index=False)
