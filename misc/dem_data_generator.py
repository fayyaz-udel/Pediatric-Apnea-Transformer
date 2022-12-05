import pandas as pd

ss = pd.read_csv(r"C:\SLEEP_STUDY.csv")
ss['date'] = pd.to_datetime(ss["SLEEP_STUDY_START_DATETIME"])
ss["age"] = ss["AGE_AT_SLEEP_STUDY_DAYS"].div(30.41).round(1)

m = pd.read_csv(r"C:\MEASUREMENT.csv") # Measurement
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

ss = ss[['id', 'bmi', 'age', 'diff']]

ss.to_csv("result.csv")
