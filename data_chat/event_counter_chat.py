import glob
import xml.etree.ElementTree as ET

import pandas as pd

'''
<ScoredEvent>
    0 <EventType>Respiratory|Respiratory</EventType>
    1 <EventConcept>Hypopnea|Hypopnea</EventConcept>
    2 <Start>2547</Start>
    3 <Duration>6.2</Duration>
    4 <SignalLocation>Airflow</SignalLocation>
</ScoredEvent>
'''

input_path = "E:\\chat\\polysomnography\\annotations-events-nsrr\\baseline\\"


def process_xml(input_file, df):
    nsrr = ET.parse(input_file).findall('ScoredEvents')
    for event in nsrr[0]:
        if event[0].text in ["Respiratory|Respiratory", "Stages|Stages"]:
            description = event[1].text.split("|")[0]
            onset = int(float(event[2].text))
            duration = int(float(event[3].text))

            df = df.append({'onset': onset, 'duration': duration, 'description': description}, ignore_index=True)
    return df


if __name__ == "__main__":
    df = pd.DataFrame(columns=['onset', 'duration', 'description'])

    for input_file in glob.glob(input_path + "*.xml"):
        df = process_xml(input_file, df)
    df["description"].value_counts().to_csv("E:\\event_count_CHAT.csv")
