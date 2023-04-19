import glob
import os

import mne
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

'''
<ScoredEvent>
    0 <EventType>Respiratory|Respiratory</EventType>
    1 <EventConcept>Hypopnea|Hypopnea</EventConcept>
    2 <Start>2547</Start>
    3 <Duration>6.2</Duration>
    4 <SignalLocation>Airflow</SignalLocation>
</ScoredEvent>
'''

input_path = "D:\\ccshs\\polysomnography\\annotations-events-nsrr\\"
output_path = "D:\\ccshs\\polysomnography\\edfs\\"  # baseline\\"


def process_xml(input_file, output_file):
    df = pd.DataFrame(columns=['onset', 'duration', 'description'])

    nsrr = ET.parse(input_file).findall('ScoredEvents')
    # events = []
    for event in nsrr[0]:
        if event[0].text in ["Respiratory|Respiratory", "Stages|Stages"]:
            description = event[1].text.split("|")[0]
            onset = int(float(event[2].text))
            duration = int(float(event[3].text))
            df = df.append({'onset': onset, 'duration': duration, 'description': description}, ignore_index=True)

    df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    for input_file in glob.glob(input_path + "*.xml"):
        output_file = output_path + os.path.basename(input_file).split(".")[0] + ".tsv"
        process_xml(input_file, output_file)
