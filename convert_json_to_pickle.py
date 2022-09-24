import json
import pickle



with open('collected/gesture_names_1_74.json', 'r') as f1:
    data1 = json.load(f1)

with open('collected/gesture_names_75_150.json', 'r') as f1:
    data2 = json.load(f1)

with open('collected/gesture_names_150_225.json', 'r') as f1:
    data3 = json.load(f1)

with open('collected/gesture_names_226_300.json', 'r') as f1:
    data4 = json.load(f1)

with open('collected/gesture_names_300_375.json', 'r') as f1:
    data5 = json.load(f1)

with open('collected/gesture_names_376_448.json', 'r') as f1:
    data6 = json.load(f1)

data_gesture_names = data1+data2+data3+data4+data5+data6



with open('collected/gestures_1_74.json', 'r') as f1:
    data01 = json.load(f1)

with open('collected/gestures_75_150.json', 'r') as f1:
    data02 = json.load(f1)

with open('collected/gestures_150_225.json', 'r') as f1:
    data03 = json.load(f1)

with open('collected/gestures_226_300.json', 'r') as f1:
    data04 = json.load(f1)

with open('collected/gestures_300_375.json', 'r') as f1:
    data05 = json.load(f1)

with open('collected/gestures_376_448.json', 'r') as f1:
    data06 = json.load(f1)

data_gestures = data01+data02+data03+data04+data05+data06

# print(data_gesture_names)


gestNames = ['Nodding', 'Stop sign',
'Thumbs down',
'Waving',
'Pointing',
'Calling someone',
'Thumbs up',
'Wave someone away',
'Shaking head']

gestNames_not_required = ['Nodding', 'Shaking head']

new_data_gesture_names = []
new_new_data_gesture_names = []
new_data_gestures = []

for ges in data_gesture_names:
    pos = gestNames.index(ges)
    pos1 = pos - 1
    # print(ges, "..................", gestNames[pos1])
    new_data_gesture_names.append(gestNames[pos1])

print(new_data_gesture_names)

for gestures in new_data_gesture_names:
    if gestures not in gestNames_not_required:
        pos = new_data_gesture_names.index(gestures)
        new_data_gestures.append(data_gestures[pos])

        new_new_data_gesture_names.append(gestures)

print(new_new_data_gesture_names)
print(len(new_data_gesture_names), len(data_gestures))
print(len(new_new_data_gesture_names), len(new_data_gestures))

with open('collected/gestures.pkl', 'wb') as f4:
    pickle.dump(new_new_data_gesture_names,f4)
    pickle.dump(new_data_gestures,f4)

