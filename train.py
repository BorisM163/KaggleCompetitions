#!/usr/bin/env python

print "~ Yael and Boris Train Model ~"

print "Load SETTINGS.json file"


import json
with open('SETTINGS.json') as f:
    SETTINGS = json.load(f)

import man_data
import lightgbm as lgb
import pandas as pd
import pickle
train_data_file = SETTINGS['TRAIN_DATA_CLEAN_PATH']
print "read the train data frame from: " + SETTINGS['TRAIN_DATA_CLEAN_PATH']
train_data_frame = pd.read_csv(train_data_file)
train_data_frame = man_data.downcast(train_data_frame)
print "change the \'event\' attribute as numbers.."
train_data_frame['event'] = train_data_frame['event'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
print "change the \'experiment\' attribute as numbers.."
train_data_frame['experiment'] = train_data_frame['experiment'].map({'CA': 2, 'DA': 3, 'SS': 1})


X_train = train_data_frame.drop(["event"], axis=1)
y_train = train_data_frame.loc[:,["event"]]





print "start training.. "
lgbGossModel = lgb.LGBMClassifier(boosting_type='goss',n_estimators=100, learning_rate=0.01)

print "start fit..."
lgbGossModel.fit(X_train, y_train)

print "save the model to: "+ SETTINGS['MODEL_PATH'] + SETTINGS['MODEL_FILE']
pickle.dump(lgbGossModel, open(SETTINGS['MODEL_PATH'] + SETTINGS['MODEL_FILE'],'wb'))


