#!/usr/bin/env python

print "~ Yael and Boris Predict ~"

print "Load SETTINGS.json file"


import json
with open('./SETTINGS.json') as f:
    SETTINGS = json.load(f)






import man_data
import pickle
import pandas as pd



print "load the model from: "+ SETTINGS['MODEL_PATH'] + SETTINGS['MODEL_FILE']
clf_lgb = pickle.load(open(SETTINGS['MODEL_PATH']+ SETTINGS['MODEL_FILE'], 'rb'))



test_file = SETTINGS['TEST_DATA_CLEAN_PATH']
print "read the test data frame from: " + SETTINGS['TEST_DATA_CLEAN_PATH']
test_frame = pd.read_csv(test_file)
test_frame = man_data.downcast(test_frame)
print "change the \'experiment\' attribute as numbers.."
test_frame['experiment'] = test_frame['experiment'].map({'CA': 2, 'DA': 3, 'SS': 1})
X_test=test_frame.drop(['id'],axis=1)
y_test=[]



print "start predict...."
lgb_prob = clf_lgb.predict_proba(X_test)





sub = pd.DataFrame(lgb_prob, columns=['A', 'B', 'C', 'D'])
sub['id'] = test_frame['id']
cols = sub.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub = sub[cols]
print "write the subbmision to: " + SETTINGS['SUBMISSION_DIR'] + SETTINGS['SUBMISSION_FILE']
sub.to_csv(SETTINGS['SUBMISSION_DIR'] + SETTINGS['SUBMISSION_FILE'], index=False)


import zipfile

with zipfile.ZipFile(SETTINGS['SUBMISSION_DIR'] + SETTINGS['SUBMISSION_FILE_ZIP'],'w') as myzip:
    myzip.write(SETTINGS['SUBMISSION_DIR'] + SETTINGS['SUBMISSION_FILE'])