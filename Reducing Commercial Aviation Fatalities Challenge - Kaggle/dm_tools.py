import plots

import itertools
import pandas as pd     #read csv files
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

#only numerical
def trees_acuracy(clf_list, X,y,label_tist):
    for clf, label in zip(clf_list,label_tist):
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))

def distributions_test2train(data_frame,test_frame,parameters):
    for par, i in zip(parameters, range(len(parameters))):
        print par
        plots.distribution([data_frame[par], test_frame[par]], label_vec=["train-" + par, "test-" + par], xlabel="x",
                           title=par + " distribution", ind=i)


#build a desicion tree and plot classifsiffication if ind >0
def decision_tree_pred(typeT, rand_state, max_depth, min_sample_leaf,X_train, y_train,X_test,y_test,ind):
    # clf_gini.tree_.apply(x_arr) -> the corresponding leaf node id for each of my training data point
    # # print(clf_gini.tree_.node_count)
    clf = DecisionTreeClassifier(criterion = typeT, random_state = rand_state ,max_depth=max_depth, min_samples_leaf=min_sample_leaf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if ind>0:
        print "Accuracy " + typeT + " is ", accuracy_score(y_test, y_pred) * 100
        x_arr = (np.array(X_test)).astype('float32')
        d = {'event': list(y_test["event"]), 'node': list(clf.tree_.apply(x_arr))}
        new_df = pd.DataFrame(data=d)
        plots.hist_count_XinY(df=new_df, countfrom="node", count_of="event", log_count=True, xlabel="node",
                          title="results in node - "+typeT+" tree", ind=ind)
    return clf, y_pred

# print(clf_gini.tree_.node_count) #get the node count.


def classifier_fit(classifaier, c_name, X_train,y_train,X_test,y_test, hist):
    print "start - "+c_name
    classifaier.fit(X_train, y_train)
    y_pred = classifaier.predict(X_test)

    if hist>0:
        print "Accuracy " + c_name + " is ", accuracy_score(y_test, y_pred) * 100
        d = {'event': list(y_test["event"]), 'node': list(y_pred)}
        new_df = pd.DataFrame(data=d)
        plots.hist_count_XinY(df=new_df, countfrom="node", count_of="event", log_count=True, xlabel="cluster",
                              title="results in class- "+c_name, ind=hist)
    return y_pred