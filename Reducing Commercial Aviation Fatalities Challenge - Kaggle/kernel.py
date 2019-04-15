#!/usr/bin/env python

import os

import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
import matplotlib.pyplot as plt

import defines
import dm_tools
import man_data

#-------read data--------
import plots

data_f = os.path.join(defines.PATH_TO_FILES, 'train.csv')
data_frame = man_data.downcast(man_data.make_dataFrame(data_f))
data_frame['event'] = data_frame['event'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
data_frame['exp'] = data_frame['exp'].map({'CA': 3, 'DA': 4, 'SS': 2})


# if defines.TEST==False: data_frame=data_frame.sample(n=5000)

X=data_frame.drop(["event"], axis=1)
y=data_frame.loc[:,["event"]]
#learn
if defines.TEST==False:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train=X_train.drop(["time","seat",'exp'],axis=1)
    X_test=X_test.drop(["time","seat",'exp'],axis=1)
else:
    test_f = os.path.join(defines.PATH_TO_FILES, 'test.csv')
    test_frame = man_data.downcast(man_data.make_dataFrame(test_f))
    test_frame['exp'] = test_frame['exp'].map({'CA': 3, 'DA': 4, 'SS': 2})

    X_train=X.drop(['exp'],axis=1)
    y_train=y
    X_test=test_frame.drop(['exp','id'],axis=1)
    y_test=[]
#test

x_arr = (np.array(X_test)).astype('float32')

#--------check distributions between the files train and test:--------
# parameters= list(data_frame.drop(['event','exp','seat'],axis=1))
# dm_tools.distributions_test2train(data_frame=data_frame, test_frame=test_frame,parameters=parameters)



#--------checkbulding trees, not a part of the data--------check

# clf_gini, pred_gini=dm_tools.decision_tree_pred(typeT="gini", rand_state=100, max_depth=3, min_sample_leaf=5,X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,ind=0)
# clf_entropy, pred_entropy=dm_tools.decision_tree_pred(typeT="entropy", rand_state=100, max_depth=3, min_sample_leaf=5,X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,ind=0)

# pred = clf_entropy.predict_proba(X_test)
# print pred

#-------accuracy - end of project
print "KNN"
clf_KN = KNeighborsClassifier(n_neighbors=1)
dm_tools.classifier_fit(clf_KN,"KNN",X_train,y_train,X_test,y_test,7)

print "random forest - boost and not boost"
clf_RF = RandomForestClassifier(random_state=1)
boost_RF=AdaBoostClassifier(base_estimator=clf_RF, n_estimators=10)
dm_tools.classifier_fit(clf_RF,"random_forest",X_train,y_train,X_test,y_test,6)
dm_tools.classifier_fit(boost_RF,"boost_random_forest",X_train,y_train,X_test,y_test,5)

print "entropy - boost and not boost"
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
boost_entropy=AdaBoostClassifier(base_estimator=clf_entropy, n_estimators=10)
dm_tools.classifier_fit(boost_entropy,"boost_entropy",X_train,y_train,X_test,y_test,4)
dm_tools.classifier_fit(clf_entropy,"entropy",X_train,y_train,X_test,y_test,3)

print "gini - boost and not boost"
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
boost_gini=AdaBoostClassifier(base_estimator=clf_gini, n_estimators=10)
dm_tools.classifier_fit(boost_gini,"boost_gini",X_train,y_train,X_test,y_test,2)
dm_tools.classifier_fit(clf_gini,"gini",X_train,y_train,X_test,y_test,1)
plt.show()
#-------classifaiers---------
#
# #qua
# print "start"
# qda = QuadraticDiscriminantAnalysis()
# qda=AdaBoostClassifier(base_estimator=qda, n_estimators=10)
# print "fit..."
# qda.fit(X_train, y_train)
# print "predict...."
# qda_prob = qda.predict_proba(X_test)
# print "write..."
# man_data.write_df(defines.TEST, qda_prob, test_frame['id'], 'QUA_boost')
#
# # ## SVM
# clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)
# print "fit..."
# clf.fit(X_train, y_train)
# print "predict...."
# pred= clf.predict_proba(X_test)
# print "write..."
# man_data.write_df(defines.TEST, pred, test_frame['id'], 'SVM')
# #
# #LDA
# print "start"
# lda = LinearDiscriminantAnalysis()
# print "fit...."
# lda.fit(X_train, y_train)
# print "predict"
# lda_prob = lda.predict_proba(X_test)
# # lda_loss = log_loss(y_test, lda_prob, labels=lda.classes_)
# # print ("loss: %.2f [%s]" % (lda_loss, "LDA"))
# man_data.write_df(defines.TEST, lda_prob, test_frame['id'], 'LDA')
#
# clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
print "gini"
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
# print "boost gini"
# clf_gini_boost=AdaBoostClassifier(base_estimator=clf_gini,n_estimators=5)
boost_gini=AdaBoostClassifier(base_estimator=clf_gini, n_estimators=10)
dm_tools.classifier_fit(boost_gini,"boost_gini",X_train,y_train,X_test,y_test,1)
plt.show()
# boost_entropy=AdaBoostClassifier(base_estimator=clf_entropy, n_estimators=10)
# # clf_dt = DecisionTreeClassifier()

clf_KN = KNeighborsClassifier(n_neighbors=1)
print "random forest"
clf_RF = RandomForestClassifier(random_state=1)
clf_gaus=GaussianNB()
# bagging_KN=BaggingClassifier(base_estimator=clf_KN, n_estimators=10, max_samples=0.8, max_features=0.8)
# print "boost rf"
boost_RF=AdaBoostClassifier(base_estimator=clf_RF, n_estimators=10)
# lr = LogisticRegression()
# sclf = StackingClassifier(classifiers=[clf_KN, clf_RF, clf_gini,clf_gaus],meta_classifier=lr)

dm_tools.classifier_fit(boost_RF, 'boost_RF', X_train,y_train,X_test,y_test, 2)

# y_pred_boostree_RF, new_df=dm_tools.classifier_fit(LogisticRegression,"boost on gini",X_train,y_train,X_test,y_test, hist=0)
# y_pred_dt=dm_tools.classifier_fit(clf_dt,"decision tree",X_train,y_train,X_test,y_test, hist=0)
# y_pred_KN=dm_tools.classifier_fit(clf_KN,"K-Neighbors-Classifier",X_train,y_train,X_test,y_test,hist=0)
# y_pred_RF, new_df=dm_tools.classifier_fit(clf_RF,"random forest",X_train,y_train,X_test,y_test, hist=0)
# y_pred_s, new_df=dm_tools.classifier_fit(sclf,"stacking classifier",X_train,y_train,X_test,y_test, hist=0)
# y_pred_RF_B, new_df=dm_tools.classifier_fit(bagging_RF,"random forest boosting classifier",X_train,y_train,X_test,y_test,hist=8)


# pred=sclf.predict_proba(X_test)
# label_vec = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
# clf_list = [clf_KN, clf_RF, clf_gaus, sclf]
# label_vec = ['sclf']
# clf_list = [sclf]

# fig = plt.figure(figsize=(20, 13))
# gs = gridspec.GridSpec(2, 2)
# grid = itertools.product([0, 1], repeat=2)

# clf_cv_mean = []
# clf_cv_std = []
# xx=np.array(X_train)
# yy=np.array(y_train).flatten()
# # for clf, label, grd in zip(clf_list, label_vec, grid):
# for clf, label in zip(clf_list, label_vec):
#     scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
#     print ("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
#     clf_cv_mean.append(scores.mean())
#     clf_cv_std.append(scores.std())
#     clf.fit(X_train, y_train)
#     print "done: fit "+label
#     pred =clf.predict_proba(X_test)
#     # pred=pred.apply(pd.to_numeric, downcast='float')
#     print "done: predeict "+ label
#     man_data.write_df(defines.TEST, pred, test_frame['id'], label)
#     # ax = plt.subplot(gs[grd[0], grd[1]])
#     # fig = plot_decision_regions(X=xx, y=yy, clf=clf)
#     # plt.title(label)


# pred_s=sclf.predict_proba(X_test)
# pred_fr=clf_RF.predict_proba(X_test)
# pred_nk=clf_KN.predict_proba(X_test)
# pred_nb=clf_gaus.predict_proba(X_test)
# plt.show()

#---------fix by classification
# X_train['label']=list(clf_dt.predict(X_train))
# X_train1=X_train[X_train['label']==1]
# y_train1=y_train[X_train['label']==1]
#
# X_test['label']=y_pred_dt
# X_test1=X_test[X_test['label']==1]
# y_test1=y_test[X_test['label']==1]
#
# X_train4=X_train[X_train['label']==4]
# y_train4=y_train[X_train['label']==4]
#
# X_test4=X_test[X_test['label']==4]
# y_test4=y_test[X_test['label']==4]


# clf_entropy, pred_entropy=dm_tools.decision_tree_pred(typeT="entropy", rand_state=100, max_depth=3, min_sample_leaf=5,X_train=X_train1, y_train=y_train1,X_test=X_test1,y_test=y_test1,ind=0)
# pred1 = clf_entropy.predict_proba(X_test1)
#
# clf_entropy, pred_entropy=dm_tools.decision_tree_pred(typeT="entropy", rand_state=100, max_depth=3, min_sample_leaf=5,X_train=X_train4, y_train=y_train4,X_test=X_test4,y_test=y_test4,ind=0)
# plt.show()
#
# pred4 = clf_entropy.predict_proba(X_test4)
# print pred4
# pred = []



#### ---- write data------
# if defines.TEST:
#     man_data.write_df(defines.TEST, pred_s, test_frame, "sclf")
#     man_data.write_df(defines.TEST, pred_nk, test_frame, "nk")
#     man_data.write_df(defines.TEST, pred_fr, test_frame, "rf")
#     man_data.write_df(defines.TEST, pred_nb, test_frame, "gaussian_nb")
# else:
#     man_data.write_df(defines.TEST,pred_s,[],"sclf")
#     man_data.write_df(defines.TEST,pred_nk,[],"nk")
#     man_data.write_df(defines.TEST,pred_fr,[],"rf")
#     man_data.write_df(defines.TEST,pred_nb,[],"nb")


# #--------examples to plots file:--------
# plots.hist_count(df=data_frame,col_name="event",xlabel="event",title="Count events",ind=1)
# plots.hist_count_XinY(df=data_frame, countfrom="exp",count_of="event",log_count=True,xlabel="expiriment",title="expiriment and results", ind=2)
# plots.hist_count_XinY(df=data_frame, countfrom="event",count_of="seat",log_count=True,xlabel="event",title="event and seat", ind=3)
# plots.distribution([data_frame["time"], test_frame["time"]], label_vec=["train-time", "test-time"], xlabel="x", title="time distribution",ind=4)
# plots.scatter_plot(data_frame,x_name="time",y_name="event",color="b",marker="o",title="tit",xlabel="xlabel", ylabel="ylabel", ind=5)

