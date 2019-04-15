#this data_frame includes plot functions

import numpy as np
import itertools
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

plt.style.use("fivethirtyeight")


#scatter plot - for corelation detection - only numerical values
def scatter_plot(data_frame,x_name,y_name,color,marker,title,xlabel, ylabel, ind):
    s = 60          #size of the frame of the marker
    lw = 0          #size of the marker
    alpha = 0.07    #transparency over the edges
    axis_width = 1.5
    tick_len = 6
    fontsize = 16
    plt.figure(ind)  # Here's the part I need

    ax = plt.scatter(data_frame[x_name].values, data_frame[y_name].values,
                     marker=marker, color=color, s=s, lw=lw,alpha=alpha)
    xrange = abs(data_frame[x_name].max() - data_frame[x_name].min())
    yrange = abs(data_frame[y_name].max() - data_frame[y_name].min())
    cushion = 0.1
    xmin = data_frame[x_name].min() - cushion * xrange
    xmax = data_frame[x_name].max() + cushion * xrange
    ymin = data_frame[y_name].min() - cushion * yrange
    ymax = data_frame[y_name].max() + cushion * yrange
    ax = plt.xlim([xmin, xmax])
    ax = plt.ylim([ymin, ymax])
    ax = plt.xlabel(xlabel, fontsize=fontsize)
    ax = plt.ylabel(ylabel, fontsize=fontsize)
    ax = plt.title(title, fontsize=fontsize+7)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tick_params('both', length=tick_len, width=axis_width,
                         which='major', right=True, top=True)
    return ax


def grid_plot(spec_row, spec_col,X,y,title_list, clf_list,ind):
    plt.figure(ind)
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(spec_row, spec_col)
    grid = itertools.product([0, 1], repeat=2)

    for clf, label, grd in zip(clf_list, title_list, grid):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(label)

#count the events in the column name
def hist_count(df,col_name,xlabel,title,ind):
    plt.figure(ind)
    plt.figure(figsize=(15, 10))
    sns.countplot(df[col_name])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=15)

#count the event in case of column x
def hist_count_XinY(df, countfrom ,count_of,log_count,xlabel,title, ind):
    plt.figure(ind)
    plt.figure(figsize=(15,10))
    sns.countplot(countfrom, hue=count_of, data=df)
    plt.xlabel(xlabel, fontsize=12)
    if log_count:
        plt.ylabel("Count (log)", fontsize=12)
        plt.yscale('log')
    else: plt.ylabel("Count", fontsize=12)

    plt.title(title, fontsize=15)

#one graph of distributions of all colomus of data in col_data_vec
def distribution(col_data_vec, label_vec, xlabel, title,ind):
    plt.figure(ind)
    plt.figure(figsize=(15, 10))
    for col_data, label in zip(col_data_vec, label_vec):
        sns.distplot(col_data, label=label)
    plt.legend()
    plt.xlabel(xlabel, fontsize=12)
    plt.title(title, fontsize=15)
