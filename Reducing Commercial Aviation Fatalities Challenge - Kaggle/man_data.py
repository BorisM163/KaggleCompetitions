#!/usr/bin/env python
import pandas as pd     #read csv files


def make_dataFrame(file):
    data_frame = pd.read_csv(file)
    return data_frame

def downcast(df):
    test_int = df.select_dtypes(include=['int'])
    t_converted_int = test_int.apply(pd.to_numeric, downcast='unsigned')
    test_float = df.select_dtypes(include=['float'])
    t_converted_float = test_float.apply(pd.to_numeric, downcast='float')
    t_converted_obj = pd.DataFrame()
    df[t_converted_int.columns] = t_converted_int
    df[t_converted_float.columns] = t_converted_float
    df[t_converted_obj.columns] = t_converted_obj
    df_new=df.copy()
    del [[t_converted_int, t_converted_float, t_converted_obj, test_int, test_float,df ]]
    return df_new
