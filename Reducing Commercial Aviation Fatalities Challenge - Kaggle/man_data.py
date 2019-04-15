import defines
import pandas as pd     #read csv files
import numpy as np
import gc


def make_dataFrame(file):
    data_frame = pd.read_csv(file)
    data_frame = data_frame.rename(columns={
                            'crew' : 'crew',
                            'experiment' : 'exp',
                            'time' : 'time',
                            'seat' : 'seat',
                            'eeg_fp1' : 'fp1',
                            'eeg_f7' : 'f7',
                            'eeg_f8' : 'f8',
                            'eeg_t4' : 't4',
                            'eeg_t6' : 't6',
                            'eeg_t5' : 't5',
                            'eeg_t3' : 't3',
                            'eeg_fp2' : 'fp2',
                            'eeg_o1' : 'o1',
                            'eeg_p3' : 'p3',
                            'eeg_pz' : 'pz',
                            'eeg_f3' : 'f3',
                            'eeg_fz' : 'fz',
                            'eeg_f4' : 'f4',
                            'eeg_c4' : 'c4',
                            'eeg_p4' : 'p4',
                            'eeg_poz' : 'poz',
                            'eeg_c3' : 'c3',
                            'eeg_cz' : 'cz',
                            'eeg_o2' : 'o2',
                            'ecg' : 'ecg',
                            'r' : 'r',
                            'gsr' : 'gsr',
                            'event' : 'event'})
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

def write_df(test,pred,test_frame_id,name):
    if test:
        sub = pd.DataFrame(pred, columns=['A', 'B', 'C', 'D'])
        sub['id'] = test_frame_id
        cols = sub.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        sub = sub[cols]
        print "writing..."
        sub.to_csv(defines.PATH_TO_FILES + name+".csv", index=False)
    else:
        sub = pd.DataFrame(pred, columns=['A', 'B', 'C', 'D'])
        sub.to_csv(defines.PATH_TO_FILES + name+".csv", index=False)
