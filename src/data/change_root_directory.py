import pandas as pd

cwt_train_df = pd.read_csv('extra files/cwt.csv')
cwt_test_df = pd.read_csv('extra files/cwt.csv')
wsst_train_df = pd.read_csv('../extra files/wsst.csv')
wsst_test_df = pd.read_csv('../extra files/wsst.csv')

wsst_train_df['file_path'] = wsst_train_df['file_path'].apply(lambda x: '/usr/local' + x[1:])
wsst_test_df['file_path'] = wsst_test_df['file_path'].apply(lambda x: '/usr/local' + x[1:])
cwt_train_df['file_path'] = cwt_train_df['file_path'].apply(lambda x: '/usr/local' + x[1:])
cwt_test_df['file_path'] = cwt_test_df['file_path'].apply(lambda x: '/usr/local' + x[1:])