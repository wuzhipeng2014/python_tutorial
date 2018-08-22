#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_test=pd.read_csv('../../data/test_data.csv')


print('df_test.shape=%s' %str(df_test.shape))
columns_list=df_test.columns.tolist()
print(columns_list)

df_corr=df_test.corr(method='pearson')

print(df_corr)


feature_label_corr_df=pd.DataFrame(df_corr.label,columns=['label'])



feature_label_corr_df['correlation']=feature_label_corr_df.label.apply(lambda x:1 if x>0 else -1)

feature_label_corr_df['label'] =feature_label_corr_df.label.apply(lambda x:x if x>0 else -x)


print(feature_label_corr_df)


use_colours=['g','g','r','r']


feature_label_corr_df.label.plot(kind='barh', color=[np.where(feature_label_corr_df["correlation"]>0, 'g', 'r')])


# feature_label_corr_df.plot.barh(stacked=True);

plt.show()



n=10
df = pd.DataFrame({"a":np.arange(1,n)})
df.plot(kind='bar', color=[np.where(df["a"]>2, 'g', 'r')])
plt.show()