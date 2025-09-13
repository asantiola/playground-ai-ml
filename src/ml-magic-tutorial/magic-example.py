# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# %%
cols = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
df = pd.read_csv('../../data/magic/magic04.data', names=cols)

# %%
df.head()

# %% - classes: gammas and hadrons
df['class'].unique()

# %% - convert these classes to numbers
df['class'] = (df['class'] == 'g').astype(int)

# %% - graph the data per label...
for label in cols[:-1]:
    plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df['class']==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.ylabel('probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()

# %% train, validation and test datasets
# 1st split is at 60%
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# %%
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values   # 2d
    y = dataframe[dataframe.columns[-1]].values    # 1d, vector
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y

# %% scale the data so theyre relative to the mean
print("gammas  :", len(train[train['class'] == 1])) # gammas
print("hadrons :", len(train[train['class'] == 0])) # hadrons

# %% scale the data!
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# %% 
#- paused at 50:00
