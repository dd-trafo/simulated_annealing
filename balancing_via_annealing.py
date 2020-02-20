#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_dist(df):
    df.groupby('A').count().plot.bar(color='#ef9a9a', rot=0)
    df.groupby('B').count().plot.bar(color='#bbdefb', rot=0)

def error(df):
    n = len(df)
    groupA = df.groupby('A').count()
    groupB = df.groupby('B').count()
    
    return (groupA.max()[0] - groupA.min()[0]) + (groupB.max()[0] - groupB.min()[0]) + 1 * np.abs(50 - n)

def generate_neighbor(df, df_universe):
    if np.random.uniform() < 0.5:
        # add 1 element
        if len(df) < len(df_universe):
            available_indices = list(set(df_universe.index).difference(set(df.index)))
            i = available_indices[np.random.randint(low=0, high=len(available_indices))]
            df = df.append(df_universe.loc[i], ignore_index=False)
    else:
        # remove element
        if len(df) > 0:
            available_indices = list(df.index)
            i = available_indices[np.random.randint(low=0, high=len(available_indices))]
            df = df.drop(i)
        
    return df


np.random.seed(555)

### generate data
n = 10000

data = dict()
level_A = ['low', 'med', 'high']
data['A'] = [level_A[i] for i in np.random.randint(low=0, high=3, size=n)]

level_B = ['a', 'b', 'c']
data['B'] = [level_B[i] for i in np.random.randint(low=0, high=3, size=n)]

df_universe = pd.DataFrame(data=data)

df = df_universe.sample(n=2)

### get initial temperature T0
error_df = error(df)
f = 0
n0 = 100
x0 = 0.8
for i in tqdm(range(n0)):
    f = f + error(generate_neighbor(df, df_universe))

avg_f = f / n0 - error_df

T0 = - avg_f / np.log(x0)

### perform annealing
steps = 1000

error_df = error(df)
alpha = 0.97
T = alpha * T0

for k in tqdm(range(steps)):
    
    T = alpha * T
    df_proposal = generate_neighbor(df, df_universe)
    error_df_proposal = error(df_proposal)
    
    if np.exp( -(error_df_proposal - error_df) / T ) > np.random.uniform():
        
        df = df_proposal
        error_df = error_df_proposal
        print(f'error: {error_df:3f}')


plot_dist(df)
print(f'len(df) = {len(df)}')
print(f'error(df) = {error(df):.5f}')