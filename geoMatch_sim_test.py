

import spatial_simulation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import variogram

n = 100

df = pd.DataFrame()

np.random.seed(10)

df['lat'] = np.random.uniform(-1.0,1.0,n)
df['lon'] = np.random.uniform(-1.0,1.0,n)
df['x_random'] = np.random.uniform(0.0, 1.0, n)
df['x_covar_sim'] = np.ones((n,))








