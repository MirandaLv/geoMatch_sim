
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pysal

inf = r"/Users/miranda/Documents/AidData/github/Simulation_geoMatch/test.csv"

dta = pd.read_csv(inf)

inmatrix = dta[['lon','lat']]
z = np.array(dta['z'])
dist_inv = 1/squareform(pdist(inmatrix, 'euclidean'))
np.fill_diagonal(dist_inv,0)
w = dist_inv.mean(axis=1)

mi = pysal.Moran(z,w)

print dist_inv








def get_w(coormatrix):

    dist_inv = 1 / squareform(pdist(coormatrix, 'euclidean'))
    np.fill_diagonal(dist_inv, 0)

    return dist_inv

