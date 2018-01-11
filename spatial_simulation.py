

# y = theta*T + sim.Y.cov.effect * X + sim.Y.e
# Yi = sim.Y.scale + (Theta * Ti) + (sim.Y.cov.effect * Xi) + (sim.Y.het.effect * (Ti * Xi)) + sim.Y.e

import simpy
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LogNorm
#from variogram import *
import matplotlib.pyplot as plt



# def spatial_simulation():

n = 200
cov_Decay = "NA"
se_Decay = "NA"
t_Decay = "NA"
sim_T_e = "NA"
T_percent = "NA"
sim_Y_scale = np.random.uniform(-1.0,1.0)
theta = np.random.uniform(0,1)
sim_Y_cov_effect = np.random.uniform(-1.0,1.0)
sim_Y_het_effect = "NA"
sim_Y_e = "NA"
spill_mag = "NA"



# from https://stackoverflow.com/questions/23943301/fast-elegant-way-to-calculate-empirical-sample-covariogram

def get_x(n):


    #x_x = float(xmax - xmin)/n
    #y_y = float(ymax - ymin)/n


    m = n * 90/100



    x = np.linspace(0, 2 * np.pi, n)
    sx = np.sin(3 * x) * np.sin(10 * x)
    density = .8 * abs(np.outer(sx, sx))
    density[:, :n // 2] += .2

    #random.seed(10)
    points = []

    while len(points) < n:

        v, iy, ix = np.random.uniform(0.0,1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0)

        ix_c = int(ix * n)
        iy_r = int(iy * n)

        if v < density[ix_c, iy_r]:

            points.append([ix, iy, density[ix_c, iy_r]])
        #points.append([ix, iy])

    locations = np.array(points)

    return locations


dta = get_x(n)
df = pd.DataFrame(dta, columns=["lon", "lat", "z"])



df['id'] = range(1,n+1,1)
df['sim_Y_scale'] = sim_Y_scale
df['theta'] = theta


df["e_vec"] = list(np.random.uniform(min(df["z"]), max(df["z"]), n))

df["probability"] = (df["z"] + abs(min(df["z"])))/(max(df["z"]) + abs(min(df["z"])))
treatment = df.sample(n=(n/2), weights=df["probability"])

df['treatment'] = df.apply(lambda x: 1 if int(x['id']) in list(treatment['id']) else 0, axis=1)

df['Y'] = theta * df["treatment"] + sim_Y_cov_effect * df["z"]
df['Y'] = df['Y'] + np.random.uniform(min(list(df['Y'])), max(list(df['Y'])))

print np.random.uniform(min(list(df['Y'])), max(list(df['Y'])))

geometry = [Point(coor) for coor in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

outf = r"/Users/miranda/Documents/AidData/github/Simulation_geoMatch/test.csv"
gdf.to_csv(outf, encoding='utf-8', sep=',')

print "done"




"""

z = dta[2,:]
x = dta[0,:]
y = dta[1,:]



#N = int(len(z)**.5)
#z = np.asarray(z)
#z = z.reshape(N,N)


#plt.imshow(z, extent=(-1.0, 1.0, -1.0, 1.0))

plt.plot(x, y,'ro')

#plt.colorbar()
plt.show()

"""


"""
t = df[df['treatment']==1]
c = df[df['treatment']==0]

plt.plot(t['longitude'], t['latitude'], 'ro')
plt.plot(c['longitude'], c['latitude'], 'bo')

plt.show()

"""






