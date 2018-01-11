
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares

# from: http://connor-johnson.com/2014/03/20/simple-kriging-in-python/

def SVh(P, h, bw):
    """
    calculate semivariogram for a single lag
    :param P: dataset that has x, y
    :param h: span (a lag)
    :param bw: bandwidth
    :return: semivariogram for a single lag
    """

    pd = squareform(pdist(P[:,:2]))

    N = pd.shape[0]

    Z = list()

    for i in range(N):

        for j in range(i+1, N):

            if (pd[i,j] >= h-bw) and (pd[i,j] <= h+bw):

                Z.append(pd[i,2] - pd[j,2])


    return np.sum(Z)/(2*len(Z))


def SV(P, hs, bw):

    """
    Experimental variogram for a collection of lags
    """

    sv = list()

    for h in hs:
        sv.append(SVh(P, h, bw))

    sv = [[hs[i], sv[i]] for i in range(len(hs)) if sv[i] > 0]

    return np.array(sv).T



def spherical(h,a,c):

    """
    :param h: lag distance
    :param a: practical range
    :param c: sill
    :return:
    """
    if h <= a:
        return c * (1.5 * (h/a) - 0.5*((h/a)**3.0))
    else:
        return c


def no_name(x, dataf, a, c):

    dataf['z'] = x * dataf['x_temp']
    dataf = dataf[['lon', 'lat', 'z']]

    P = np.array(df)

    distance_list = sorted(set(pdist(P[:, :2])))

    bw = np.max(distance_list)/100

    hs = np.arange(0.0, distance_list[-1], bw)

    x_pred, y_pred = SV(P, hs, bw).tolist()

    y_true = [spherical(h,a,c) for h in x_pred]


    residual = np.array(y_true) - np.array(y_pred)

    return residual



n = 200

lat = np.random.uniform(-1.0, 1.0, n)
lon = np.random.uniform(-1.0, 1.0, n)

df = pd.DataFrame()
df['lon'] = lon
df['lat'] = lat
df['x_temp'] = np.random.uniform(0,10.0, n)


a = 0.05
c = 1
iters = 10



coef = np.random.uniform(0,1.0, n)

rest = least_squares(no_name, x0=coef, args=(df, a,c), verbose=2, gtol=1e-12)

df["z"] = rest.x * df["x_temp"]

z = np.array(df['z'])




"""

df['id'] = range(1, n+1, 1)
df['temp_x'] = [1] * n
df['beta'] = np.random.uniform(0,1.0, 100)
df['z'] = df['temp_x'] * df['beta']

test_df = df[['lat', 'lon', 'z']]
"""


bw = 0.01


P = np.array(df[['lon','lat', 'z']])
pd = squareform(pdist(P[:,:2]))

min_dist = np.min(pd[np.nonzero(pd)])
max_dist = np.max(pd[np.nonzero(pd)])

#a = np.random.uniform(min_dist, max_dist)
#c = (np.max(df['z']) - np.min(df['z']))/2

hs = np.arange(0, np.max(pd), bw)

val = SV(P, hs, bw)


x1 = val[0,:]
y1 = val[1,:]

y = list()
for h in x1:
    y.append(spherical(h,a,c))




#plt.plot(x1, y)
#plt.plot(x1, y1)
#plt.show()


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



df['id'] = range(1,n+1,1)
df['theta'] = theta


df["e_vec"] = list(np.random.uniform(min(df["z"]), max(df["z"]), n))

# df["probability"] = (df["z"] + abs(min(df["z"])))/(max(df["z"]) + abs(min(df["z"])))

treatment = df.sample(n=(n/2)) #, weights=df["probability"]
df['treatment'] = df.apply(lambda x: 1 if int(x['id']) in list(treatment['id']) else 0, axis=1)

# df['treatment'] = df.apply(lambda x: 1 if int(x['id']) in list(treatment['id']) else 0, axis=1)

df['Y'] = theta * df["treatment"] + sim_Y_cov_effect * df["z"]
df['Y'] = df['Y'] + np.random.uniform(min(list(df['Y'])), max(list(df['Y'])))


outf = r"/Users/miranda/Documents/AidData/github/Simulation_geoMatch/test.csv"
df.to_csv(outf, encoding='utf-8', sep=',')


# random assign T
# calculate Moran I of t vector and x, yt, yc, y, distance
# estimate spill.par





#N = int(len(df['z'])**.5)
#z = np.asarray(list(df['z']))
#z = z.reshape(N,N)


#plt.imshow(z, extent=(-1.0, 1.0, -1.0, 1.0))

#plt.plot(list(df['lon']), list(df['lat']),'ro')
#plt.colorbar()
#plt.show()
