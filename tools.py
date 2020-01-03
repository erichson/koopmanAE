import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper')
import matplotlib.patches as mpatches

import seaborn as sns
cmap = sns.cubehelix_palette(light=1, dark=0, start=3,  as_cmap=True)


import numpy as np


import torch.nn.init as init

def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1],1)

    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    else:
        return "dimenional error"




def weights_init(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)




def plot_flow_prediction(input, target, prediction, m, n, dataset):
    import cmocean

    x2 = np.arange(0, m, 1)
    y2 = np.arange(0, n, 1)
    mX, mY = np.meshgrid(x2, y2)


    fig, ax = plt.subplots(3, 4, facecolor="white",  edgecolor='k', figsize=(25,13))
    ax = ax.ravel()

    j = 0

    for k in range(3):
        for i in range(4):

            if k == 0:
                frame = input[i]
            elif k == 1:
                frame = target[i]
            elif k == 2:
                 frame = prediction[i]


            minmax = np.max(np.abs(frame))

            ax[j].contourf(mX, mY, frame.T, 50, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
            #ax[j].imshow(frame.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)

            if dataset == 'flow':
                wedge = mpatches.Wedge((-3,33), 9, 270, 90, ec="#636363", color='#636363',lw = 5,zorder=200)
                ax[j].add_patch(p=wedge)

            #plt.title('Epoch number ' + str(epoch), fontsize = 16 )
            ax[j].tick_params(axis='y', labelsize=0)
            ax[j].tick_params(axis='x', labelsize=0)
            ax[j].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            j += 1

    plt.tight_layout()
    plt.show()





def plot_sst_prediction(input, target, prediction, m, n, dataset):
    import cmocean

    import os
    import conda

    conda_file_dir = conda.__file__
    conda_dir = conda_file_dir.split('lib')[0]
    proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
    os.environ["PROJ_LIB"] = proj_lib

    from mpl_toolkits.basemap import Basemap

    lats = np.load('data/lats.npy')
    lons = np.load('data/lons.npy')



    fig, ax = plt.subplots(3, 4, facecolor="white",  edgecolor='k', figsize=(25,8))
    ax = ax.ravel()

    mintemp = np.min(input[0])
    maxtemp = np.max(input[0])
    minmax = np.maximum(mintemp,maxtemp)
    j = 0
    for k in range(3):
        for i in range(4):

            if k == 0:
                frame = input[i]
            elif k == 1:
                frame = target[i]
            elif k == 2:
                 frame = prediction[i]




            m = Basemap(projection='mill',
                        lon_0 = 180,
                        llcrnrlat = 10,
                        llcrnrlon = 256,
                        urcrnrlat = 31,
                        urcrnrlon = 294.0,
                        #resolution='l',
                        ax=ax[j])

            #m = Basemap(projection='mill', lon_0 = 180)
            m.pcolormesh(lons, lats, frame, cmap=cmocean.cm.balance, latlon=True, alpha=1.0, shading='gouraud',
                         vmin = -minmax, vmax=minmax)
            m.fillcontinents(color='lightgray', lake_color='aqua')
            m.drawmapboundary(fill_color='lightgray')
            m.drawcoastlines(3)

            #ax[j].imshow(frame, cmap=cc.cm.rainbow, interpolation='none', vmin=-minmax, vmax=minmax)

            #plt.title('Epoch number ' + str(epoch), fontsize = 16 )
            #ax[j].tick_params(axis='y', labelsize=0)
            #ax[j].tick_params(axis='x', labelsize=0)
            #ax[j].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            j += 1

    plt.tight_layout()
    plt.show()





def plot_sst(sstframe):
    from mpl_toolkits.basemap import Basemap, cm, interp
    import colorcet as cc

    from netCDF4 import Dataset, date2index, num2date
    #from mpl_toolkits.basemap import Basemap, cm, interp
    #from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
    #from matplotlib import ticker

    import os
    import conda

    conda_file_dir = conda.__file__
    conda_dir = conda_file_dir.split('lib')[0]
    proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
    os.environ["PROJ_LIB"] = proj_lib

    from mpl_toolkits.basemap import Basemap



    data_land_mask = 'data/lsmask.nc'
    data = 'data/sst.wkmean.1990-present.nc'

    nc = Dataset(data, mode='r')
    ncLAND = Dataset(data_land_mask, mode='r')

    # read sst.  Will automatically create a masked array using
    # missing_value variable attribute. 'squeeze out' singleton dimensions.
    sst_land = ncLAND.variables['mask'][:].squeeze()

    # read lats and lons (representing centers of grid boxes).
    lats_global = nc.variables['lat'][56:120]
    lons_global = nc.variables['lon'][230:294]


    nc.close()

    lons_global, lats_global = np.meshgrid(lons_global,lats_global)
    sst_land[56:120, 230:294]

    fig = plt.figure(figsize=(10, 6), facecolor="white",  edgecolor='k', dpi=200)
    # create Basemap instance.
    basemap_img = Basemap(projection='mill', lon_0 = 180)

    #im1 = basemap_img.pcolormesh(lons_global, lats_global, sstframe, shading='flat', cmap=cc.cm.gray, latlon=True)
    im1 = basemap_img.pcolormesh(lons_global, lats_global, sstframe, shading='flat',
                                 cmap=cc.cm.rainbow, latlon=True, alpha=1.0)
    basemap_img.drawcoastlines()
    basemap_img.fillcontinents()
    #cb = basemap_img.colorbar(im1,"bottom", size="5%", pad="2%")


    # add a title.
    plt.tight_layout()
    plt.axis('off')
    plt.grid(False)
    plt.show()


def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mu,
                  width=w, height=h,
                  angle=theta, facecolor=facecolor, **kwargs)
    ax.add_artist(ell)

    return ax.add_patch(ell)
