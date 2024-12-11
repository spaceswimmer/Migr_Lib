import numpy as np
import os
import struct
from matplotlib import pyplot as plt

def plot_image(image, xx, zz, cmap='gray', src=None, rec=None, show=False, eik=None, quantile=0.9):
    fig, axs = plt.subplots(1, 1, dpi=200, figsize=(10, 10))
    handle=axs.pcolormesh(xx, zz, image, cmap=cmap, vmin=-np.quantile(image, quantile), vmax=np.quantile(image, quantile))
    axs.set_aspect('equal')
    axs.invert_yaxis()
    axs.set_xlabel('Distance, m')
    axs.set_ylabel('Depth, m')
    cbar = fig.colorbar(handle, fraction=0.025, pad=0.04)
    cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('Reflectivity, c. u.', rotation=270)

    cbar.set_ticks([])
    cbar.set_ticklabels([])
    if src is not None:
        axs.scatter(src[:, 0], src[:, 1], s=30, c='red', marker='v')
    if rec is not None:
        axs.scatter(rec[:, 0], rec[:, 1], s=10, c='blue', marker='^')

    if eik is not None:
        CS = axs.contour(xx, zz, eik, 15, colors='w', )
        axs.clabel(CS, fontsize=10, inline=True)
    if show:
        plt.show()
    else:
        return axs


def readin_bin(path=None, seek_num=None, nt=None, nr=None, dx=None, dz=None):
    FA = open(path, "rb")
    FA.seek(seek_num) # skip N bit to start reading in data
    out_data = np.empty((nt,nr))
    for rr in range(nr):
        for tt in range(nt):
            data = FA.read(4) # read 4 bit float
            data_float = struct.unpack("f", data)[0]
            out_data[tt][rr] = data_float
    x = np.linspace(0, nr*dx-dx, num=out_data.shape[1])
    z = np.linspace(0, nt*dz-dz, num=out_data.shape[0])
    xx, zz = np.meshgrid(x, z)
    return out_data, xx, zz


def writeout_bin(path=None, in_data=None, seek_num=None, nt=None, nr=None):
    FA = open(path, "wb")
    for rr in range(nr):
        for tt in range(nt):
            data = in_data[tt][rr] #FA.read(4)
            data_float = struct.pack("f", data)
            FA.write(data_float)
    FA.close()


def npy2bin(inname=None, ouname=None, seek_num=0, nt=None, nr=None):
  indata = np.load(inname)
  writeout_bin(path=ouname, in_data=indata, seek_num=seek_num, nt=nt, nr=nr)
  return indata

def bin2npy(inname=None, ouname=None, seek_num=0, nt=None, nr=None, dx=10, dz=10):
  indata, _, _ = readin_bin(path=inname, seek_num=seek_num, nt=nt, nr=nr, dx=dx, dz=dz)
  np.save(ouname, indata)
  return indata


def plot_model(model, xx, zz, cmap='turbo', src=None, rec=None, show=False, eik=None):
    fig, axs = plt.subplots(1, 1, dpi=200, figsize=(10, 10))
    handle=axs.pcolormesh(xx, zz, model, cmap='turbo')
    axs.set_aspect('equal')
    axs.invert_yaxis()
    axs.set_xlabel('Distance, m')
    axs.set_ylabel('Depth, m')
    cbar = fig.colorbar(handle, fraction=0.025, pad=0.04)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Velocity, m/s', rotation=270)
    if src is not None:
        axs.scatter(src[:, 0], src[:, 1], s=30, c='red', marker='v')
    if rec is not None:
        axs.scatter(rec[:, 0], rec[:, 1], s=10, c='blue', marker='^')

    if eik is not None:
        CS = axs.contour(xx, zz, eik, 15, colors='w', )
        axs.clabel(CS, fontsize=10, inline=True)
    if show:
        plt.show()
    else:
        return axs
