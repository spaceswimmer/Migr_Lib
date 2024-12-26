import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import segyio
import glob
from util import *
import matplotlib.pyplot as plt
"""
original: Оригинальный скрипт - нету residual, нужно считать true_d, smooth_d, u0
real: Случай для реальных данных - есть residual (поле отраженных волн), нужно считать только u0
model: Случай для модельных данных - нету residual, но есть true_d, нужно посчитать smooth_d, u0, после чего можно вычислить residual
"""
preset = "model"

#для чтения файла модели
NT = 467 #int(sys.argv[1])
NR = 801 #int(sys.argv[2])

"""
'origin' : начальная точка модели в метрах
'spacing' : размер ячеек модели в метрах
'path' : путь к файлу со скоростями продольных волн
'res_path' : путь к файлу с полем отраженных волн
'nbl' : количество дампинг слоев
'bcs' : тип дампинг слоя (обычно damp)
'so' : space order 
"""
model_param = {'origin' : (0,0),
               'spacing' : (10., 10.),
               'path' : '/home/spaceswimmer/Documents/NeoGen_Modeling/Data/2D_Scenarios/sc_1/Vp 2D interpolated.npy',
               'res_path' : ['/home/spaceswimmer/Documents/NeoGen_Modeling/Results/2D_vankor/Regular',
                             '/home/spaceswimmer/Documents/NeoGen_Modeling/Results/2D_vankor/Smooth'],
               'nbl' : 20,
               'bcs' : "damp",
               'so' : 8,
               'filter': (9,9)}

"""
t0: время начала моделирования
tn: время окончания моделирования
f0: частота вейвлета
"""
t0=0.
tn=3001.
f0=.015

"""
геометрия моделирования обычно из сегваев
-----------------------
_nsrc: количество источников
src: расположение источников
_nrec: количество приемников
rec: расположение приемников
"""

if preset == "original":
    #Скорости
    model_vp, xx, zz = readin_bin(model_param['path'], seek_num=0, nt=NT, nr=NR, dx=10, dz=10)
    model_vp = model_vp/1000
    
    #Геометрия
    nshots = 30
    src = np.empty((nshots, 2))
    src[:, 0] = np.linspace(2000,6000,nshots)
    src[:, 1] = 0.

    nreceivers = 201
    rec = np.empty((nreceivers, 2))
    rec[:,0] = np.linspace(0, 6000, nreceivers)
    rec[:,1] = 0.

elif preset == "real":
    #Скорости
    model_vp, _, _ = readin_bin(model_param['path'], seek_num=0, nt=NT, nr=NR, dx=10, dz=10)
    model_vp = model_vp/1000

    #Подгрузка поля отраженных волн
    src = segyio.open(model_param['res_path'], mode='r', endian='big', ignore_geometry=True)
    samples = src.samples
    data = segyio.tools.collect(src.trace[:])
    sou_x = src.attributes(segyio.TraceField.SourceX)[:]
    sou_z = src.attributes(segyio.TraceField.SourceSurfaceElevation)[:]
    rec_x = src.attributes(segyio.TraceField.GroupX)[:]
    rec_z = src.attributes(segyio.TraceField.ReceiverGroupElevation)[:]
    src.close()

    #Геометрия
    nreceivers = np.unique(rec_z).size
    nshots = np.unique(sou_x).size
    idx = sou_x == np.unique(sou_x)[10]
    csg_rec_z = rec_z[idx]

    src = np.empty((nshots, 2), dtype=np.float32)
    src[:, 0] = np.unique(sou_x)
    src[:, 1] = 0.

    rec = np.empty((nreceivers, 2))
    rec[:, 0] = 4000.
    rec[:, 1] = csg_rec_z
elif preset == "model":
    #Скорости
    model_vp = np.load(model_param['path'])

    src = segyio.open(model_param['res_path'][0]+'/2d_vankor_SRC-0.sgy')
    sou_x = src.attributes(segyio.TraceField.SourceX)[:]/src.attributes(segyio.TraceField.SourceGroupScalar)
    rec_x = src.attributes(segyio.TraceField.GroupX)[:]
    src.close()

    #Геометрия
    nrec = np.unique(rec_x).size
    rec = np.empty((nrec, 2))
    rec[:, 0] = rec_x
    rec[:, 1] = 0

    nshots = len(glob.glob(model_param['res_path'][0]+'/*.sgy'))

    # print(model_vp)