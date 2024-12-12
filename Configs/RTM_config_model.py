import numpy as np
import segyio
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
               'path' : r"Data/Models/model_vp_52.bin",
               'res_path' : r"Data/R_52.sgy",
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
геометрия моделирования
-----------------------
_nsrc: количество источников
src: расположение источников
_nrec: количество приемников
rec: расположение приемников
"""

#дальше не трогать
if preset == "original":
    nshots = 30
    src = np.empty((nshots, 2))
    src[:, 0] = np.linspace(2000,6000,nshots)
    src[:, 1] = 0.

    nreceivers = 201
    rec = np.empty((nreceivers, 2))
    rec[:,0] = np.linspace(0, 6000, nreceivers)
    rec[:,1] = 0.

elif preset == "real":
    #Подгрузка поля отраженных волн
    src = segyio.open(model_param['res_path'], mode='r', endian='big', ignore_geometry=True)
    samples = src.samples
    data = segyio.tools.collect(src.trace[:])
    sou_x = src.attributes(segyio.TraceField.SourceX)[:]
    sou_z = src.attributes(segyio.TraceField.SourceSurfaceElevation)[:]
    rec_x = src.attributes(segyio.TraceField.GroupX)[:]
    rec_z = src.attributes(segyio.TraceField.ReceiverGroupElevation)[:]
    src.close()

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
    