import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

"""
This is a python script for executing convinient RTM migration
Please use "RTM_config.py" to configure parameters for it. 
If you create an additional config file, import it with the line below
"""
from Configs.RTM_config_model import *

import numpy as np
import copy
import gc
import glob
from tqdm import tqdm
from time import time
from devito import gaussian_smooth, TimeFunction, Eq, Operator, solve, Function
from util import *
from examples.seismic import SeismicModel, AcquisitionGeometry, PointSource, Receiver, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver

from devito import configuration
configuration['log-level'] = 'WARNING'
#Using GPU
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'pgcc'
configuration['language'] = 'openacc'

def ImagingOperator(model, image):
    # Define the wavefield with the size of the model and the time dimension
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                     save=geometry.nt)
    
    # Define the wave equation, but with a negated damping term
    eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(v.backward, solve(eqn, v.backward))
    
    # Define residual injection at the location of the forward receivers
    dt = model.critical_dt
    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)    
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)

    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image - u * v)

    return Operator([stencil] + res_term + [image_update],
                    subs=model.spacing_map)

if __name__ == "__main__":
    begin = time()
    
    model = SeismicModel(vp = model_vp.T,
                         origin = model_param['origin'], 
                         shape = model_vp.T.shape, 
                         spacing = model_param['spacing'],
                         space_order=model_param['so'],
                         nbl=model_param['nbl'],
                         bcs=model_param['bcs'])
    
    model0 = copy.deepcopy(model)
    gaussian_smooth(model0.vp, sigma=model_param['filter'])

    src_coordinates = np.empty((1, 2))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = 20.  # Depth is 20m
    geometry = AcquisitionGeometry(model, 
                                   rec, 
                                   src_coordinates, 
                                   t0, 
                                   tn, 
                                   f0=f0, 
                                   src_type='Ricker')
    
    image = Function(name='image', grid=model.grid)
    op_imaging = ImagingOperator(model, image)
    end = time()
    print('Model creation finished in:', end-begin ,'sec')
    
    begin=time()
    match preset:
        case "original":
            solver = AcousticWaveSolver(model, geometry, space_order=4)
            true_d , _, _ = solver.forward(vp=model.vp)
            smooth_d, _, _ = solver.forward(vp=model0.vp)

            for i in range(nsrc):
                print('Imaging source %d out of %d' % (i+1, nsrc))
                
                # Update source location
                geometry.src_positions[0, :] = src[i, :]

                # Generate synthetic data from true model
                true_d, _, _ = solver.forward(vp=model.vp)
                
                # Compute smooth data and full forward wavefield u0
                smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)
                
                # Compute gradient from the data residual  
                v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
                residual = smooth_d.data - true_d.data
                op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
                        residual=residual)
                gc.collect()
                
            from examples.seismic import plot_image
            plot_image(np.diff(image.data, axis=1))

        case "real":
            solver = AcousticWaveSolver(model, geometry, space_order=4)

            images = []
            image = Function(name='image', grid=model.grid)
            op_imaging = ImagingOperator(model, image)

            for i in range(nshots):
                print('Imaging source %d out of %d' % (i+1, nshots))
                idx = sou_x == np.unique(sou_x)[i] # В моем скрипте sou_x в одиночку определял ансамбль
                csg_ensemble = data[idx, :]
                csg_rec_z = rec_z[idx]
                # Update source location
                geometry.src_positions[0, :] = src[i, :]

                # Вызов для модельных данных
                # true_d, _, _ = solver.forward(vp=model.vp)
                
                # Вызов для модельных данных
                # smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)

                # Вызов для реальных данных - smooth_d не нужен
                _, u0, _ = solver.forward(vp=model0.vp, save=True)
                
                # Compute gradient from the data residual  
                v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)


                # Receiver нужен чисто для того, чтобы создать объект data.data
                real_d = Receiver(name='rec', grid=model.grid,
                        time_range=TimeAxis(start=t0, step=samples[1], num=samples.size), npoint=nreceivers,
                        coordinates=rec)
                real_d.data[:] = csg_ensemble.T
                real_d = real_d.resample(model.critical_dt)

                # Вызов для реальных данных - резидуал здесь это сразу поле отраженных волн
                op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
                        residual=real_d.data)

                # Вызов для модельных данных
                # op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
                #            residual=true_d.data-smooth_d.data)


                # Немного поменял исходный скрипт, чтобы можно было смотреть имейджи от отдельных шотов
                images.append(np.array(image.data))
                image.data.fill(0.)
                # del v, real_d, smooth_d, u0
                # del v, u0 # ХЗ работает ли эта фигня
                gc.collect() # Вот эта фигня точно ускоряет процесс

        case "model":
            solver = AcousticWaveSolver(model, geometry, space_order=4)

            images = []
            image = Function(name='image', grid=model.grid)
            op_imaging = ImagingOperator(model, image)

            reg = sorted(glob.glob(model_param['res_path'][0]+'/*.sgy'))
            smth = sorted(glob.glob(model_param['res_path'][1]+'/*.sgy'))
            assert len(reg) == len(smth), 'Разное количество сглаженных и несглаженных сейсмограмм'
            for i in range(2): #nshots
                print('Imaging source %d out of %d' % (i+1, nshots))

                regular = segyio.open(reg[i], mode='r', endian='big', ignore_geometry=True)
                regular_data = segyio.tools.collect(regular.trace[:])
                samples = regular.samples
                sou_x = regular.attributes(segyio.TraceField.SourceX)[:]/100
                regular.close()

                print(sou_x)
                print(geometry.rec_positions)

                smooth = segyio.open(smth[i], mode='r', endian='big', ignore_geometry=True)
                smooth_data = segyio.tools.collect(smooth.trace[:])
                smooth.close()

                # Update source location
                geometry.src_positions[:, 0] = np.unique(sou_x)
                geometry.src_positions[:, 1] = 0

                # Вызов для реальных данных - smooth_d не нужен
                _, u0, _ = solver.forward(vp=model0.vp, save=True)
                
                # Compute gradient from the data residual  
                v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

                # Receiver нужен чисто для того, чтобы создать объект data.data
                real_d = Receiver(name='rec1', grid=model.grid,
                        time_range=TimeAxis(start=t0, step=samples[1], num=samples.size), npoint=nrec,
                        coordinates=rec)
                real_d.data[:] = (smooth_data - regular_data).T
                real_d = real_d.resample(model.critical_dt)

                # fig, axs = plt.subplots(1, 3 ,figsize=(5,10))
                # axs[0].imshow(smooth_data, aspect='auto', vmin=-1e-6, vmax = 1e-6, )
                # axs[1].imshow(regular_data, aspect='auto', vmin=-1e-6, vmax = 1e-6,)
                # axs[2].imshow(smooth_data-regular_data, aspect='auto', vmin=-1e-6, vmax = 1e-6,)
                # plt.show()
                # То же самое для smooth
                # smooth_d = Receiver(name='rec2', grid=model.grid,
                #         time_range=TimeAxis(start=t0, step=samples[1], num=samples.size), npoint=nrec,
                #         coordinates=rec)
                # smooth_d.data[:] = smooth_data.T
                # smooth_d = smooth_d.resample(model.critical_dt)

                # print(real_d.data.shape)
                # print(smooth_d.data.shape)
                # Вызов для реальных данных - резидуал здесь это сразу поле отраженных волн
                op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
                        residual=real_d.data)

                # Немного поменял исходный скрипт, чтобы можно было смотреть имейджи от отдельных шотов
                images.append(np.array(image.data))
                image.data.fill(0.)
                # del v, real_d, smooth_d, u0
                # del v, u0 # ХЗ работает ли эта фигня
                gc.collect() # Вот эта фигня точно ускоряет процесс
    np.save('Results/RTM_'+preset+'_images.npy', images)
    end=time()
    print('Modeling finished in:', end-begin, 'sec')

        
