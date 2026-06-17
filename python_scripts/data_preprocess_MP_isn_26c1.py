#!/APSshare/anaconda/x86_64/bin/python
import h5py
import numpy as np
import os #for change directory
import scipy.io as sio #for read/write matlab file
from numpy import *
import time
import scipy.ndimage
import hdf5plugin
import glob
import pdb
import multiprocessing

import sys
import csv


# sys.path.append("/home/beams1/B222816/scripts")
sys.path.append("/net/micdata/data3/ptycho_tools/utility")
from readMDA import *
import datetime

#scan= 20

scans=[292]
user_dir='/net/micdata/data1/isn/2026-1/2026-1-Luktuke/'
data_dir =user_dir+'Raw/Scan_{scanNo:04d}/PTYCHO/'
result_dir = user_dir+'results//scan{:03d}/'
pos_dir=user_dir+'Processed/SOCKETSERVER/'
#xrf_file = user_dir+'img.dat/bnp_fly{:04d}.mda.h5'
#xrf_file = user_dir+'img.dat/bnp_fly{:04d}.mda.h5'
#mda_file=user_dir+'mda/bnp_fly{:04d}.mda'



energy = 15
det_sample_dist = 6.16
det_pixel_size = 75e-6
cen_x = 517
cen_y = 801  #756
sds= 2  #scan points downsampling
det_Npixel = 256
lam = 1.23984193e-9/energy
dx = lam*det_sample_dist/det_pixel_size/det_Npixel #pixel size
filePath = '/entry/data/data'
print("real-space pixel size: " + str(dx))

                # Calculate crop indices
y_start = int(cen_y - det_Npixel//2)
y_end = int(cen_y + (det_Npixel + 1)//2)
x_start = int(cen_x - det_Npixel//2)
x_end = int(cen_x + (det_Npixel + 1)//2)


N_scan_x_lb = 0
N_scan_y_lb = 0
roi=f'0_Ndp{det_Npixel}_sds{sds}' 

def process_file(i, data_dir, scanNo, N_scan_y_lb, filePath, y_start, y_end, x_start, x_end):
    fileName = data_dir.format(scanNo=scanNo) + 'scan_{:04d}_{:05d}.h5'.format(scanNo, i + N_scan_y_lb+1)
    try:
        h5_data = h5py.File(fileName, 'r')
        dp_temp = h5_data[filePath][()]
        dp_temp[dp_temp < 0] = 0
        dp_temp[dp_temp > 1e7] = 0
        print(fileName, dp_temp.shape)
        ith_line_N = dp_temp.shape[0]
        if ith_line_N < 5:
            print(f'A lot of pixels are missed on this line: {ith_line_N} pixels, Skip!')
            h5_data.close()
            return None

        dp_temp = np.array(dp_temp)
        dp_crop = dp_temp[:, y_start:y_end, x_start:x_end]
        h5_data.close()

        return dp_crop
    except Exception as e:
        print(f"Has problem with opening file: {fileName}, Error: {str(e)}")
        return None

def main():
   #while True:
   for i in range(1):
        for scanNo in scans:
            
            #data_dir_next = mda_file.format(scanNo)
            #dataExist = os.path.isfile(data_dir_next)
            #if not dataExist:
                #print('Scan No.' + str(scanNo) + '-Data does not exist!')
                #continue

            dataName = 'data_roi' + roi + '_dp.hdf5'
            dataExists = os.path.isfile(result_dir.format(scanNo)+dataName)
            if dataExists:
                print('Scan No.' + str(scanNo) + '-Data already exists!')
                continue    
            #pdb.set_trace()
            if not os.path.exists(result_dir.format(scanNo)): os.makedirs(result_dir.format(scanNo))
            print(result_dir.format(scanNo))

            #############Getting scan position############


            #N_scan_x = x_pos.shape[0]
            #print ('X scan points: {:d}'.format(N_scan_x))
            # Determine number of HDF5 files for this scan and use that as N_scan_y
            data_dir_scan = data_dir.format(scanNo=scanNo)
            h5_pattern = os.path.join(data_dir_scan, f'scan_{scanNo:04d}_*.h5')
            h5_files = sorted(glob.glob(h5_pattern))
            N_scan_y = len(h5_files)
            print(f'Found {N_scan_y} diffraction files under {data_dir_scan}')

            #############Loading diffraction patterns and positions############
            diffs = []
            scan_posx = []
            scan_posy = []

            # Create a pool of workers
            with multiprocessing.Pool() as pool:
                results = pool.starmap(process_file, [(i, data_dir, scanNo, N_scan_y_lb, filePath, y_start, y_end, x_start, x_end) for i in range(N_scan_y - N_scan_y_lb)])
                #results = pool.starmap(process_file, [(i, data_dir, scanNo, N_scan_y_lb, filePath, y_start, y_end, x_start, x_end, x_pos, y_pos) for i in range(51,76)])
            # Collect results from the pool
            for result in results:
                if result is not None:
                    dp_crop = result
                    if len(diffs) == 0:
                        diffs = dp_crop
                    else:
                        diffs = np.append(diffs, dp_crop, axis=0)

            #print scan_posx.shape[0]
            #print scan_posy.shape[0]
            print (f"Number of diffraction patterns: {len(diffs)}")
            pos_file= f'{pos_dir}scan_{scanNo:d}_position.csv'
            positions=np.genfromtxt(pos_file, delimiter=',', skip_header=1)
            ppX = -positions[:,1]*1E-6
            ppY = positions[:, 2]*1E-6
            print(f"ppX shape: {ppX.shape}\n ppY shape: {ppY.shape}")
            #data_points = min(diffs.shape[0], ppX.shape[0])
            start_dp=201*50
            data_points= ppX.shape[0]-start_dp
            print(f"Min Data points: {data_points}")
            ##################### save data ##########################
            print("saving " + 'data_roi' + roi + '.hdf5' + " to " + result_dir.format(scanNo))
            f = h5py.File(result_dir.format(scanNo) + 'data_roi' + roi + '_dp.hdf5', "w")
            f.create_dataset("dp", dtype='float32', data=diffs[start_dp:data_points:sds], compression="gzip")
            #f.create_dataset("dp", data=diffs, compression="gzip")
            f.close()
            #save other parameters
            f = h5py.File(result_dir.format(scanNo) + 'data_roi' + roi + '_para.hdf5', "w")
            f.create_dataset("lambda", shape=(1,),dtype='float64', data=lam)
            f.create_dataset("dx", shape=(1,),dtype='float64', data=dx)
            f.create_dataset("ppY", dtype='float64', data=ppY[start_dp:data_points:sds])
            f.create_dataset("ppX", dtype='float64', data=ppX[start_dp:data_points:sds])

            f.create_dataset("N_scan_y_lb", shape=(1,),dtype='int', data=N_scan_y_lb)
            f.create_dataset("N_scan_x_lb", shape=(1,),dtype='int', data=N_scan_x_lb)
            f.create_dataset("N_scan_y", shape=(1,),dtype='int', data=N_scan_y)
            #f.create_dataset("N_scan_x", shape=(1,),dtype='int', data=N_scan_x)
            f.close()
        time.sleep(5)

if __name__ == "__main__":
    main()
