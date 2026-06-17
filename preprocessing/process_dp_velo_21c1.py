import h5py
import numpy as np
import os #for change directory
import scipy.io as sio #for read/write matlab file
from numpy import *
import time
import scipy.ndimage
import glob
import sys
import csv

userdata_dir = '/mnt/micdata2/velociprobe/2021-1/commission/'
#N_scan_y_lb_s = np.array([0,89,178,267,356])
scanNo_s = [100]


#pos_dir = '/mnt/micdata2/lamni/2020-2/comm_33IDD_2/specES1/scan_positions/'

#for scanNo in scanNo_s:
for i in range(0,len(scanNo_s)):
    scanNo = scanNo_s[i]
    print('scanNo:',scanNo)
    #data_dir = '/mnt/micdata2/velociprobe/2021-1/commission/ptycho/scan' + '%03d' %  (scanNo) + '/' 
    #result_dir = '/mnt/micdata2/velociprobe/2021-1/commission/results/TP_ML/scan' + '%03d' %  (scanNo) + '/' 
    data_dir = '/mnt/micdata2/velociprobe/2021-1/commission/ptycho/fly' + '%03d' %  (scanNo) + '/' 
    result_dir = '/mnt/micdata2/velociprobe/2021-1/commission/results/TP_ML/fly' + '%03d' %  (scanNo) + '/' 
    det_Npixel = 64
    roi = '0_Ndp' + str(det_Npixel)
    ############### check if data is already processed #################
    dataName = 'data_roi' + roi + '_dp.hdf5'
    dataExists = os.path.isfile(result_dir+dataName)
    if dataExists:
        print('Scan No.' + str(scanNo) + '-Data already exists!')
        #continue


    print("Creating geometry")
    energy = 10.0
    det_sample_dist = 2.335 #measured
    det_pixel_size = 75e-6

    lam = 1.23984193e-9/energy
    dx = lam*det_sample_dist/det_pixel_size/det_Npixel #pixel size
    print("real-space pixel size: " + str(dx))
    print("wavelength: " + str(lam))

    ############## load data ######################
    print("loading diffraction patterns")
    cen_x = 536
    cen_y = 259
    N_dp_x_input = np.min([500,det_Npixel])
    N_dp_y_input = np.min([500,det_Npixel])
    index_x_lb = (cen_x - np.floor(N_dp_x_input/2.0)).astype(np.int)
    index_x_ub = (cen_x + np.ceil(N_dp_x_input/2.0)).astype(np.int)
    index_y_lb = (cen_y - np.floor(N_dp_y_input/2.0)).astype(np.int)
    index_y_ub = (cen_y + np.ceil(N_dp_y_input/2.0)).astype(np.int)
 
    saveDiffractionPattern = True
    resampleFactor = 1
    resizeFactor = 1

    filePath = 'entry/data/data'
    roi = '0_Ndp' + str(det_Npixel)
    if resampleFactor>1:
        roi = roi + '_resample'+str(resampleFactor)

    ############## determine N_scan_x ######################
    fileName = data_dir + 'fly'+'%03d' %  (scanNo)  + '_data_' + '%06d' %  (1) +'.h5'
    #fileName = data_dir + 'scan'+'%03d' %  (scanNo)  + '_data_' + '%06d' %  (1) +'.h5'
    h5_data = h5py.File(fileName,'r')
    dp_temp = h5_data[filePath].value
    N_scan_x = dp_temp.shape[0]
    print('N_scan_x='+str(N_scan_x))


    ############## determine N_scan_y ######################
    list = os.listdir(data_dir)
    N_scan_y = len(list)-1
    print('N_scan_y='+str(N_scan_y))

    N_scan_x_lb = 0
    N_scan_y_lb = 0
    
    dp = np.zeros((N_scan_y*N_scan_x,int(N_dp_y_input*resizeFactor),int(N_dp_x_input*resizeFactor)))

    print(dp.shape)
    for i in range(N_scan_y):
        fileName = data_dir + 'fly'+'%03d' %  (scanNo)  + '_data_' + '%06d' %  (i+1+N_scan_y_lb) +'.h5'
        #fileName = data_dir + 'scan'+'%03d' %  (scanNo)  + '_data_' + '%06d' %  (i+1+N_scan_y_lb) +'.h5'
        h5_data = h5py.File(fileName,'r')
        dp_temp = h5_data[filePath].value
        print(fileName, dp_temp.shape)
        for j in range(N_scan_x):
            index = i*N_scan_x + j
            scipy.ndimage.interpolation.zoom(dp_temp[j+N_scan_x_lb,index_y_lb:index_y_ub,index_x_lb:index_x_ub],[resizeFactor,resizeFactor],dp[index,:,:], 1)

    dp[dp<0] = 0
    dp[dp>1e7] = 0

    dp = dp[::resampleFactor,:,:]

    ############## load scan positions ######################
    print("loading scan positions")
    position_dir = '/mnt/micdata2/velociprobe/2021-1/commission/positions/' 
    #position_file = 'scan' + '%03d' %  (scanNo) + '_pos.csv'
    position_file = 'fly' + '%03d' %  (scanNo) + '_pos.csv'
    with open(position_dir + position_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        N_scan_tot = sum(1 for row in spamreader)
    ppX = np.zeros((N_scan_tot,1))
    ppY = np.zeros((N_scan_tot,1))
    i = 0

    with open(position_dir + position_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            temp = row[0].split(',')
            ppY[i] = float(temp[0])
            ppX[i] = float(temp[1])
            i = i + 1
    ppX = ppX[0:N_scan_x*N_scan_y]
    ppY = ppY[0:N_scan_x*N_scan_y]    

    '''
    if N_scan_pos>N_scan_dp: #if there are more positions than dp
        ppX = ppX[0:N_scan_dp]
        ppY = ppY[0:N_scan_dp]
    else:
        dp = dp[0:N_scan_pos,:,:]
    '''

    #ppY = ppY_tot[N_scan_x*N_scan_y_lb:N_scan_x*N_scan_y_lb+N_scan_x*N_scan_y]
    #ppX = ppX_tot[N_scan_x*N_scan_y_lb:N_scan_x*N_scan_y_lb+N_scan_x*N_scan_y]

    #shift positions to center around 0,0)
    #ppX = ppX - (np.max(ppX) + np.min(ppX))/2
    #ppY = ppY - (np.max(ppY) + np.min(ppY))/2

    ##################### resample dp ##########################
    #ppX = ppX[::resampleFactor]
    #ppY = ppY[::resampleFactor]

    #dp = dp[0:N_scan_tot,:,:]

    ##################### pad zeros to dp ##########################
    if det_Npixel>N_dp_x_input: #currently only works for even size
        pad_before = (det_Npixel-N_dp_x_input)//2
        pad_after = (det_Npixel-N_dp_x_input)//2
        dp = np.pad(dp,((0,0),(pad_before,pad_after),(pad_before,pad_after)),'constant')

    print(dp.shape)
    
    
    ##################### save data ##########################
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    print(result_dir)

    print("saving " + 'data_roi' + roi + '.hdf5' + " to " + result_dir)
    #save diffraction pattern
    f = h5py.File(result_dir + 'data_roi' + roi + '_dp.hdf5', "w")
    f.create_dataset("dp", shape=dp.shape,dtype='float32', data=dp, compression="gzip")
    f.close()
    
    
    #save other parameters
    f = h5py.File(result_dir + 'data_roi' + roi + '_para.hdf5', "w")
    #f.create_dataset("angle", shape=(1,),dtype='float64', data=rot_ang)
    f.create_dataset("lambda", shape=(1,),dtype='float64', data=lam)
    f.create_dataset("dx", shape=(1,),dtype='float64', data=dx)
    f.create_dataset("ppY", shape=ppY.shape,dtype='float64', data=ppY)
    f.create_dataset("ppX", shape=ppX.shape,dtype='float64', data=ppX)
    f.create_dataset("N_scan_y_lb", shape=(1,),dtype='int', data=N_scan_y_lb)
    f.create_dataset("N_scan_x_lb", shape=(1,),dtype='int', data=N_scan_x_lb)
    f.create_dataset("N_scan_y", shape=(1,),dtype='int', data=N_scan_y)
    f.create_dataset("N_scan_x", shape=(1,),dtype='int', data=N_scan_x)
    f.close()
    
    #except:
    #    print("An exception occurred")
    #    file = open(userdata_dir + 'bad_data.txt','a') 
    #    file.write(str(scanNo)+'\n')
    #    file.close()

    #print('wait for 1 min')
    #time.sleep(60)
