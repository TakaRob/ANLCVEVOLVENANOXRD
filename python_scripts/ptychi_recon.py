import os
import ptychi.pear as pear
import time
import traceback
import sys
sys.path.append("/net/micdata/data3/ptycho_tools/utility")


# Configuration parameters
scan_list = [302]  # Predetermined list of scans to process

gpu_id = 2

number_of_iterations = 2000  # the size used in this reconstruction

preprocess_size = 128 # 
downsampling= 1
det_sample_dist_m = 6.16
beam_energy_kev = 15
det_sample_dist_m/=downsampling
ROI=0
data_roi_name=f'data_roi{ROI:d}_Ndp{preprocess_size*2}_sds1'
print(data_roi_name)
# setup folder structure
# to add scan number append .format(scan_num)
data_main_dir            = "/net/micdata/data1/isn/2026-1/2026-1-Luktuke"
h5_dir                   = os.path.join(data_main_dir,'ptycho')
results_path             = os.path.join(data_main_dir, 'results/scan{:03d}') #ML_results_path
reconstructed_probe_path = os.path.join(results_path, 'temp')

data_file_name           = os.path.join(results_path, f'{data_roi_name}_dp.hdf5')
para_file_name           = os.path.join(results_path, f'{data_roi_name}_para.hdf5')


#default_init_probe = '/mnt/micdata3/Amey/Experimental_data/cement_isn_feb2026/probe_fzp.h5'
default_init_probe = '/net/micdata/data1/isn/2026-1/2026-1-Luktuke/ptychi_recons/S0174/Ndp128_LSQML_s1667_gaussian_p5_cp_opr1_ic_pc1000_f_ul2_dpFlip_ud/recon_Niter5000.h5'


for i in range(1):
#while True:
	# Use predetermined list of scans
	# Apply GPU distribution if using multiple GPUs
	scans = scan_list
	
	print(f'Processing scans: {scans}')


	for scan_num in scans:
		
		# Check if preprocessed data exists
		#if not os.path.isfile(data_file_name.format(scan_num)):
			#print(f'{data_file_name.format(scan_num)} data doesn\'t exist! Please preprocess the data first!')
			#continue
		
		params = {
			'data_directory': data_main_dir,
			#path_to_init_probe': reconstructed_probe_file.format(scan_num-1) if os.path.isfile(reconstructed_probe_file.format(scan_num-1)) else init_probe_path,  # init_probe_path,
			'path_to_init_probe': default_init_probe, #init_probe_path.format(scan_num-2) if os.path.isfile(init_probe_path.format(scan_num)) else default_init_probe,  
			'path_to_init_object': '',               #init_object_path,
			'path_to_init_positions': '',  #path to init position
			'scan_num': scan_num,
			'instrument': 'isn',
			'diff_pattern_size_pix': preprocess_size,
			'diff_pattern_center_x': 128,
			'diff_pattern_center_y': 128,
			'beam_energy_kev': beam_energy_kev,
			'det_sample_dist_m': det_sample_dist_m,   #saxs 10, waxs9.77
			'load_processed_hdf5': True,
			'path_to_processed_hdf5_dp': data_file_name.format(scan_num),
			'path_to_processed_hdf5_pos': para_file_name.format(scan_num),
			'flip_diffraction_patterns_up_down': True,
			'position_correction': True,
			'position_correction_update_limit': 2,
			'position_correction_affine_constraint': False,
			'position_correction_gradient_method': 'fourier',
			'position_correction_start_iteration': 500,
			"use_model_FZP_probe": False,
			"init_probe_propagation_distance_mm": 0,
			"orthogonalize_initial_probe": True,
			"intensity_correction": True,
			"center_probe": True,
			"probe_support": False,
			"number_probe_modes": 5,
			"update_object_w_higher_probe_modes": False,
			"number_opr_modes": 1,
			"update_batch_size": None,
			"number_of_batches": 20,
			"batch_selection_scheme": "uniform",
			"momentum_acceleration": False,
			"number_of_slices": 1,
			"object_thickness_m": 4e-05,
			"layer_regularization": 0,
			"position_correction_layer": None,
			"number_of_iterations": number_of_iterations,
			"save_freq_iterations": 100,
			"recon_dir_suffix": "",
			"recon_parent_dir": "",
			"gpu_id": gpu_id,
			"save_diffraction_patterns": False,
			'collect_object_phase': True,
			'collect_probe_magnitude': True,
			'add_random_posError': 20e-9,
			
		}

		batching_mode_suffix = {
			'compact': 'c',
			'random': 'r',
			'uniform': 's'
		}.get(params['batch_selection_scheme'], '')



		try:
			# Check if required files exist
			file_paths = [
				params['path_to_init_probe'],
				#params['path_to_processed_hdf5_dp'],
				#params['path_to_processed_hdf5_pos']
			]
			
			for path in file_paths:
				if not os.path.exists(path):
					raise ValueError(f"Required file not found: {path}")

			# Attempt reconstruction
			pear.ptycho_recon(**params)

			print(f"Successfully completed reconstruction for scan {scan_num}")

		except Exception as e:
			print(f"\nError in reconstruction for scan {scan_num}:")
			print(f"Error type: {type(e).__name__}")
			print(f"Error message: {str(e)}")
			print("\nFull traceback:")
			traceback.print_exc()
			
			# Log parameter state for debugging
			# print("\nParameter values at time of error:")
			# for key, value in sorted(params.items()):
			# 	print(f"  {key}: {value}")
			
			print(f"\nSkipping scan {scan_num} and continuing with reconstruction of next scan...")
			
			continue
		time.sleep(5)
