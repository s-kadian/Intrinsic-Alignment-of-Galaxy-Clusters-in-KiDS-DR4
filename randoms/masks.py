from astropy.io import ascii, fits
import pandas as pd
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import sys

my_path = os.path.dirname(os.path.abspath(__file__))
catalog_path = "MASKS_AMICO_FIX/"
filenames = sorted(os.listdir(catalog_path))

n_total = 52442 * 10
total_weight = 0
for file in filenames:
	mask = fits.open(catalog_path+file)
	mask_info = np.array(mask[0].data)[20:220, 20:220]
	#mask_info = np.where(mask_info!=0, 1, mask_info)
	total_weight += np.sum(1 - mask_info)
tile = 0

cluster_df = pd.DataFrame(np.array(ascii.read("../data/map_detections_refined_noBuffer_all.cat", format = 'no_header', delimiter='\s')))
cluster_df.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"Xphys", "Yphys", "Zphys", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBSTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115', 'uid', 'tile']
z = cluster_df.Zphys
sigma = 20
for i in range(len(z)):
	z[i] = round(np.random.normal(z[i], sigma/3000, None), 3)
zpdf, zbin = np.histogram(z, bins=1000, range=(0,1))
z_values = np.zeros(1000)
z_prob = zpdf/np.sum(zpdf)

for i in range(1000):
	z_values[i] += (zbin[i]+zbin[i+1])/2

for file in filenames:
	if tile != 0:
		output_old = output
	ra_mid = mask[0].header['CENTER_1']
	ra_step = mask[0].header['STEP_1']
	dec_mid = mask[0].header['CENTER_2']
	dec_step = mask[0].header['STEP_2']
	
	ra = np.arange(start=ra_mid-(ra_step*120), stop=ra_mid+(ra_step*119.5), step=ra_step)[21:219]
	dec = np.arange(start=dec_mid-(dec_step*120), stop=dec_mid+(dec_step*119.5), step=dec_step)[21:219]

	mask = fits.open(catalog_path+file)
	#temp_member_data = masks[1].data
	mask_info = np.array(mask[0].data)[21:219, 21:219]
	#mask_info = np.where(mask_info!=0, 1, mask_info)
	weight = np.sum(1 - mask_info)
	n = int(np.round (weight * n_total / total_weight))

	ra_prob = np.sum(mask_info, axis = 0)
	ra_prob = ra_prob / np.sum(ra_prob)
	dec_prob = np.sum(mask_info, axis = 1)
	dec_prob = dec_prob / np.sum(dec_prob)

	rand_ra = np.around(np.random.choice(ra, n, p = ra_prob), 3)
	rand_dec = np.around(np.random.choice(dec, n, p = dec_prob), 3)

	output = np.column_stack((rand_ra, rand_dec))
	if tile != 0:
		output = np.vstack((output_old, output))
	print(tile)
	tile += 1

rand_z = np.around(np.random.choice(z_values, len(output), p = z_prob), 3)
output = np.column_stack((output, rand_z))

np.savetxt("DR4_temp_randoms_refined", output, header='\t'.join(('RA', 'dec', 'z')))