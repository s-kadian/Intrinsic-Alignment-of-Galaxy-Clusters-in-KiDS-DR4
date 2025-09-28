import pandas as pd
import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt, cm, colors
from matplotlib.colors import ListedColormap
import treecorr
from astropy.cosmology import Planck18
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
import os

plt.rcParams['text.usetex'] = True

cluster_df = pd.DataFrame(np.array(ascii.read("./data/map_detections_refined_noBuffer_all.cat", format = 'no_header', delimiter='\s')))
cluster_df.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"XSKY", "YSKY", "ZSKY", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBDASTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115', 'uid', 'tile']

random = []
with open('./randoms/DR4_temp_randoms_refined') as file:
	for line in file:
		if not line.startswith("#"):
			random.append([float(text) for text in line.split()])
random_df = pd.DataFrame(random, columns = ['RA', 'Dec', 'redshift'])

ellipticity_df = pd.read_pickle("./ellipticity/merged_df.pkl")
ellipticity_df.columns = ["ID", "XSKY", "YSKY", "ZSKY", "LAMB", "LAMBDASTAR", 'uid', 'tile', 'epsilon', 'abs_epsilon', 'angle']


#cluster_df_split = np.array_split(cluster_df[cluster_df.ZSKY <= 1][cluster_df.LAMBDASTAR>=15].sort_values(by = 'ZSKY'), 3)
ellipticity_df_split = np.array_split(ellipticity_df[ellipticity_df.LAMBDASTAR >= 15].sort_values(by = 'LAMBDASTAR'), 3)
ellipticity_df_split_z = np.array_split(ellipticity_df[ellipticity_df.LAMBDASTAR >= 15].sort_values(by = 'ZSKY'), 3)

z = []
lambdastar = []
for i in range(3):
	z.append(ellipticity_df_split_z[i].ZSKY.min())
	lambdastar.append(ellipticity_df_split[i].LAMBDASTAR.min())
	if i==2:
		z.append(ellipticity_df_split_z[i].ZSKY.max())
		lambdastar.append(ellipticity_df_split[i].LAMBDASTAR.max())

plt.scatter(ellipticity_df[ellipticity_df.LAMBDASTAR>=15].ZSKY, np.log10(ellipticity_df[ellipticity_df.LAMBDASTAR>=15].LAMBDASTAR), marker = '.', s=5, c='mediumvioletred', alpha = 0.25)
for i in range(4):
	plt.axvline(z[i], color='k', linestyle='dashed', linewidth=1)
	plt.axhline(np.log10(lambdastar[i]), color='k', linestyle='dashed', linewidth=1)
plt.xlim([z[0], z[-1]])
plt.ylim(np.log10([lambdastar[0], lambdastar[-1]]))
plt.xlabel(r'$z$')
plt.ylabel(r'$log_{10}(\lambda^{*})$')
plt.savefig('shape_catalog.png', dpi=300)
plt.close()

plt.scatter(cluster_df[cluster_df.LAMBDASTAR>=15].ZSKY, np.log10(cluster_df[cluster_df.LAMBDASTAR>=15].LAMBDASTAR), marker='o', s=5, c='darkmagenta', alpha = 0.25)
for i in range(4):
	plt.axvline(z[i], color='k', linestyle='dashed', linewidth=1)
	#if i!=3:
	#	print(cluster_df[cluster_df.LAMBDASTAR>=15][cluster_df[cluster_df.LAMBDASTAR>=15].ZSKY.between(z[i], z[i+1])].shape[0])
	#plt.axhline(np.log10(lambdastar[i]), color='k', linestyle='dashed', linewidth=1)
plt.xlim([z[0], z[-1]])
plt.ylim(np.log10([lambdastar[0], lambdastar[-1]]))
plt.xlabel(r'$z$')
plt.ylabel(r'$log_{10}(\lambda^{*})$')
plt.savefig('cluster_catalog.png', dpi=300)
plt.close()