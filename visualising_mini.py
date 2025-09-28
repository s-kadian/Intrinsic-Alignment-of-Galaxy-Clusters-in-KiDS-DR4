import pandas as pd
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

cluster_df = pd.DataFrame(np.array(ascii.read("./data/map_detections_refined_noBuffer_all.cat", format = 'no_header', delimiter='\s')))
cluster_df.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"Xphys", "Yphys", "Zphys", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBSTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115', 'uid', 'tile']

random = []
#with open('./masks/DR4_temp_randoms') as file:
with open('./randoms/DR4_temp_randoms_refined') as file:
	for line in file:
		if not line.startswith("#"):
			random.append([float(text) for text in line.split()])
random_df = pd.DataFrame(random, columns = ['RA', 'Dec', 'z'])

pos = [180, 1.4]
size = 2
ra = [pos[0] - size/2, pos[0] + size/2]
dec = [pos[1] - size/2, pos[1] + size/2]
rand_cond_dec = (random_df.Dec.between(dec[0], dec[1]))
rand_cond_ra = (random_df.RA.between(ra[0], ra[1]))
data_cond_dec = (cluster_df.Yphys.between(dec[0], dec[1])) 
data_cond_ra = (cluster_df.Xphys.between(ra[0], ra[1]))
#random_df_ = random_df[(random_df.Dec.between(dec_[0], dec_[1])) and (random_df.RA.between(ra_[0], ra_[1]))]
#cluster_df_ = cluster_df[(cluster_df.Yphys.between(dec[0], dec[1])) and (cluster_df.Xphys.between(ra[0], ra[1]))]
plt.scatter(random_df[rand_cond_dec][rand_cond_ra].RA, random_df[rand_cond_dec][rand_cond_ra].Dec, s = 3, c = 'plum')
plt.scatter(cluster_df[data_cond_dec][data_cond_ra].Xphys, cluster_df[data_cond_dec][data_cond_ra].Yphys, s = 3, c = 'purple')
plt.xlabel
plt.show()
plt.close()