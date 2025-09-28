import pandas as pd
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

cluster_df = pd.DataFrame(np.array(ascii.read("./data/map_detections_refined_noBuffer_all.cat", format = 'no_header', delimiter='\s')))
cluster_df.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"Xphys", "Yphys", "Zphys", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBSTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115', 'uid', 'tile']

random = []
#with open('./masks/DR4_temp_randoms') as file:
with open('./randoms/DR4_temp_randoms_refined') as file:
	for line in file:
		if not line.startswith("#"):
			random.append([float(text) for text in line.split()])
random_df = pd.DataFrame(random, columns = ['RA', 'Dec', 'z'])

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (11, 7))
ax1.hist(cluster_df['Zphys'], bins = 20, color = 'darkmagenta', edgecolor = 'black')
ax2.hist(random_df['z'], bins = 20, color = 'darkmagenta', edgecolor = 'black',)
for axis in [ax1, ax2]:
	axis.set_xlabel('z')
	#axis.set_ylabel('Frequency')
	if axis == ax1:
		title = 'Cluster Catalog'
	else:
		title = "Random"
	axis.set_title(title)
plt.savefig("visualising_redshifts_compared.png", dpi = 300)
plt.close()

cond1 = [random_df.Dec<-5, random_df.Dec>-5]
cond2 = [cluster_df.Yphys<-5, cluster_df.Yphys>-5]
#fig_name = ['SGC.png', 'equitorial.png']
fig_name = ['visualising_SGC.png', 'visualising_equitorial.png']
for i in range(2):
	plt.figure(figsize = (15, 7))
	plt.scatter(np.where(random_df[cond1[i]]['RA']>=300, random_df[cond1[i]]['RA']-360.0, random_df[cond1[i]]['RA']), random_df[cond1[i]]['Dec'], s=1, marker = '.', c = 'plum', alpha = 0.5)
	plt.scatter(np.where(cluster_df[cond2[i]]['Xphys']>=300, cluster_df[cond2[i]]['Xphys']-360.0, cluster_df[cond2[i]]['Xphys']), cluster_df[cond2[i]]['Yphys'], s=1, marker = '.', c = 'darkmagenta')
	plt.xlabel('RA [degrees]')
	plt.ylabel('dec [degrees]')
	#plt.show()
	plt.savefig(fig_name[i], bbox_inches = 'tight', dpi = 1000)	
	plt.close()