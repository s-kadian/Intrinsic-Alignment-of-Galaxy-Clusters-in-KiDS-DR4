from astropy.io import ascii, fits
import pandas as pd
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as ptches

def ellipticity(centre, theta, prob):
	separation = theta - centre.values																			#calculating separations of members from cluster centre
	separation.loc[separation['RA']>30, 'RA'] = separation.loc[separation['RA']>30, 'RA'] - 360					#fixing the dicontinuity at 0-360 arising due to cyclic nature of coordinates
	separation.loc[separation['RA']<-30, 'RA'] = separation.loc[separation['RA']<-30, 'RA'] + 360				
	#separation.loc[separation['Dec']>30, 'Dec'] = separation.loc[separation['Dec']>30, 'Dec'] - 360				#fixing the dicontinuity at 180-(-180) arising due to cyclic nature of coordinates
	#separation.loc[separation['Dec']<-30, 'Dec'] = separation.loc[separation['Dec']<-30, 'Dec'] + 360

	correction = np.absolute(np.cos(np.radians((theta['Dec'] + centre['Yphys'].values)/2)))
	separation['RA'] = separation['RA']*correction

	separation_scaled = separation*prob.values																				

	Q = (separation.T.dot(separation_scaled)).values															#'the correlation matrix'
	epsilon = (Q[0, 0] - Q[1, 1] + 2j*Q[0,1])/(Q[0, 0] + Q[1,1] + 2*np.sqrt(np.linalg.det(Q)))					#epsilon	

	return epsilon

def plot_frequencies(ell_data):
	ell_data["angle"].hist(bins = 50, grid = False, color='purple', edgecolor = 'black')
	plt.xlabel("Angle in degrees")
	plt.ylabel("Frequency")
	#plt.text(55, 65, "Mean = " + str(round(ell_data["angle"].mean(), 3)))
	plt.savefig('frequency_angle', dpi = 200)
	plt.close()

	np.absolute(ell_data["epsilon"]).hist(bins = 30, grid = False, color='purple', edgecolor = 'black')
	plt.xlabel(r"|$\epsilon$|")
	plt.ylabel("Frequency")
	#plt.text(0.32, 90, "Mean = " + str(round(np.absolute(ell_data["epsilon"]).mean(),4)))
	#plt.text(0.32, 85, r"Mean of $\epsilon$= " + str(round(ell_data["epsilon"].mean().real,3)+round(ell_data["epsilon"].mean().imag,3)*1j))
	plt.axvline(np.absolute(ell_data["epsilon"]).mean(), color='k', linestyle='dashed', linewidth=1)
	plt.savefig('frequency_absolute_epsilon_ppt', dpi = 200)
	plt.close()

	return None

def get_ellipticity_data():
	my_path = os.path.dirname(os.path.abspath(__file__))
	catalog_path = "../DETECTIONS_DERIVED/"
	files = iter(sorted(os.listdir(catalog_path))[0:-3]) 		#[0:-3]

	ellipticity_data = []
	total_counts = []

	for filename in files:
		with fits.open(catalog_path+filename) as member_file:		#member file .fit; cluster file .cat
			tile = filename.split('_m')[0]
			print(tile)

			cluster_data = pd.DataFrame(np.array(ascii.read(catalog_path+next(files))))
			cluster_data.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"Xphys", "Yphys", "Zphys", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBSTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115']

			temp_member_data = member_file[1].data
			member_data = []
			for member in temp_member_data:
				if member[4]!=0 and any(float(prob)>=0.8 for prob in member[6]):
					temp = [member[0], member[1], member[2], member[5][0], member[6][0]]
					member_data.append(temp)
			member_data = pd.DataFrame(member_data, columns = ['ID', 'RA', 'Dec', 'clusterID', 'prob'])
			counts = member_data["clusterID"].value_counts()
			#total_counts += counts.values
			total_counts.extend(counts.values)
			IDs = counts.index[counts.gt(20)]

			sns.scatterplot(data = member_data[(member_data['clusterID'].isin(IDs))], x = 'RA', y = 'Dec', hue='clusterID', legend=False, palette = sns.color_palette("bright"), s=3)

			for ID in IDs:

				#Calculating ellipticity
				centre = cluster_data[cluster_data.ID == ID][['Xphys', 'Yphys']]
				theta = member_data[member_data.clusterID == ID][['RA','Dec']]
				prob = member_data[member_data.clusterID == ID][['prob']]
				E = ellipticity(centre, theta, prob)
				phi = np.degrees(np.angle(E))/2																		#angle of ellipse in degrees
				axis_ratio = (1 - np.absolute(E))/(1 + np.absolute(E))											#axis ratio of ellipse
				ellipticity_data.append([ID, tile, E, np.absolute(E), phi])	#ID, tile, complex ellipticity, absolute ellipticity, angle
				E 

				#Plotting ellipses
				sns.scatterplot(data = centre, x = 'Xphys', y = 'Yphys', c='black', s=10)
				ax = plt.gca()
				ells = ptches.Ellipse(centre.values[0], width=0.05, height=0.05*axis_ratio, angle=phi, edgecolor='black', fc='None')		#width choosen randomly
				ax.add_artist(ells)

		plt.savefig(my_path+"/Visualisation/"+str(tile)+'.jpg', dpi=300)
		plt.close()

	plt.hist(total_counts, bins = 200, range = (0,100), color = 'purple')
	plt.text(2.25, 3, "Total clusters:" + str(len(total_counts)), fontsize=10, transform = ax.transAxes)
	plt.xlabel('No. of galaxies in clusters')
	plt.ylabel('No. of clusters')
	plt.savefig(my_path+"/histogram.jpg", dpi = 300)
	plt.close()

	ellipticity_data = pd.DataFrame(ellipticity_data, columns = ['ID', 'tile', 'epsilon', 'abs_epsilon', 'angle'])
	ellipticity_data.to_pickle("./ellipticity_df.pkl")

	return None

def plot_epsilon_lambda(merged_df):
	#merged_df = pd.merge(cluster_df, ellipticity_df, how ='inner', on =['tile', 'ID'])
	#print(merged_df[merged_df.LAMBSTAR<=10.])
	split_df = np.array_split(merged_df.sort_values(by = 'LAMBSTAR'), 3)
	err_color = ['mediumvioletred', 'purple', 'darkblue']																		#error bar colours
	p_color = ['pink', 'plum', 'cornflowerblue']																				#point colours
	for i in range(3):
		plt.scatter(split_df[i]["LAMBSTAR"], split_df[i]["abs_epsilon"], color = p_color[i], s = 1)
		x = split_df[i]["LAMBSTAR"].mean()																				#Calculating the mean of lambda* in the ith bin
		y = split_df[i]["abs_epsilon"].mean()																			#Calculating the mean of |ellipticity| in the ith bin
		y_err = split_df[i]["abs_epsilon"].std()																			#Calculating the error in the ellipticity
		plt.errorbar(x, y, yerr = y_err, fmt='.k', capsize = 3, ecolor = err_color[i], elinewidth = 1, capthick = 1)			#plotting the means and errorbars in the ith bin
	
	plt.xlabel(r'$\lambda^{*}$')
	plt.ylabel(r'$|\epsilon|$')
	plt.savefig('lambda_vs_epsilon', dpi = 200)
	plt.close()

	return None





#get_ellipticity_data()

ellipticity_df = pd.read_pickle("ellipticity_df.pkl")
#plot_frequencies(ellipticity_data)

cluster_df = pd.DataFrame(np.array(ascii.read("../DETECTIONS_DERIVED/map_detections_refined_noBuffer_all.cat", format = 'no_header', delimiter='\s')))
cluster_df.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"Xphys", "Yphys", "Zphys", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBSTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115', 'uid', 'tile']

merged_df = pd.merge(cluster_df[["ID", "Xphys", "Yphys", "Zphys", "LAMB", "LAMBSTAR", 'uid', 'tile']], ellipticity_df, how ='inner', on =['tile', 'ID'])
merged_df.to_pickle("./merged_df.pkl")
#plot_epsilon_lambda(merged_df)



