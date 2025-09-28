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

cluster_df = pd.DataFrame(np.array(ascii.read("../data/map_detections_refined_noBuffer_all.cat", format = 'no_header', delimiter='\s')))
cluster_df.columns = ["ID",	"Xpix",	"Ypix",	"Zpix",	"XSKY", "YSKY", "ZSKY", "SN", "SN_NO_CLUSTER", "AMP", "RICH", "LIKE", "MSKFRC", "POISS",	"LAMB",	"LAMBDASTAR",	"Zpix_sigm", "Zpix_sigp", "Zphys_sigm",	"Zphys_sigp", "Pz-0", "Pz-1", "Pz-2", "Pz-3", "Pz-4", "Pz-5", "Pz-6", "Pz-7", "Pz-8", "Pz-9", "Pz-10", "Pz-11", "Pz-12", "Pz-13", "Pz-14", "Pz-15", "Pz-16", "Pz-17", "Pz-18", "Pz-19", "Pz-20", "Pz-21", "Pz-22", "Pz-23",	"Pz-24", "Pz-25", "Pz-26", "Pz-27",	"Pz-28", "Pz-29", "Pz-30", "Pz-31", "Pz-32", 'Pz-33', 'Pz-34', 'Pz-35', 'Pz-36', 'Pz-37', 'Pz-38', 'Pz-39', 'Pz-40', 'Pz-41', 'Pz-42', 'Pz-43', 'Pz-44', 'Pz-45', 'Pz-46', 'Pz-47', 'Pz-48', 'Pz-49', 'Pz-50', 'Pz-51', 'Pz-52', 'Pz-53', 'Pz-54', 'Pz-55', 'Pz-56', 'Pz-57', 'Pz-58', 'Pz-59', 'Pz-60', 'Pz-61', 'Pz-62', 'Pz-63', 'Pz-64', 'Pz-65', 'Pz-66', 'Pz-67', 'Pz-68', 'Pz-69', 'Pz-70', 'Pz-71', 'Pz-72', 'Pz-73', 'Pz-74', 'Pz-75', 'Pz-76', 'Pz-77', 'Pz-78', 'Pz-79', 'Pz-80', 'Pz-81', 'Pz-82', 'Pz-83', 'Pz-84', 'Pz-85', 'Pz-86', 'Pz-87', 'Pz-88', 'Pz-89', 'Pz-90', 'Pz-91', 'Pz-92', 'Pz-93', 'Pz-94', 'Pz-95', 'Pz-96', 'Pz-97', 'Pz-98', 'Pz-99', 'Pz-100', 'Pz-101', 'Pz-102', 'Pz-103', 'Pz-104', 'Pz-105', 'Pz-106', 'Pz-107', 'Pz-108', 'Pz-109', 'Pz-110', 'Pz-111', 'Pz-112', 'Pz-113', 'Pz-114', 'Pz-115', 'uid', 'tile']

random = []
with open('../randoms/DR4_temp_randoms_refined') as file:
	for line in file:
		if not line.startswith("#"):
			random.append([float(text) for text in line.split()])
random_df = pd.DataFrame(random, columns = ['RA', 'Dec', 'redshift'])

ellipticity_df = pd.read_pickle("../ellipticity/merged_df.pkl")
ellipticity_df.columns = ["ID", "XSKY", "YSKY", "ZSKY", "LAMB", "LAMBDASTAR", 'uid', 'tile', 'epsilon', 'abs_epsilon', 'angle']

def clustering_signal(cluster_df, random_df, n_bins, npatches):
	#random_df_ = random_df[(random_df['redshift'].astype(float).between(zmin, zmax))]
	ra_rand = random_df['RA'].values.astype(float)
	dec_rand = random_df['Dec'].values.astype(float)
	r_rand = Planck18.comoving_distance(random_df['redshift'].values.astype(float))*Planck18.h
	km = kmeans_sample(np.column_stack([ra_rand, dec_rand]), npatches, maxiter=100, tol=1.0e-5)
	rand_labels = km.labels
	print(np.unique(rand_labels))
	rand = treecorr.Catalog(ra = ra_rand, dec = dec_rand, r = r_rand, ra_units = 'degrees', dec_units = 'degrees', patch = rand_labels, is_rand = True)

	#cluster_df_ = cluster_df[(cluster_df['ZSKY'].between(zmin, zmax)) & (cluster_df['LAMBDASTAR'] > 15)]
	x = cluster_df['XSKY'].values																						
	y = cluster_df['YSKY'].values
	z = Planck18.comoving_distance(cluster_df['ZSKY'].values)*Planck18.h
	cluster_labels = km.find_nearest(np.column_stack([x, y]))
	data = treecorr.Catalog(ra = x, dec = y, r = z, ra_units = 'degrees', dec_units = 'degrees', patch=cluster_labels)

	zmin = cluster_df.ZSKY.min()
	zmax = cluster_df.ZSKY.max()

	config = {"nbins":n_bins, "min_sep":5.,  "max_sep":120., "metric":'Euclidean', "var_method":'jackknife'}
	dd = treecorr.NNCorrelation(config)																				
	dr = treecorr.NNCorrelation(config)										
	rr = treecorr.NNCorrelation(config)																			

	dd.process(data)
	dr.process(data,rand)
	rr.process(rand)
	fname = 'clustering_['+str(round(zmin,2))+'-'+str(round(zmax, 2))+']'
	dd.write(fname, rr=rr, dr=dr)
	xi, varxi = dd.calculateXi(rr = rr,dr = dr)
	err = varxi**0.5

	clustering = []
	with open(fname) as file:
		for line in file:
			if not line.startswith("#"):
				clustering.append([text for text in line.split()])

	fig = plt.figure()
	ax = plt.gca()
	ax.errorbar(x = dd.meanr, y = xi, yerr = err, fmt='.k', capsize = 3, ecolor = 'black', elinewidth = 1, capthick = 1, ms = 3)
	ax.set_xscale('log')
	plt.text(0.75, .9, r'z $\in$'+' ['+str(round(zmin, 2))+', '+str(round(zmax,2))+']', fontsize=10, transform = ax.transAxes)
	#plt.text(0.75, .9, r'z $\in$'+' ['+str(zmin)+', '+str(zmax)+']', fontsize=10, transform = ax.transAxes)
	ax.set_yscale('log')
	plt.xlabel(r's [$h^{-1}$ Mpc]')
	plt.ylabel(r'$\xi_{gg}$(s)')
	#plt.xlim([5,200])
	#plt.xticks([6, 10, 20, 40, 70], [6, 10, 20, 40, 70])
	#plt.ylim([0.02,1])
	#plt.show()

	plt.savefig(fname+'.png', dpi = 300)
	plt.close()

	return None

########################################################################################################################################################################

def compute_wgg(cluster_df, random_df, min_rpar, max_rpar, nbins_rpar, nbins, outfile, save=True, brute=True):
	Pi = np.linspace(min_rpar, max_rpar, nbins_rpar + 1)

	wgg_3D = np.zeros([nbins_rpar, nbins])
	varw_3D = np.zeros([nbins_rpar, nbins])
	DD_3D = np.zeros([nbins_rpar, nbins])
	DR_3D = np.zeros([nbins_rpar, nbins])
	#RD_3D = np.zeros([nbins_rpar, nbins])
	RR_3D = np.zeros([nbins_rpar, nbins])
	meanr_3D = np.zeros([nbins_rpar, nbins])
	meanlogr_3D = np.zeros([nbins_rpar, nbins])

	x = cluster_df['XSKY'].values																						
	y = cluster_df['YSKY'].values
	z = Planck18.comoving_distance(cluster_df['ZSKY'].values)*Planck18.h
	data = treecorr.Catalog(ra = x, dec = y, r = z, ra_units = 'degrees', dec_units = 'degrees')

	ra_rand = random_df['RA'].values.astype(float)
	dec_rand = random_df['Dec'].values.astype(float)
	r_rand = Planck18.comoving_distance(random_df['redshift'].values.astype(float))*Planck18.h
	rand = treecorr.Catalog(ra = ra_rand, dec = dec_rand, r = r_rand, ra_units = 'degrees', dec_units = 'degrees', is_rand = True)

	zmin = cluster_df.ZSKY.min()
	zmax = cluster_df.ZSKY.max()
	
	for p in range(nbins_rpar):

		config = {"min_rpar": Pi[p], "max_rpar":Pi[p+1], "nbins":nbins, "brute":brute, "min_sep":5, "max_sep":80., "metric": 'Rperp'}

		dd = treecorr.NNCorrelation(config)										
		dr = treecorr.NNCorrelation(config)
		rr = treecorr.NNCorrelation(config)

		dd.process_cross(data, data)
		rr.process_cross(rand, rand)
		dr.process_cross(data, rand)

		dd.finalize()
		rr.finalize()
		dr.finalize()

		xi, varxi = dd.calculateXi(rr = rr, dr=dr)

		#if all(dd.weight) == 0:
		#	wgg_3D[p] = np.zeros(nbins)
		#else:
		#	wgg_3D[p] += xi
		wgg_3D[p] += xi
		varw_3D[p] += np.diag(dd.cov)**0.5
		DD_3D[p] += dd.weight
		DR_3D[p] += dr.weight
		#RD_3D[p] += dr.weight
		RR_3D[p] += rr.weight
		meanr_3D[p] += dd.meanr * dd.weight
		meanlogr_3D[p] += dd.meanlogr * dd.weight

	wgg = np.sum(wgg_3D * (Pi[1] - Pi[0]), axis=0)
	varw = np.sum(varw_3D, axis=0)
	DDpair = np.sum(DD_3D, axis=0)
	DRpair = np.sum(DR_3D, axis=0)
	#RDpair = np.sum(RD_3D, axis=0)
	RRpair = np.sum(RR_3D, axis=0)
	meanr = np.sum(meanr_3D, axis=0) / DDpair
	meanlogr = np.sum(meanlogr_3D, axis=0) / DDpair
	r = np.column_stack((dd.rnom, meanr, meanlogr))
	output = np.column_stack((r, wgg, varw**0.5, DDpair, DRpair, RRpair))
	if save:
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgg','noise','DDpairs','DRpairs','RDpairs','RRpairs')))

		fig, axs = plt.subplots(2, 2, figsize=(15, 15))
		data = [[DD_3D, DR_3D], [RR_3D, wgg_3D]]
		title = [["DD", "DR"], ["RR", r'$\xi_{gg}$']]
		for (i,j) in [[0,0], [0,1], [1,0], [1,1]]:
			im = axs[i, j].pcolormesh(meanr, Pi[:20], data[i][j], shading='nearest', cmap='RdPu')
			plt.colorbar(im,ax=axs[i, j])
			axs[i, j].set_title(title[i][j])
			axs[i, j].set_xlabel(r'$r_p$ [$h^{-1}$ Mpc]')
			axs[i, j].set_ylabel(r'$\Pi$ [$h^{-1}$ Mpc]')
			#axs[i, j].xlabel()
		plt.savefig(outfile+'_3D_data.png', dpi =400)
		plt.close()
	
	return output

def jackkinfe_wgg(cluster_df, random_df, min_rpar, max_rpar, nbins_rpar, nbins, npatches, outfile, save=False, brute=False):

	random_df_ = random_df
	ra_rand = random_df_['RA'].values.astype(float)
	dec_rand = random_df_['Dec'].values.astype(float)
	km = kmeans_sample(np.column_stack([ra_rand, dec_rand]), npatches, maxiter=100, tol=1.0e-5)
	random_df_['label'] = km.labels
	#rand_labels_sliced = slice_labels(rand_labels, z_rand, 200)

	cluster_df_ = cluster_df
	x = cluster_df_['XSKY'].values																						
	y = cluster_df_['YSKY'].values
	cluster_df_['label'] = km.find_nearest(np.column_stack([x, y]))
	#cluster_labels_sliced = slice_labels(cluster_labels, z, 200)

	my_cmap = plt.cm.plasma(np.arange(plt.cm.RdPu.N))
	my_cmap[:,0:3] *= 0.5 
	my_cmap = ListedColormap(my_cmap)
	cond1 = [dec_rand>-5, dec_rand<-5]
	cond2 = [y>-5, y<-5]	
	for i in range(2):
		rand_labels = random_df_.label
		cluster_labels = cluster_df_.label
		if i==0:
			plt.scatter(ra_rand[cond1[i]], dec_rand[cond1[i]], c=rand_labels[cond1[i]], s=3, cmap = my_cmap)
			plt.scatter(x[cond2[i]], y[cond2[i]], c=cluster_labels[cond2[i]], s=1, cmap = 'plasma')
		else:
			plt.scatter(np.where(ra_rand>300, ra_rand-360., ra_rand)[cond1[i]], dec_rand[cond1[i]], c=rand_labels[cond1[i]], s=3, cmap = my_cmap)
			plt.scatter(np.where(x>300, x-360., x)[cond2[i]], y[cond2[i]], c=cluster_labels[cond2[i]], s=1, cmap = 'plasma')
		plt.xlabel('RA[degrees]')
		plt.ylabel('Dec[degrees]')
		plt.savefig(outfile+'_patches_'+str(i)+'.png', dpi = 400)
		plt.close()

	plt.scatter(ra_rand, dec_rand, c=rand_labels, s=3, cmap = my_cmap)
	plt.scatter(x, y, c=cluster_labels, s=1, cmap = 'plasma')
	plt.xlabel('RA[degrees]')
	plt.ylabel('Dec[degrees]')
	plt.savefig(outfile+'_patches_.png', dpi = 400)
	plt.close()

	meanr_jack = np.empty([npatches, nbins])
	wgg_jack = np.empty([npatches, nbins])
	for i in range(npatches):
		print(i)
		random_df_temp = random_df_[random_df_.label!=i]
		cluster_df_temp = cluster_df_[cluster_df_.label!=i]
		output = compute_wgg(cluster_df_temp, random_df_temp, min_rpar, max_rpar, nbins_rpar, nbins, outfile, save, brute)
		meanr_jack[i] = output.T[2]
		wgg_jack[i] = output.T[3]
		
		#plt.scatter(meanr_jack[i], wgg_jack[i], s=1, c='black')
	#plt.show()
	#plt.close()
	
	meanr = meanr_jack.mean(axis=0)
	wgg = wgg_jack.mean(axis=0)
	var_wgg = np.diagonal(np.cov(np.transpose(wgg_jack)))*(npatches - 1)/npatches
	jackkinfe_err = var_wgg**0.5

	output = []
	with open(outfile) as file:
		for line in file:
			if not line.startswith("#"):
				output.append([text for text in line.split()])
	i = 0
	for row in output:
		row.append(jackkinfe_err[i])
		i+=1
	np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgg','noise','DDpairs','DRpairs','RDpairs','RRpairs', 'std dev')), fmt='%s')

	#fig = plt.figure()
	#ax = plt.gca()
	#ax.errorbar(x = meanr, y = wgg, yerr = var_wgg**0.5, fmt='.k', capsize = 3, ecolor = 'black', elinewidth = 1, capthick = 1, ms = 3)
	#ax.set_xscale('log')
	#ax.set_yscale('log')
	#plt.text(0.75, .9, r'z $\in$'+' ['+str(zmin)+', '+str(zmax)+']', fontsize=10, transform = ax.transAxes)
	#plt.xlabel(r'$r_p$ [$h^{-1}$ Mpc]')
	#plt.ylabel(r'$w_{gg}$($r_p$)')	
	#plt.savefig(outfile+'_jackknife.png', dpi=300)
	#plt.close()

	return None

def plot_wgg(outfile):
	output = []
	file_info = outfile.replace('[','').replace(']','').split('_')
	zmin, zmax = file_info[1].split('-')
	with open(outfile) as file:
		for line in file:
			if not line.startswith("#"):
				output.append([text for text in line.split()])
	meanr = [float(row[1]) for row in output]
	wgg = [float(row[3]) for row in output]
	err = [float(row[-1]) for row in output]

	fig = plt.figure()
	ax = plt.gca()
	ax.errorbar(x = meanr, y = wgg, yerr = err, fmt='.k', capsize = 3, ecolor = 'black', elinewidth = 1, capthick = 1, ms = 3)
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.text(0.75, .9, r'z $\in$'+' ['+str(zmin)+', '+str(zmax)+']', fontsize=10, transform = ax.transAxes)
	plt.xlabel(r'$r_p$ [$h^{-1}$ Mpc]')
	plt.ylabel(r'$w_{gg}$($r_p$)[$h^{-1}$ Mpc]')	
	plt.savefig(outfile+'.png', dpi=300)
	plt.close()
	
	return None

def plot_wgg_together():
	prefixed = [filename for filename in sorted(os.listdir('.')) if filename.startswith("wgg") and filename.endswith(']')]
	fig, axs = plt.subplots(3, 1, figsize=(5, 7), sharex=True, sharey=True)
	for j in list(range(3)):
		output = []
		file_info = prefixed[j].replace('[','').replace(']','').split('_')
		zmin, zmax = file_info[1].split('-')
		with open(prefixed[j]) as file:
			for line in file:
				if not line.startswith("#"):
					output.append([text for text in line.split()])
		meanr = [float(row[1]) for row in output]
		wgg = [float(row[3]) for row in output]
		err = [float(row[-1]) for row in output]
		ax = axs[j]
		ax.errorbar(x = meanr, y = wgg, yerr = err, fmt='.k', capsize = 3, ecolor = 'black', elinewidth = 1, capthick = 1, ms = 3)
		#ax.axhline(0, color='k', linestyle='dashed', linewidth=1)
		ax.set_xscale('log')
		ax.tick_params(axis="y", which='both', direction="in", right=True, left=True)
		ax.tick_params(axis="x", which='both', direction="in", bottom=True, top=True)
		ax.set_xticks([6, 10, 20, 40, 70], [6, 10, 20, 40, 70])
		ax.set_title(zmin+r'$<$ z $\leq$'+zmax, loc="right", y=.5, rotation=270, ha="left", va="center")
			#ax.text(-0.1, 0.5, zmin+r'<z\leq'+zmax, ha='center', va='center', rotation='vertical')

	#fig.subplots_adjust(wspace=None, hspace=None)
	#fig.text(0.5, 0.04, r'$r_p$ [$h^{-1}$ Mpc]', ha='center')
	#fig.text(0.04, 0.5, r'$w_{g+}$($r_p$) [$h^{-1}$ Mpc]', va='center', rotation='vertical')
	fig.supxlabel(r'$r_p$ [$h^{-1}$ Mpc]', y=0)
	fig.supylabel(r'$w_{gg}$($r_p$) [$h^{-1}$ Mpc]', x = 0)
	fig.tight_layout(pad=0.0)
	plt.savefig("wgg.png", bbox_inches = 'tight', dpi=300)
	plt.close()

############################################################################################################################################################

def compute_wgplus(cluster_df, random_df, ellipticity_df, min_rpar, max_rpar, nbins_rpar, nbins, outfile, save=True, brute=True):
	cluster_df_ = cluster_df
	x = cluster_df_['XSKY'].values																						
	y = cluster_df_['YSKY'].values
	r = Planck18.comoving_distance(cluster_df_['ZSKY'].values)*Planck18.h
	data1 = treecorr.Catalog(ra = x, dec = y, r = r, ra_units = 'degrees', dec_units = 'degrees')

	ellipticity_df_ = ellipticity_df
	x_e = ellipticity_df_['XSKY'].values
	y_e = ellipticity_df_['YSKY'].values
	r_e = Planck18.comoving_distance(ellipticity_df_['ZSKY'].values)*Planck18.h
	g1 = -ellipticity_df_['epsilon'].values.real
	g2 = -ellipticity_df_['epsilon'].values.imag
	w_e = ellipticity_df_['LAMBDASTAR'].values
	data2 = treecorr.Catalog(ra = x_e, dec = y_e, r = r_e, w = w_e, ra_units = 'degrees', dec_units = 'degrees', g1 = g1, g2 = g2)
	varg = treecorr.calculateVarG(data2)

	random_df_ = random_df
	ra_rand = random_df_['RA'].values.astype(float)
	dec_rand = random_df_['Dec'].values.astype(float)
	r_rand = Planck18.comoving_distance(random_df_['redshift'].values.astype(float))*Planck18.h
	rand = treecorr.Catalog(ra = ra_rand, dec = dec_rand, r = r_rand, ra_units = 'degrees', dec_units = 'degrees', is_rand = True)

	Pi = np.linspace(min_rpar, max_rpar, nbins_rpar + 1)

	gt_3D = np.zeros([nbins_rpar, nbins])
	gx_3D = np.zeros([nbins_rpar, nbins])
	varg_3D = np.zeros([nbins_rpar, nbins])
	DD_3D = np.zeros([nbins_rpar, nbins])
	DS_3D = np.zeros([nbins_rpar, nbins])
	RS_3D = np.zeros([nbins_rpar, nbins])
	meanr_3D = np.zeros([nbins_rpar, nbins])
	meanlogr_3D = np.zeros([nbins_rpar, nbins])

	for p in range(nbins_rpar):
		config = {"min_rpar": Pi[p], "max_rpar": Pi[p+1], "nbins":nbins, "brute":brute, "min_sep":5, "max_sep":80., "metric": 'Rperp'}

		ng = treecorr.NGCorrelation(config)
		rg = treecorr.NGCorrelation(config)
		ng.process_cross(data1, data2)
		rg.process_cross(rand, data2)
		ng.varxi = varg

		norm = rg.weight

		gt_3D[p] += (ng.xi - rg.xi) / norm
		gx_3D[p] += (ng.xi_im - rg.xi_im) / norm
		varg_3D[p] += (ng.varxi  + rg.varxi) / norm		
		DD_3D[p] += ng.npairs
		DS_3D[p] += ng.weight
		RS_3D[p] += rg.weight
		meanr_3D[p] += ng.meanr
		meanlogr_3D[p] += ng.meanlogr

	gt = np.sum(gt_3D * (Pi[1] - Pi[0]), axis=0)
	gx = np.sum(gx_3D * (Pi[1] - Pi[0]), axis=0)
	varg = np.sum(varg_3D, axis=0)
	DS = np.sum(DS_3D, axis=0)
	RS = np.sum(RS_3D, axis=0)
	meanr = np.sum(meanr_3D, axis=0) / DS
	meanlogr = np.sum(meanlogr_3D, axis=0) / DS
	r = np.column_stack((ng.rnom, meanr, meanlogr))
	output = np.column_stack((r, gt, gx, varg**0.5, DS, RS))

	if save:
		file_info = outfile.replace('[','').replace(']','').split('_')
		zmin, zmax = file_info[1].split('-')
		lambda_min, lambda_max = file_info[2].split('-')
		print(zmin, lambda_min)
		print("R size: ", random_df_.shape[0])
		print("C size: ", cluster_df_.shape[0])
		print("E size: ", ellipticity_df_.shape[0])
		#outfile = 'wgplus_['+str(zmin)+'-'+str(zmax)+']'+'_['+str(lambda_min)+', '+str(lambda_max)+']'
		np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgplus','wgcross','noise','DSpairs','RSpairs')))
	
		fig, axs = plt.subplots(2, 2, figsize=(15, 15))
		data = [[DD_3D, DS_3D], [RS_3D, gt_3D]]
		title = [[r"$D_sD_d$", "weighted DD"], [r"$D_sR$", r'$\xi_{g+}$']]
		for (i,j) in [[0,0], [0,1], [1,0], [1,1]]:
			im = axs[i, j].pcolormesh(meanr, Pi[:nbins_rpar], data[i][j], shading='nearest', cmap='RdPu')
			plt.colorbar(im,ax=axs[i, j])
			axs[i, j].set_title(title[i][j])
			axs[i, j].set_xlabel(r'$r_p$ [$h^{-1}$ Mpc]')
			axs[i, j].set_ylabel(r'$\Pi$ [$h^{-1}$ Mpc]')
		plt.savefig(outfile+'_3D_data.png', dpi = 400)
		plt.close()

	return output

def jackkinfe_wgplus(cluster_df, random_df, ellipticity_df, min_rpar, max_rpar, nbins_rpar, nbins, npatches, outfile, save=False, brute=False):
	random_df_ = random_df
	ra_rand = random_df_['RA'].values.astype(float)
	dec_rand = random_df_['Dec'].values.astype(float)
	km = kmeans_sample(np.column_stack([ra_rand, dec_rand]), npatches, maxiter=100, tol=1.0e-5)
	random_df_['label'] = km.labels

	cluster_df_ = cluster_df
	x = cluster_df_['XSKY'].values																						
	y = cluster_df_['YSKY'].values
	cluster_df_['label'] = km.find_nearest(np.column_stack([x, y]))
	
	ellipticity_df_ = ellipticity_df
	x_e = ellipticity_df_['XSKY'].values
	y_e = ellipticity_df_['YSKY'].values
	ellipticity_df_['label'] = km.find_nearest(np.column_stack([x_e, y_e]))

	#meanr_jack = np.empty([npatches, nbins])
	wgg_jack = np.empty([npatches, nbins])
	for i in range(npatches):
		print(i)
		random_df_temp = random_df_[random_df_.label!=i]
		cluster_df_temp = cluster_df_[cluster_df_.label!=i]
		ellipticity_df_temp = ellipticity_df_[ellipticity_df_.label!=i]
		output = compute_wgplus(cluster_df_temp, random_df_temp, ellipticity_df_temp, min_rpar, max_rpar, nbins_rpar, nbins, outfile, save, brute)
		#meanr_jack[i] = output.T[2]
		wgg_jack[i] = output.T[3]
	wgg = wgg_jack.mean(axis=0)
	var_wgg = np.diagonal(np.cov(np.transpose(wgg_jack)))*(npatches - 1)/npatches
	jackkinfe_err = var_wgg**0.5

	output = []
	with open(outfile) as file:
		for line in file:
			if not line.startswith("#"):
				output.append([text for text in line.split()])
	i = 0
	for row in output:
		row.append(str(jackkinfe_err[i]))
		i+=1
	#output = np.array(output)
	np.savetxt(outfile, output, header='\t'.join(('rnom','meanr','meanlogr','wgplus','wgcross','noise','DSpairs','RSpairs', 'error')), fmt='%s')

	return None

def plot_wgplus(outfile):
	output = []
	file_info = outfile.replace('[','').replace(']','').split('_')
	zmin, zmax = file_info[1].split('-')
	lambda_min, lambda_max = file_info[2].split('-')	
	with open(outfile) as file:
		for line in file:
			if not line.startswith("#"):
				output.append([text for text in line.split()])
	meanr = [float(row[1]) for row in output]
	wgplus = [float(row[3]) for row in output]
	err = [float(row[-1]) for row in output]

	fig = plt.figure()
	ax = plt.gca()
	#print(gt)
	ax.errorbar(x = meanr, y = wgplus, yerr = err, fmt='.k', capsize = 3, ecolor = 'black', elinewidth = 1, capthick = 1, ms = 3)
	ax.set_xscale('log')
	#ax.set_yscale('log')
	plt.text(0.70, 0.01, r'z $\in$'+'['+str(zmin)+', '+str(zmax)+']\n'+r'$\lambda^{*} \in $ ['+str(lambda_min)+', '+str(lambda_max)+']', fontsize=10, transform = ax.transAxes)
	plt.xlabel(r'$r_p$ [$h^{-1}$ Mpc]')
	plt.ylabel(r'$w_{g+}$($r_p$) [$h^{-1}$ Mpc]')
	plt.savefig(outfile+'.png', dpi=300)
	#plt.show()
	plt.close()

def plot_wgplus_together():
	prefixed = [filename for filename in sorted(os.listdir('.')) if filename.startswith("wgplus") and filename.endswith(']')]
	fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
	for i in list(range(3)):
		for j in list(range(3)):
			output = []
			file_info = prefixed[i+j].replace('[','').replace(']','').split('_')
			zmin, zmax = file_info[1].split('-')
			lambda_min, lambda_max = file_info[2].split('-')
			with open(prefixed[i+j]) as file:
				for line in file:
					if not line.startswith("#"):
						output.append([text for text in line.split()])
			meanr = [float(row[1]) for row in output]
			wgplus = [float(row[3]) for row in output]
			wgplustimesr = [float(row[1])*float(row[3]) for row in output]
			err = [float(row[-1]) for row in output]
			ax = axs[i][j]
			ax.errorbar(x = meanr, y = wgplus, yerr = err, fmt='.k', capsize = 3, ecolor = 'black', elinewidth = 1, capthick = 1, ms = 3)
			ax.axhline(0, color='k', linestyle='dashed', linewidth=1)
			ax.set_xscale('log')
			ax.tick_params(axis="y", which='both', direction="in", right=True, left=True)
			ax.tick_params(axis="x", which='both', direction="in", bottom=True, top=True)
			ax.set_xticks([6, 10, 20, 40, 70], [6, 10, 20, 40, 70])
			if i==0:
				ax.set_title(lambda_min+ r'$< \lambda \leq$'+lambda_max)
			if j==2:
				ax.set_title(zmin+r'$<$ z $\leq$'+zmax, loc="right", y=.5, rotation=270, ha="left", va="center")
				#ax.text(-0.1, 0.5, zmin+r'<z\leq'+zmax, ha='center', va='center', rotation='vertical')

	#fig.subplots_adjust(wspace=None, hspace=None)
	#fig.text(0.5, 0.04, r'$r_p$ [$h^{-1}$ Mpc]', ha='center')
	#fig.text(0.04, 0.5, r'$w_{g+}$($r_p$) [$h^{-1}$ Mpc]', va='center', rotation='vertical')
	fig.supxlabel(r'$r_p$ [$h^{-1}$ Mpc]', y=0)
	fig.supylabel(r'$w_{g+}$($r_p$) [$h^{-1}$ Mpc]', x = 0)
	fig.tight_layout(pad=0.0)
	plt.savefig("wgplus.png", bbox_inches = 'tight', dpi=300)
	plt.close()

############################################################################################################################################################

#cluster_df_lowz = cluster_df[cluster_df.ZSKY<=0.6]
#cluster_df_split = np.array_split(cluster_df_lowz[cluster_df.LAMBDASTAR>=15].sort_values(by = 'ZSKY'), 3)
#ellipticity_df_lowz = ellipticity_df[ellipticity_df.ZSKY<=0.6]
ellipticity_df_split = np.array_split(ellipticity_df[ellipticity_df.LAMBDASTAR >= 15].sort_values(by = 'LAMBDASTAR'), 3)
ellipticity_df_z_split = np.array_split(ellipticity_df[ellipticity_df.LAMBDASTAR >= 15].sort_values(by = 'ZSKY'), 3)
rpar_lim = [100, 125, 150]

for i in range(3):
	z_min = ellipticity_df_z_split[i].ZSKY.min()
	z_max = ellipticity_df_z_split[i].ZSKY.max()
	cluster_df_subset = cluster_df[cluster_df.LAMBDASTAR>=15][cluster_df.ZSKY.between(z_min, z_max)]
	random_df_subset = random_df[random_df.redshift.between(z_min, z_max)]
	#clustering_signal(cluster_df_subset, random_df_subset, 10, 15)

	outfile = outfile = 'wgg_['+str(round(z_min, 2))+'-'+str(round(z_max, 2))+']'
	#parameters = {'cluster_df': cluster_df, 'random_df':random_df, 'zmin': zmin[i], 'zmax':zmax[i], 'min_rpar':-150, 'max_rpar':150, 'nbins_rpar':20, 'nbins':10}
	#compute_wgg(cluster_df_subset, random_df_subset, -rpar_lim[i], rpar_lim[i], 20, 10, outfile, save=True, brute=False)
	#jackkinfe_wgg(cluster_df_subset, random_df_subset, -rpar_lim[i], rpar_lim[i], 20, 10, 15, outfile, save=False, brute=False)
	#plot_wgg(outfile)

	for j in range(3):
		lambda_min = ellipticity_df_split[j]["LAMBDASTAR"].min()
		lambda_max = ellipticity_df_split[j]["LAMBDASTAR"].max()
		ellipticity_df_subset = ellipticity_df_split[j][ellipticity_df_split[j]['ZSKY'].between(z_min, z_max)]
		outfile = 'wgplus_['+str(round(z_min, 2))+'-'+str(round(z_max, 2))+']'+'_['+str(round(lambda_min, 2))+'-'+str(round(lambda_max, 2))+']'
		#compute_wgplus(cluster_df_subset, random_df_subset, ellipticity_df_subset, -rpar_lim[i], rpar_lim[i], 20, 10, outfile, brute=False)
		#if i==2:
			#if j==0:
				#print("skipping error calculation")
			#else:
				#jackkinfe_wgplus(cluster_df_split[i], random_df_subset, ellipticity_df_subset, z_min, z_max, -rpar_lim[i], rpar_lim[i], 20, 10, 35, outfile)
		#else:
		#jackkinfe_wgplus(cluster_df_subset, random_df_subset, ellipticity_df_subset, -rpar_lim[i], rpar_lim[i], 20, 10, 15, outfile)
		#plot_wgplus(outfile)

#plot_wgg_together()
plot_wgplus_together()