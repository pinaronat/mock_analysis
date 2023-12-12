#!/usr/bin/env python3

import argparse
import pathlib
import pandas as pd
from skbio.stats.ordination import pcoa
from skbio.diversity import beta_diversity
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def parse_arguments():
	parser = argparse.ArgumentParser(prog = "run_analyis.py", description = "Apply PCOA and K-Medoid Clustering with Bray-Curtis or Jaccard distance.")

	parser.add_argument("-d", "--dist", dest = "distance", choices = ["bray-curtis", "jaccard"], 
						help = "distance metric to apply PCOA on")
	parser.add_argument("-i", "--inpath", dest = "input_path", type = pathlib.Path, 
						default = pathlib.Path().absolute(), 
						help = "path where input files will be found -- default: current directory")
	parser.add_argument("-f", "--infile", dest = "input_file",
						help = "input filename")
	parser.add_argument("-o", "--outpath", dest = "output_path", type = pathlib.Path, 
						default = pathlib.Path().absolute(), 
						help = "path where output files will be saved to -- default: current directory")

	args = parser.parse_args()

	return args

def main(args):
	input_path = args.input_path.absolute()
	input_filename = args.input_file
	output_path = args.output_path
	distance = args.distance

	df = pd.read_csv(input_path / input_filename) # get input file as df


	### DATA CLEANUP
	df = df.drop("Taxonomic Rank", axis = 1) # remove the column with Taxonomic Rank info
	df = df.transpose() # transpose to have samples as columns
	df.columns = df.iloc[0,:].values # set column names with sample IDs
	df = df.drop(index = "Organism").apply(pd.to_numeric) # remove the original column with sample IDs and cast as numeric


	### PCOA & SCREE-PLOT
	# create distance matrices based on the selected distance metric, and apply PCOA
	if distance == "bray-curtis":
		id_list = df.index
		dist = beta_diversity(counts = df.values, ids = id_list, metric = "braycurtis") # calculate bray curtis distance
		
	elif distance == "jaccard":
		dist_array = pairwise_distances(df.to_numpy(), metric = "jaccard")
		dist = pd.DataFrame(dist_array, index = df.index, columns = df.index)
			
	pcoa_obj = pcoa(dist) # apply pcoa to distance matrix/df

	# create and save scree plot
	fig = plt.figure(figsize = (15, 7))	
	explained_vars = pcoa_obj.eigvals/sum(pcoa_obj.eigvals)*100
	barplot = plt.bar(x = explained_vars.index, height = explained_vars, width = 1.2, alpha = 0.6)

	for rect in barplot:
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width()/2.0, height, f"{height:.0f}", ha = "center", va = "bottom")

	plt.xlabel("Coordinates")
	plt.ylabel("Explained Variance")
	plt.title("Scree Plot".format(distance))
	plt.savefig(output_path / "scree_plot_{}.png".format(distance))


	### K-MEDOID CLUSTERING
	# choose k based on sillhouette score
	silhou = []
	k_lst = [2, 3, 4, 5, 6, 7, 8] 
	for k in k_lst: # for each possible k
		model = KMedoids(n_clusters = k, method = "pam", random_state = 1) # create model with a fixed random state
		pam_model = model.fit(pcoa_obj.samples) # fit the model to pcoa applied samples 
		silhou.append(silhouette_score(pcoa_obj.samples, pam_model.labels_)) # append silhouette score to initialized list

	# create line plot to visualize sillhouette scores
	fig = plt.figure(figsize = (9, 6))
	plt.plot(k_lst, silhou, c = "violet") # plot silhouette scores as a line plot
	plt.title("Sillhouette Score Line Plot") 
	plt.xlabel("Cluster Number (k)")
	plt.ylabel("Silhouette Score")
	plt.savefig(output_path / "sillhouette_plot_{}.png".format(distance), dpi = 200, bbox_inches = "tight")

	# select k with maximum sillhouette score
	max_sillhouette = max(silhou) 
	max_ind = silhou.index(max_sillhouette) # get the index of the max
	k = k_lst[max_ind] # get k value

	# create model with the selected k value
	model = KMedoids(n_clusters = k, method = "pam", random_state = 1)
	pam_model = model.fit(pcoa_obj.samples) # fit the model

	pred_y = model.predict(pcoa_obj.samples) # get predicted y values

	colors_main = ["skyblue", "lightcoral", "violet", "darkgreen", "darkred"] # create a colors array
	colors = []
	for i in range(0,len(pred_y)): # for each value in predictions (y)
		colors.append(colors_main[pred_y[i]]) # append a color name from the main list

	fig = plt.figure(figsize = (9,6))
	ax = fig.add_subplot()

	pcoa_df = pcoa_obj.samples
	Xax = pcoa_df.PC1
	Yax = pcoa_df.PC2
	labels = pcoa_df.index

	for line in range(0, df.shape[0]):
		ax.text(Xax.iloc[line], Yax.iloc[line], s = labels[line], size = 6)

	ax.scatter(Xax, Yax, c = colors)
	plt.title ("Scatter Plot - Clusters")
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.savefig(output_path / "cluster_scatter_plot_{}.png".format(distance), dpi = 200, bbox_inches = "tight")


if __name__ == "__main__":
	args = parse_arguments()
	main(args)