from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.style as style

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.style as style


from sklearn.datasets import load_digits
from sklearn.manifold import MDS


def silhouette_PlotAvgOrSample(dataset,X, n_clusters, cluster_labels, centers, fr,true_labels=np.array([]) ,alg='KMeans',dimensions=(0,1),measureOndim=False, palette='tab10'):

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    cluster_labels = np.array(cluster_labels)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    if not measureOndim:
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    else:
        sample_silhouette_values = silhouette_samples(X[:,[dimensions[0],dimensions[1]]], cluster_labels)
        
    y_lower = 0
    cmap = plt.cm.get_cmap('jet',n_clusters)
    for i in np.unique(cluster_labels):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        # import pdb; pdb.set_trace()
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        if true_labels.size > 0:
            ith_smaple_groundTruth = true_labels[cluster_labels == i]
            df = pd.DataFrame({'ith_cluster_silhouette_values':ith_cluster_silhouette_values ,'ith_smaple_groundTruth':ith_smaple_groundTruth})
            df = df.sort_values(by=['ith_cluster_silhouette_values'])
            ith_cluster_silhouette_values = df['ith_cluster_silhouette_values'].values
            ith_smaple_groundTruth_color =  [cmap(i) for i in df['ith_smaple_groundTruth'].values.astype(np.int64)]
        else:
            ith_cluster_silhouette_values.sort()
            ith_smaple_groundTruth_color  = cm.nipy_spectral(float(i) / n_clusters)

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper+1),
                          np.ones(size_cluster_i+1)*(-1), np.ones(size_cluster_i+1),
                          facecolor=color, edgecolor=color, alpha=0.5)
        for yy, xx, cc in zip(np.arange(y_lower, y_upper),ith_cluster_silhouette_values, ith_smaple_groundTruth_color):
            ax1.fill_betweenx([yy,yy+1],0,[xx,xx],facecolor=cc, edgecolor='k')

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
    ax2.scatter(X[:, dimensions[0]], X[:, dimensions[1]], marker='8', s=200, lw=0, alpha=1,
                c=colors, edgecolor='k')
    ax2.scatter(X[:, dimensions[0]], X[:, dimensions[1]], marker='X', s=100, lw=0, alpha=1,
                c=[cmap(i) for i in true_labels.astype(np.int64)], edgecolor='k')

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, dimensions[0]], centers[:, dimensions[1]], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[dimensions[0]], c[dimensions[1]], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(f"Feature space for the {dimensions[0]} feature")
    ax2.set_ylabel(f"Feature space for the {dimensions[1]} feature")

    plt.suptitle(f"Silhouette analysis for {alg} clustering on sample data {dataset} "
                  f"with n_clusters = {n_clusters} with ConstFr {fr}",
                 fontsize=14, fontweight='bold')
