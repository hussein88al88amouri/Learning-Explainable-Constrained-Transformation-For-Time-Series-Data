#%%
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
from Utilities import *
matplotlib.rcParams['figure.figsize'] = (10, 10)
import scipy as sp
import pylab
def QMAHistplot(QMunderstudyName,QMunderstudy,scopetoplot,dataset=None, outdir=None,figsize=(7,5), lw=5):
     
    if 'Local' in QMunderstudyName and 'Local' in scopetoplot:
        # fig, axs = plt.subplots(2, np.unique(QMunderstudy.Cluster).shape[0], sharex=True, sharey=True)
        fig1 = plt.figure( figsize=figsize, tight_layout=True)
        fig2 = plt.figure(figsize=figsize, tight_layout=True)
        k = 1
        ax1, ax2 = {}, {}
        for g in QMunderstudy.groupby(['Cluster']):
            sortedIG = g[1].sort_values(by='IG', ascending=False) if "Single" in QMunderstudyName else QMunderstudy
            CsIG = sortedIG['IG'].cumsum()
            CsIG = CsIG/CsIG.max()
            ax1[f'1{k}'] = fig1.add_subplot(f'1{np.unique(QMunderstudy.Cluster).shape[0]}{k}', autoscale_on=True)#, sharey=True)
            ax1[f'1{k}'].plot(np.arange(CsIG.values.shape[0]), CsIG.values, linewidth=lw)
            # ax1[f'1{k}'].set_xticks([])
            ax1[f'1{k}'].set_ylabel('Normalized Cumulative IG')
            ax1[f'1{k}'].set_title(f'Cluster  {int(g[1]["Cluster"].iloc[0])}')
            ax1[f'1{k}'].set_xlabel("number of shapelets")
            # ax1[f'1{k}'].set_suptitle(QMunderstudyName)
            ax1[f'1{k}'].set_ylim(0,1.1)
            
            ax2[f'2{k}'] = fig2.add_subplot(f'1{np.unique(QMunderstudy.Cluster).shape[0]}{k}', autoscale_on=True)#, sharey=True)
            ax2[f'2{k}'].hist(np.arange(CsIG.values.shape[0]), weights=sortedIG['IG'],bins=CsIG.values.shape[0], label=g[1]['Cluster'].iloc[0])
            ax2[f'2{k}'].set_xticks([])
            ax2[f'2{k}'].set_ylabel('IG')
            ax2[f'2{k}'].set_title(f'Cluster  {int(g[1]["Cluster"].iloc[0])}')
            ax2[f'2{k}'].set_xlabel("number of shapelets")
            # ax2[f'2{k}'].set_suptitle(QMunderstudyName)
        
            k += 1
        fig1.subplots_adjust(wspace=0, hspace=0.05)
        fig2.subplots_adjust(wspace=0, hspace=0.05)
        if outdir:
            fig1.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_CE_histplot_CIG.pdf', bbox_inches='tight')        
            fig2.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_CE_histplot_IG.pdf', bbox_inches='tight')        
            plt.close('all')
    if 'Global' in QMunderstudyName and 'Global' in scopetoplot:
        # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        fig1 = plt.figure(num=1, figsize=figsize, tight_layout=True)
        fig2 = plt.figure(num=2, figsize=figsize, tight_layout=True)
        ax1 = fig1.add_subplot(111, autoscale_on=True)
        ax2 = fig2.add_subplot(111, autoscale_on=True)
        sortedIG = QMunderstudy.sort_values(by='IGMulti', ascending=False) if "Single" in QMunderstudyName else QMunderstudy
        CsIG = sortedIG['IGMulti'].cumsum()
        CsIG = CsIG/CsIG.max()
        ax1.plot(np.arange(CsIG.values.shape[0]), CsIG.values, linewidth=lw)
        # ax1.set_xticks([])
        # ax1.set_xlabel("Shapelets")
        ax1.set_xlabel("number of shapelets")
        ax1.set_ylabel('Normalized Cumulative IG')
        ax1.set_ylim(0,1.1)
        ax2.hist(np.arange(CsIG.values.shape[0]), weights=sortedIG['IGMulti'],bins=CsIG.values.shape[0])
        # ax2.set_xticks([])
        ax2.set_xlabel("number of shapelets")
        ax2.set_ylabel('IG')
        fig1.subplots_adjust(wspace=0, hspace=0.05)
        fig2.subplots_adjust(wspace=0, hspace=0.05)
        if outdir:
            fig1.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_histplot_CIG.pdf', bbox_inches='tight')        
            fig2.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_histplot_IG.pdf', bbox_inches='tight')        
            plt.close('all')
        return CsIG
def QMShapeletsplots(QMunderstudyName,QMunderstudy,scopetoplot,notallcluster,clustertoplot,shapeletstoplot,shapelets_df):
    if 'Local' in QMunderstudyName and scopetoplot in QMunderstudyName:
        for g in QMunderstudy.groupby(['Cluster']):
            sortedIG = g[1].sort_values(by='IG', ascending=False) if 'Single' in QMunderstudyName else g[1] 
            if notallcluster:
                if int(sortedIG.Cluster.iloc[0]) != clustertoplot:
                    continue
            fig, axs = plt.subplots()
            plt.axis("off")
            plt.suptitle(f'{QMunderstudyName} Cluster {int(sortedIG.Cluster.iloc[0])}')
            shapeletidx = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
            shapeletsplots = shapelets_df[shapeletidx]
            ig = sortedIG.IG.to_numpy()[:shapeletstoplot]
            rowcolunm, _= numSubplots(shapeletsplots.shape[0])
            k = 1
            for i, j, q in zip(shapeletsplots, ig, shapeletidx):
                fig.add_subplot(rowcolunm[0], rowcolunm[1], k, frameon=True)
                plt.plot(i/np.max(np.abs(i)))
                plt.title(f"{np.round(j,3)},idx {q}") if 'Single' in QMunderstudyName else plt.title(f"{np.round(j,3)},+idx {q}" if k != 1 else f"{np.round(j,3)},idx {q}")
                k += 1
    if 'Global' in  QMunderstudyName and scopetoplot in QMunderstudyName:
        sortedIG = QMunderstudy.sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
        fig, axs = plt.subplots()
        plt.axis("off")
        plt.suptitle(f'{QMunderstudyName}')
        shapeletidx = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
        shapeletsplots = shapelets_df[shapeletidx]
        ig = sortedIG.IGMulti.to_numpy()[:shapeletstoplot]
        rowcolunm, _= numSubplots(shapeletsplots.shape[0])
        k = 1
        for i, j, q in zip(shapeletsplots, ig, shapeletidx):
            fig.add_subplot(rowcolunm[0], rowcolunm[1], k, frameon=True)
            plt.plot(i/np.max(np.abs(i)))
            plt.title(f"{np.round(j,3)},idx {q}") if 'Single' in QMunderstudyName else plt.title(f"{np.round(j,3)},+idx {q}" if k != 1 else f"{np.round(j,3)},idx {q}")
            k += 1

def QMDistTSCplot(QMunderstudyName,QMunderstudy,notallqm,qmtoplot,scopetoplot,plotSinglebycluster,clustertoplot,shapeletstoplot,nmi,ari,embTslearnNUmpy,CSKlabels,labels,n_clusters,dataset):
    for g in QMunderstudy.groupby(['Cluster']) if "Local" in QMunderstudyName  else [[None, QMunderstudy]]:
        if plotSinglebycluster and "Local" in QMunderstudyName:
            if int(g[1].Cluster.iloc[0]) != clustertoplot:
                continue
        if "SingleGlobal" in QMunderstudyName:
            shapelets_ = g[1].sort_values(
                by='IGMulti', ascending=False).Shapelet.iloc[:shapeletstoplot].to_numpy(dtype=np.int32)
        elif "SingleLocal" in QMunderstudyName:
            shapelets_ = g[1].sort_values(
                by='IG', ascending=False).Shapelet.iloc[:shapeletstoplot].to_numpy(dtype=np.int32)
        else:
            shapelets_ = g[1].iloc[shapeletstoplot].Shapelet
        plt.figure()
        plt.suptitle(f'{QMunderstudyName} for Cluster {int(g[1].Cluster.iloc[0])}' if "Local" in QMunderstudyName else QMunderstudyName)
        plott = sortdata(embTslearnNUmpy.T[shapelets_].T, labels)
        plt.imshow(plott[0], aspect='auto', extent=[0, shapelets_.shape[0],
                   0, plott[0].shape[0]], interpolation='nearest', cmap='hot')
        ax = plt.gca()
        ax.set_xticks(np.arange(0, shapelets_.shape[0], 1))
    
        ax.set_xticks(np.arange(0, shapelets_.shape[0], 0.5), minor=True)
        plt.grid(which='major', axis="x", color='white',
                 linestyle='-', linewidth=1, alpha=0.8)
        plt.tick_params(axis='x', which='minor')
        ax.set_xticklabels(list(shapelets_), minor=False, rotation=-45, ha="left")
        clustersMemCount = np.unique(plott[1], return_counts=True)
        ip = clustersMemCount[1][0]
        for i in clustersMemCount[1][1:]:
            plt.axhline(y=plott[0].shape[0]-ip-1,
                        color='g', linestyle='-', alpha=1)
            ip += i
        for j, pos in zip(ax.get_xticks(minor=True), np.array(ax.get_xticks(minor=True)).astype(np.int32)):
            ip = clustersMemCount[1][0]
            for i, C_current in zip(clustersMemCount[1], clustersMemCount[0]):
                Cdistm = np.round(
                    np.mean(plott[0][plott[1] == C_current].T[pos]), decimals=3)
                Cdiststd = np.round(
                    np.std(plott[0][plott[1] == C_current].T[pos]), decimals=3)
                ax.text(j-0.3, plott[0].shape[0]-ip+i/2, str(Cdistm)+u"\u00B1"+str(
                    Cdiststd), style='italic', bbox={'facecolor': 'cyan', 'alpha': 1, 'pad': 10})
                ip += i
        ax.set_yticklabels([])
        plt.colorbar(cmap='hot')
        plott_sortedpred = sortdata(embTslearnNUmpy.T[shapelets_].T, CSKlabels)
        clustersMemCountpred = np.unique(plott_sortedpred[1], return_counts=True)
        ip_pred = clustersMemCountpred[1][0]
        fig, axs = plt.subplots(shapelets_.shape[0], 1, sharex=True)
        for shp, shpnum in zip(range(shapelets_.shape[0]), shapelets_):
            axs[shp].plot(plott_sortedpred[0].T[shp]) if str(type(
                axs)) != "<class 'matplotlib.axes._subplots.AxesSubplot'>" else axs.plot(plott[0].T[shp])
            ip = clustersMemCount[1][0]
            ip_pred = clustersMemCountpred[1][0]
            for i, j in zip(clustersMemCount[1][1:], clustersMemCountpred[1][1:]):
                axs[shp].axvline(x=plott[0].shape[0]-ip-1, color='g', linestyle='-', alpha=1) if str(type(
                    axs)) != "<class 'matplotlib.axes._subplots.AxesSubplot'>" else axs.axvline(x=plott[0].shape[0]-ip-1, color='g', linestyle='-', alpha=1)
                ip += i
                axs[shp].axvline(x=plott_sortedpred[0].shape[0]-ip_pred-1, color='k', linestyle='--', alpha=1) if str(type(
                    axs)) != "<class 'matplotlib.axes._subplots.AxesSubplot'>" else axs.axvline(x=plott_sortedpred[0].shape[0]-ip_pred-1, color='g', linestyle='-', alpha=1)
                ip_pred += j
            axs[shp].set_title(f'{shpnum}') if str(type(
                axs)) != "<class 'matplotlib.axes._subplots.AxesSubplot'>" else axs.set_title(f'{shpnum}')
            plt.suptitle(f'{QMunderstudyName} {int(g[1].Cluster.iloc[0])}' if "Local" in QMunderstudyName else QMunderstudyName)
        _, nmi[QMunderstudyName], ari[QMunderstudyName] = KMeanClustering(
            embTslearnNUmpy.T[shapelets_].T, n_clusters, labels)
        FeatureSpaceUMAP( embTslearnNUmpy.T[shapelets_].T, dataset, labels, "ALl data", dirsave=None, description=f'{QMunderstudyName} for Cluster {int(g[1].Cluster.iloc[0])}' if "Local" in QMunderstudyName else QMunderstudyName, ml=None, cl=None,random_state=10)
        print(
            f'{QMunderstudyName}: nmi {np.round(np.mean(nmi[QMunderstudyName]),decimals=2)}  ari {np.round(np.mean(ari[QMunderstudyName]),decimals=2)} \nShapelets{shapelets_}')
        
def QMShapeletDistPerClusterPlot(QMunderstudyName,QMunderstudy,scopetoplot,notallcluster,clustertoplot,shapeletstoplot,embTslearnNUmpy,CSKlabels,labels,shapelets_df,dataset):
    if 'Global' in  QMunderstudyName and 'Global' in scopetoplot :
            sortedIG = QMunderstudy.sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
            nbcluster = np.unique(CSKlabels).shape[0]
            rows, colunms = shapeletstoplot, 1 + nbcluster
            fig, axs = plt.subplots(rows, colunms, tight_layout=True)
            shapeletidx = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
            shapeletsplots = shapelets_df[shapeletidx]
            ig = sortedIG.IGMulti.to_numpy()[:shapeletstoplot]
            a = pd.concat([pd.DataFrame(data = embTslearnNUmpy,columns=range(embTslearnNUmpy.shape[1])),pd.DataFrame({"Predlables":CSKlabels,"Truelabels":labels})],axis=1)
            a.reset_index(level=0, inplace=True)
            colors = dict(zip(np.unique(labels),sns.color_palette("Set2", np.unique(labels).shape[0])))
            for r in range(rows):
                axs[r][0].plot(shapeletsplots[r])
                axs[r][0].set_title(f'Shp {shapeletidx[r]}, IG {np.round(ig[r],2)}')
                for i in a.groupby('Predlables'):
                    sns.scatterplot(data=i[1],x=i[1].index,y=shapeletidx[r],hue='Truelabels',palette=colors,ax=axs[r][i[0]+1])
                    axs[r][i[0]+1].set_ylim(np.min(a[shapeletidx[r]][1:])-0.01,np.max(a[shapeletidx[r]][1:])+0.01)
                    # axs[r][i[0]+1].set_xticklabels([])
                    axs[r][i[0]+1].set_xlabel(f'Clustersize:{i[1].shape[0]}')
                    axs[r][i[0]+1].set_ylabel(None)    
                    axs[r][i[0]+1].set_title(f'Cluster {i[0]}')    
                    axs[r][i[0]+1].legend(title='TrueLabels', fontsize='8', title_fontsize='8',markerscale=0.5)
                    if i[0] != 0:
                        # pass
                        axs[r][i[0]+1].set_yticklabels([])            
            plt.suptitle(f'{QMunderstudyName} on {dataset}')
    if 'Local' in  QMunderstudyName and 'Local' in scopetoplot :
            for g in QMunderstudy.groupby(['Cluster']):
                sortedIG = g[1].sort_values(by='IG', ascending=False) if 'Single' in QMunderstudyName else g[1] 
                if notallcluster:
                    if int(sortedIG.Cluster.iloc[0]) != clustertoplot:
                        continue
                nbcluster = np.unique(CSKlabels).shape[0]
                rows, colunms = shapeletstoplot, 1 + nbcluster
                fig, axs = plt.subplots(rows, colunms, tight_layout=True)
                shapeletidx = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
                shapeletsplots = shapelets_df[shapeletidx]
                ig = sortedIG.IG.to_numpy()[:shapeletstoplot]
                a = pd.concat([pd.DataFrame(data = embTslearnNUmpy,columns=range(embTslearnNUmpy.shape[1])),pd.DataFrame({"Predlables":CSKlabels,"Truelabels":labels})],axis=1)
                a.reset_index(level=0, inplace=True)
                colors = dict(zip(np.unique(labels),sns.color_palette("Set2", np.unique(labels).shape[0])))
                for r in range(rows):
                    axs[r][0].plot(shapeletsplots[r])
                    axs[r][0].set_title(f'Shp {shapeletidx[r]}, IG {np.round(ig[r],2)}')
                    for i in a.groupby('Predlables'):
                        sns.scatterplot(data=i[1],x=i[1].index,y=shapeletidx[r],hue='Truelabels',palette=colors,ax=axs[r][i[0]+1])
                        axs[r][i[0]+1].set_ylim(np.min(a[shapeletidx[r]][1:])-0.01,np.max(a[shapeletidx[r]][1:])+0.01)
                        # axs[r][i[0]+1].set_xticklabels([])
                        axs[r][i[0]+1].set_xlabel(f'Clustersize:{i[1].shape[0]}')
                        axs[r][i[0]+1].set_ylabel(None)    
                        axs[r][i[0]+1].set_title(f'Cluster {i[0]}')    
                        if i[0] != 0:
                            # pass
                            axs[r][i[0]+1].set_yticklabels([])            
                plt.suptitle(f'{QMunderstudyName} for Cluster {int(g[1].Cluster.iloc[0])} on {dataset}')

def plt_Cscore_vs_Rshapelets(shapelets, embeddings,labels):
    """
    shapelets, dataframe containing shapelet idx and IG
    labels, predictions
    """
    shapeletsidx = shapelets.Shapelet
    CSKnmis, CSKaris = {}, {}
    for i in tqdm(range(shapeletsidx.shape[0])):
        ii = shapeletsidx[i]
        _, CSKnmis[i], CSKaris[i] = KMeanClustering(embeddings[:,ii], n_classes(labels), labels, verbos=False)    
        CSKnmis[i] = np.round(np.mean(CSKnmis[i]), decimals=2)
        CSKaris[i] = np.round(np.mean(CSKaris[i]), decimals=2)
    fig, axs = plt.subplots(tight_layout=True)
    IGsorted = shapelets[shapelets.columns[2]]
    a = [pd.Series({'d':i,'IG':IGsorted.iloc[i],'nmi':CSKnmis[i]}) for i in range(len(IGsorted))]
    df = pd.DataFrame(columns={'d','IG','nmi'})
    df = df.append(a)
    sns.scatterplot(data=df,x='d', y='nmi', size='IG')
    axs.set_xlabel('Sorted IG')
    axs.set_xlabel('Cluster score')


def QMShapeletOrderline(QMunderstudyName, QMunderstudy, scopetoplot, notallcluster, clustertoplot, shapeletstoplot, embTslearnNUmpy, CSKlabels, labels, shapelets_df, dataset,outdir=None,figsize=((8, 7)), lw=7, lp=7, s=100):
    listmarkers = ["o", "d", "X", "^", "D", "v", "<", ">"]
    colorsl = dict(zip(np.unique(labels), sns.color_palette("copper", np.unique(labels).shape[0]).as_hex()))
    colorsp = dict(zip(np.unique(CSKlabels), sns.color_palette("tab10", np.unique(CSKlabels).shape[0]).as_hex()))
    markersl = dict(zip(np.unique(labels), [listmarkers[i] for i in range(np.unique(labels).shape[0])]))
    px, py, pz, s2d, rd3 = 8, 8, 2, 280, 1.5
    if 'Global' in QMunderstudyName and 'Global' in scopetoplot:
        sortedIG = QMunderstudy.sort_values(by='IGMulti', ascending=False) if 'Single' in QMunderstudyName else QMunderstudy
        ig = sortedIG.IGMulti.to_numpy()[:shapeletstoplot]
        shapeletidx = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
        shapeletsplots = shapelets_df[shapeletidx]
        dataToplot = pd.concat([pd.DataFrame(data=embTslearnNUmpy, columns=range(embTslearnNUmpy.shape[1])), pd.DataFrame({"Predlabels": CSKlabels, "Truelabels": labels})], axis=1)
        dataToplot.reset_index(level=0, inplace=True)
        axs = {}
        for c in range(1, shapeletstoplot+1):
            r = c-1
            fig = plt.figure(num=int(f'1{c}'), figsize=figsize, tight_layout=True)  
            axs[f'1{c}'] = fig.add_subplot(111, autoscale_on=True)
            axs[f'1{c}'].plot(shapeletsplots[r], linewidth=lw)
            axs[f'1{c}'].set_xlim(0, len(shapeletsplots[r])-1)
            axs[f'1{c}'].set_title(f"Shapelet $S_{{{shapeletidx[r]}}}$")
            if outdir:
                fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_shapelet_{shapeletidx[r]}_IG_{ig[r]}.pdf', bbox_inches='tight')       
            if c == 1:
                fig = plt.figure(num=int(f'2{c}'), figsize=figsize, tight_layout=True)
                axs[f'2{c}'] = fig.add_subplot(111, autoscale_on=True)
                sns.kdeplot(data=dataToplot, y=None, x=shapeletidx[r], hue='Predlabels', fill=True, multiple="layer", linewidth=2, palette=colorsp, alpha=0.4, ax=axs[f'2{c}'])
                sns.rugplot(data=dataToplot, x=shapeletidx[r], y=None, hue='Truelabels', palette=colorsl, height=0.04, expand_margins=True, ax=axs[f'2{c}'])
                axs[f'2{c}'].legend([],[], frameon=False)
                # axs[f'2{c}'].set_xlabel(f'Shapelet {shapeletidx[0]}', labelpad=lp)
                axs[f'2{c}'].set_xlabel(f"$S_{{{shapeletidx[0]}}}$", labelpad=lp)
                if outdir:
                    fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_scatter_shapelets_{shapeletidx[r]}_IG_{ig[r]}.pdf', bbox_inches='tight')
            if c == 2:
                fig = plt.figure(num=int(f'2{c}'), figsize=figsize, tight_layout=True)
                axs[f'2{c}'] = fig.add_subplot(111, autoscale_on=True)
                sns.kdeplot(data=dataToplot, y=shapeletidx[r], x=shapeletidx[r-1], hue='Predlabels',fill=True, multiple="layer", alpha=0.4, palette=colorsp, n_levels=4, ax=axs[f'2{c}'])
                # sns.kdeplot(data=dataToplot,x=shapeletidx[r-1],y=None,hue='Predlabels', fill=True, multiple="layer", alpha=0.4, palette=colorsp, n_levels=4, ax=axs[f'2{c}'])
                # sns.kdeplot(data=dataToplot,x=None,y=shapeletidx[r],hue='Predlabels', fill=True, multiple="layer", alpha=0.4, palette=colorsp, n_levels=4, ax=axs[f'2{c}'])
                sns.scatterplot(data=dataToplot, x=shapeletidx[r-1], y=shapeletidx[r], hue='Predlabels', style='Truelabels', markers=markersl, alpha=0.7, s=s2d, palette=colorsp, ax=axs[f'2{c}'])
                sns.rugplot(data=dataToplot, x=shapeletidx[r-1], y=shapeletidx[r], hue='Truelabels', palette=colorsl, height=0.05, expand_margins=True, ax=axs[f'2{c}'])
                axs[f'2{c}'].legend([],[], frameon=False)
                axs[f'2{c}'].set_xlabel(f"$S_{{{shapeletidx[0]}}}$", labelpad=lp)
                axs[f'2{c}'].set_ylabel(f"$S_{{{shapeletidx[1]}}}$", labelpad=lp)
                if outdir:
                    fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_scatter_shapelets_{shapeletidx[r]}_{shapeletidx[r-1]}_IG_{ig[r]}.pdf', bbox_inches='tight')
            if c == 3:
                bins=100j
                delta = 0.5
                fig = plt.figure(num=int(f'2{c}'), figsize=figsize, tight_layout=True)
                axs[f'2{c}'] = fig.add_subplot(111, projection='3d', autoscale_on=True)
                for i in range(dataToplot.shape[0]):
                    cr = colorsp[int(dataToplot.iloc[i].Predlabels)]
                    m = markersl[int(dataToplot.iloc[i].Truelabels)]
                    axs[f'2{c}'].scatter(dataToplot[shapeletidx[0]][i], dataToplot[shapeletidx[1]][i], dataToplot[shapeletidx[2]][i], color=cr, marker=m, alpha=0.7, s=s, edgecolors='white')
                xminG, xmaxG = min(dataToplot[shapeletidx[0]]), max(dataToplot[shapeletidx[0]])
                yminG, ymaxG = min(dataToplot[shapeletidx[1]]), max(dataToplot[shapeletidx[1]])
                zminG, zmaxG = min(dataToplot[shapeletidx[2]]), max(dataToplot[shapeletidx[2]])
                for ii in  dataToplot[list(shapeletidx) + ["Predlabels"]].groupby("Predlabels"):
                    cr = colorsp[ii[0]]
                    points = ii[1][shapeletidx]
                    density = Get3dkde(points[shapeletidx].values,bins=bins)
                    xmin, xmax, = min(points[shapeletidx[0]]), max(points[shapeletidx[0]])
                    ymin, ymax = min(points[shapeletidx[1]]), max(points[shapeletidx[1]])
                    zmin, zmax = min(points[shapeletidx[2]]), max(points[shapeletidx[2]])
                    axs[f'2{c}'] = plot3dkde(density, [(xmin,xminG),(xmax,xmaxG),(ymin,yminG),(ymax,ymaxG),(zmin,zminG),(zmax,zmaxG)], axs[f'2{c}'], delta=0.5, bins=100j,levels=30, colors=cr, alpha=0.6)
                axs[f'2{c}'].set_xlim(xminG - delta/rd3, xmaxG + delta/rd3)
                axs[f'2{c}'].set_ylim(yminG - delta/rd3, ymaxG + delta/rd3)
                axs[f'2{c}'].set_zlim(zminG - delta/rd3, zmaxG + delta/rd3)
                axs[f'2{c}'].set_xlabel(f"$S_{{{shapeletidx[0]}}}$", labelpad=lp+px)
                axs[f'2{c}'].set_ylabel(f"$S_{{{shapeletidx[1]}}}$", labelpad=lp+py)
                axs[f'2{c}'].set_zlabel(f"$S_{{{shapeletidx[2]}}}$", labelpad=lp+pz)
                if outdir:
                    fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_scatter_shapelets_{shapeletidx[r]}_{shapeletidx[r-1]}_{shapeletidx[r-2]}_IG_{ig[r]}.pdf', bbox_inches='tight')
        fig = plt.figure(figsize=(1,1), tight_layout=True)
        axtemp = fig.add_subplot(111)
        titlep = Line2D([np.NaN],[np.NaN],color='none',label='$\\bf{Clusters}$')
        patchesp = [mpatches.Patch(color=c[1],label=f'{c[0]}') for c in colorsp.items()]
        titlel = Line2D([np.NaN],[np.NaN],color='none',label='$\\bf{Classes}$')
        patchesl = [mpatches.Patch(color=c[1],label=f'{c[0]}') for c in colorsl.items()]
        patchesm = [plt.scatter([np.NaN],[np.NaN], s=50, label=f'{m[0]}', color='k', marker=m[1]) for m in markersl.items()]
        handles= [titlep] + patchesp + [titlel] + patchesm + patchesl 
        leg = axtemp.legend(handles=handles,facecolor='none', frameon=False,bbox_to_anchor=[0,0,1.9,1.35])
        for itm, lbl in zip(leg.legendHandles, leg.texts):
            if lbl._text  in ['$\\bf{Clusters}$','$\\bf{Classes}$']:
                width=35
                lbl.set_ha('left')
                lbl.set_position((-1*width,0))
        plt.gca().set_axis_off()
        if outdir:
            fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_legend_sce.pdf', bbox_inches='tight', pad_inches = 0)
            plt.close('all')
    if 'Local' in QMunderstudyName and 'Local' in scopetoplot:
        for g in QMunderstudy.groupby(['Cluster']):
            sortedIG = g[1].sort_values(by='IG', ascending=False) if 'Single' in QMunderstudyName else g[1]
            if notallcluster:
                if int(sortedIG.Cluster.iloc[0]) != clustertoplot:
                    continue
            # colorslc = {g[0]:colorsl[int(g[0])], -1:['#3C363A']} 
            # colorspc = {g[0]:colorsp[int(g[0])], -1:['#3C363A']}
            # markerslc = {g[0]:markersl[int(g[0])], -1:listmarkers[-1]}
            ig = sortedIG.IG.to_numpy()[:shapeletstoplot] 
            shapeletidx = sortedIG.Shapelet.to_numpy().astype(np.int64)[:shapeletstoplot] if 'Single' in QMunderstudyName else sortedIG.Shapelet.iloc[-1].astype(np.int64)[:shapeletstoplot]
            shapeletsplots = shapelets_df[shapeletidx]
            dataToplot = pd.concat([pd.DataFrame(data=embTslearnNUmpy, columns=range(embTslearnNUmpy.shape[1])), pd.DataFrame({"Predlabels": CSKlabels, "Truelabels": labels})], axis=1)
            dataToplot.reset_index(level=0, inplace=True)
            axs = {}
            for c in range(1, shapeletstoplot+1):
                r = c-1
                fig = plt.figure(num=int(f'1{c}{g[0]}'), figsize=figsize, tight_layout=True)  
                axs[f'1{c}{g[0]}'] = fig.add_subplot(111, autoscale_on=True)
                axs[f'1{c}{g[0]}'].plot(shapeletsplots[r], linewidth=lw)
                axs[f'1{c}{g[0]}'].set_xlim(0, len(shapeletsplots[r])-1)
                axs[f'1{c}{g[0]}'].set_title(f"Shapelet $S_{{{shapeletidx[r]}}}$")
                if outdir:
                    fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_GE_shapelet_{shapeletidx[r]}_IG_{ig[r]}.pdf', bbox_inches='tight')       
                if c == 1:
                    fig = plt.figure(num=int(f'2{c}{g[0]}'), figsize=figsize, tight_layout=True)
                    axs[f'2{c}{g[0]}'] = fig.add_subplot(111, autoscale_on=True)
                    sns.kdeplot(data=dataToplot, y=None, x=shapeletidx[r], hue='Predlabels', fill=True, multiple="layer", linewidth=2, palette=colorsp, alpha=0.4, ax=axs[f'2{c}{g[0]}'])
                    sns.rugplot(data=dataToplot, x=shapeletidx[r], y=None, hue='Truelabels', palette=colorsl, height=0.04, expand_margins=True, ax=axs[f'2{c}{g[0]}'])
                    axs[f'2{c}{g[0]}'].legend([],[], frameon=False)
                    axs[f'2{c}{g[0]}'].set_xlabel(f"S_{{{shapeletidx[0]}}}$", labelpad=lp)
                    if outdir:
                        fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_CE_{g[0]}_scatter_shapelets_{shapeletidx[r]}_IG_{ig[r]}.pdf', bbox_inches='tight')
                if c == 2:
                    fig = plt.figure(num=int(f'2{c}{g[0]}'), figsize=figsize, tight_layout=True)
                    axs[f'2{c}{g[0]}'] = fig.add_subplot(111, autoscale_on=True)
                    sns.kdeplot(data=dataToplot, y=shapeletidx[r], x=shapeletidx[r-1], hue='Predlabels',fill=True, multiple="layer", alpha=0.4, palette=colorsp, n_levels=4, ax=axs[f'2{c}{g[0]}'])
                    sns.scatterplot(data=dataToplot, x=shapeletidx[r-1], y=shapeletidx[r], hue='Predlabels', style='Truelabels', sizes=(50, 300), alpha=0.7, s=s2d, palette=colorsp, ax=axs[f'2{c}{g[0]}'])
                    sns.rugplot(data=dataToplot, x=shapeletidx[r-1], y=shapeletidx[r], hue='Truelabels', palette=colorsl, height=0.05, expand_margins=True, ax=axs[f'2{c}{g[0]}'])
                    axs[f'2{c}{g[0]}'].legend([],[], frameon=False)
                    axs[f'2{c}{g[0]}'].set_xlabel(f"$S_{{{shapeletidx[0]}}}$", labelpad=lp)
                    axs[f'2{c}{g[0]}'].set_ylabel(f"$S_{{{shapeletidx[1]}}}$", labelpad=lp)
                    if outdir:
                        fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_CE_{g[0]}_scatter_shapelets_{shapeletidx[r]}_{shapeletidx[r-1]}_IG_{ig[r]}.pdf', bbox_inches='tight')
                if c == 3:
                        bins=100j
                        delta = 0.5
                        fig = plt.figure(num=int(f'2{c}{g[0]}'), figsize=figsize, tight_layout=True)
                        axs[f'2{c}{g[0]}'] = fig.add_subplot(111, projection='3d', autoscale_on=True)
                        for i in range(dataToplot.shape[0]):
                            cr = colorsp[int(dataToplot.iloc[i].Predlabels)]
                            m = markersl[int(dataToplot.iloc[i].Truelabels)]
                            axs[f'2{c}{g[0]}'].scatter(dataToplot[shapeletidx[0]][i], dataToplot[shapeletidx[1]][i], dataToplot[shapeletidx[2]][i], color=cr, marker=m, alpha=0.7, s=s, edgecolors='white')
                        xminG, xmaxG = min(dataToplot[shapeletidx[0]]), max(dataToplot[shapeletidx[0]])
                        yminG, ymaxG = min(dataToplot[shapeletidx[1]]), max(dataToplot[shapeletidx[1]])
                        zminG, zmaxG = min(dataToplot[shapeletidx[2]]), max(dataToplot[shapeletidx[2]])
                        for ii in  dataToplot[list(shapeletidx) + ["Predlabels"]].groupby("Predlabels"):
                            cr = colorsp[ii[0]]
                            points = ii[1][shapeletidx]
                            density = Get3dkde(points[shapeletidx].values,bins=bins)
                            xmin, xmax, = min(points[shapeletidx[0]]), max(points[shapeletidx[0]])
                            ymin, ymax = min(points[shapeletidx[1]]), max(points[shapeletidx[1]])
                            zmin, zmax = min(points[shapeletidx[2]]), max(points[shapeletidx[2]])
                            axs[f'2{c}{g[0]}'] = plot3dkde(density, [(xmin,xminG),(xmax,xmaxG),(ymin,yminG),(ymax,ymaxG),(zmin,zminG),(zmax,zmaxG)], axs[f'2{c}{g[0]}'], delta=0.5, bins=100j,levels=30, colors=cr, alpha=0.6)
                        axs[f'2{c}{g[0]}'].set_xlim(xminG - delta/rd3, xmaxG + delta/rd3)
                        axs[f'2{c}{g[0]}'].set_zlim(zminG - delta/rd3, zmaxG + delta/rd3)
                        axs[f'2{c}{g[0]}'].set_ylim(yminG - delta/rd3, ymaxG + delta/rd3)
                        axs[f'2{c}{g[0]}'].set_xlabel(f"S_{{{shapeletidx[0]}}}$", labelpad=lp+px)
                        axs[f'2{c}{g[0]}'].set_ylabel(f"S_{{{shapeletidx[1]}}}$", labelpad=lp+py)
                        axs[f'2{c}{g[0]}'].set_zlabel(f"S_{{{shapeletidx[2]}}}$", labelpad=lp+pz)
                        if outdir:
                            fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_CE_{g[0]}_scatter_shapelets_{shapeletidx[r]}_{shapeletidx[r-1]}_{shapeletidx[r-2]}_IG_{ig[r]}.pdf', bbox_inches='tight')
            fig = plt.figure(num=int(f'99{g[0]}'),figsize=(1,1), tight_layout=True)
            axtemp = fig.add_subplot(111)
            titlep = Line2D([np.NaN],[np.NaN],color='none',label='$\\bf{Clusters}$')
            patchesp = [mpatches.Patch(color=c[1],label=f'{c[0]}') for c in colorsp.items()]
            titlel = Line2D([np.NaN],[np.NaN],color='none',label='$\\bf{Classes}$')
            patchesl = [mpatches.Patch(color=c[1],label=f'{c[0]}') for c in colorsl.items()]
            patchesm = [plt.scatter([np.NaN],[np.NaN], s=50, label=f'{m[0]}', color='k', marker=m[1]) for m in markersl.items()]
            handles= [titlep] + patchesp + [titlel] + patchesm + patchesl 
            leg = axtemp.legend(handles=handles,facecolor='none', frameon=False,bbox_to_anchor=[0,0,1.9,1.35])
            for itm, lbl in zip(leg.legendHandles, leg.texts):
                if lbl._text  in ['$\\bf{Clusters}$','$\\bf{Classes}$']:
                    width=35
                    lbl.set_ha('left')
                    lbl.set_position((-1*width,0))
            plt.gca().set_axis_off()
            if outdir:
                fig.savefig(f'{outdir}/{dataset}_{QMunderstudyName}_CE_{g[0]}_legend_sce.pdf', bbox_inches='tight', pad_inches = 0)
                plt.close('all')
                
def Get3dkde(points, kernel="Gaussian",bins=100j):
    if kernel == "Gaussian":
        kde =  sp.stats.gaussian_kde
    else:
        kde = kernel
    xmin, xmax = min(points[:,0]), max(points[:,0])
    ymin, ymax = min(points[:,1]), max(points[:,1]) 
    zmin, zmax = min(points[:,2]), max(points[:,2])
    x, y, z = np.mgrid[xmin:xmax:bins, ymin:ymax:bins, zmin:zmax:bins]
    positions = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    krnl = kde(points.T)
    density = np.reshape(krnl(positions).T,x.shape)
    return density

def plot3dkde(density, bounds, ax, delta=0.5, bins=100j, levels=50, colors=None ,cmap=None, alpha=0.5):
    #bounds = [xmin,xmax,ymin,ymax,zmin,zmax]

    #projection to the z-axis
    plotdat = np.sum(density, axis=2)
    plotdat = plotdat /np.max(plotdat)
    plotx, ploty = np.mgrid[bounds[0][0]:bounds[1][0]:bins, bounds[2][0]:bounds[3][0]:bins]
    ax.contour(plotx,ploty,plotdat,offset=bounds[4][1]-delta,zdir="z",levels=levels, colors=colors, cmap=cmap, alpha=alpha)

    #projection to the y-axis
    plotdat = np.sum(density, axis=1)
    plotdat = plotdat /np.max(plotdat)
    plotx, plotz = np.mgrid[bounds[0][0]:bounds[1][0]:bins, bounds[4][0]:bounds[5][0]:bins]
    ax.contour(plotx,plotdat,plotz,offset=bounds[3][1]+delta,zdir="y",levels=levels, colors=colors, cmap=cmap, alpha=alpha)

    #projection to the x-axis
    plotdat = np.sum(density, axis=0)
    plotdat = plotdat /np.max(plotdat)
    ploty, plotz = np.mgrid[bounds[2][0]:bounds[3][0]:bins, bounds[4][0]:bounds[5][0]:bins]
    ax.contour(plotdat,ploty,plotz,offset=bounds[0][1]-delta,zdir="x",levels=levels, colors=colors, cmap=cmap, alpha=alpha)
    return ax
# %%
