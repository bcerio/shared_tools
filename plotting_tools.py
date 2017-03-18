import matplotlib

matplotlib.use('Agg')

import sys
import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import math

reload(sys)
sys.setdefaultencoding('utf-8')

#import seaborn
#seaborn.palplot(seaborn.color_palette('muted'))

sys.path.append(os.path.join(os.environ['ROOTDIR'],'shared_tools'))
from logging_tools import Logger

logger = Logger()

def plot_hist(data_array,n_bins,axis_range,group_labels=None,xlabel='',ylabel='',title='',figure_name='',do_log=False,bin_vec=None,norm_hist=False,error_bar=False,do_stats=False):
    logger = Logger()
    color_vec = ['blue','green','red','black','purple']
    additional_colors = [plt.get_cmap('gnuplot')(i) for i in np.linspace(0, 1, 12)]
    color_vec = color_vec + additional_colors
    if type(data_array) is not dict:
        n_groups = 1
    else:
        n_groups = len(data_array)
    logger.log('Based on shape of input data, plotting %s groups against eachother' % n_groups)
    if type(data_array) is dict and group_labels == None:
        #group_labels = ['group %s' % i for i in np.arange(n_groups)]
        group_labels = {kk:kk for kk in data_array.keys()}
    elif type(data_array) is dict:
        if len(group_labels) != n_groups:
            logger.warning('Label list does not match N_groups. Setting to default')
            group_labels = {kk:kk for kk in data_array.keys()}
        else:
            logger.log('Group labels set to: %s' % str(group_labels))
    else:
        group_labels = ['']

    if do_stats:
        for label in group_labels.keys():
            group_labels[label] = '%s (mean=%s,median=%s)' % (label,round(np.mean(data_array[label]),2),round(np.median(data_array[label]),2))

    if figure_name == '':
        figure_name = 'figure'

    if not bin_vec:
        plot_variable_bin = False
        bin_vec = []
    else:
        plot_variable_bin = True

    if n_groups == 1:
        xx_overflow = []
        for x in xx:
            if not plot_variable_bin:
                if x >= axis_range[1]:
                    xx_overflow.append(axis_range[1]-0.00001)
                else:
                    xx_overflow.append(x)
            else:
                if x >= bin_vec[-1]:
                    xx_overflow.append(bin_vec[-1]-0.00001)
                else:
                    xx_overflow.append(x)
        data_array = xx_overflow
    else:
        xx_overflow_dict = {}
        for key,xx in data_array.iteritems():
            xx_overflow = []
            for x in xx:
                if not plot_variable_bin:
                    if x >= axis_range[1]:
                        xx_overflow.append(axis_range[1]-0.00001)
                    else:
                        xx_overflow.append(x)
                else:
                    if x >= bin_vec[-1]:
                        xx_overflow.append(bin_vec[-1]-0.00001)
                    else:
                        xx_overflow.append(x)

            xx_overflow_dict[key] = xx_overflow

        data_array = xx_overflow_dict        
        
    bin_centers = []
    for ii,bin_bound in enumerate(bin_vec):
        if ii == len(bin_vec) - 1: break
        bin_centers.append(bin_vec[ii] + (bin_vec[ii+1]-bin_vec[ii])/2)

    plt.figure()
    bin_contents = []
    if n_groups > 1:
        for i,key in enumerate(data_array.keys()):
            if not plot_variable_bin:
                n,bins,patches = plt.hist(data_array[key],n_bins,normed=int(norm_hist),facecolor=color_vec[i],alpha=0.3,range=axis_range,label=str(group_labels[key]),histtype='step',fill=False,linewidth=1.75,color=color_vec[i])
            else:
                n,bins,patches = plt.hist(data_array[key],bins=bin_vec,normed=int(norm_hist),facecolor=color_vec[i],alpha=0.3,range=axis_range,label=str(group_labels[key]),histtype='step',fill=False,linewidth=1.75,color=color_vec[i])
            if error_bar:
                mid = 0.5*(bins[1:] + bins[:-1])
                plt.errorbar(mid,n,yerr=map(lambda x: (float(x)/np.sum(n))**0.5,n),fmt='none',color=color_vec[i],ecolor=color_vec[i],alpha=0.3)
            bin_contents = np.concatenate((bin_contents,n))
    else:
        if not plot_variable_bin:
            n,bins,patches = plt.hist(data_array,n_bins,normed=int(norm_hist),facecolor=color_vec[0],alpha=0.6,range=axis_range,label=group_labels[0],histtype='step',fill=False,linewidth=1.75)
        else:
            n,bins,patches = plt.hist(data_array,bins=bin_vec,normed=int(norm_hist),facecolor=color_vec[0],alpha=0.6,range=axis_range,label=group_labels[0],histtype='step',fill=False,linewidth=1.75)
        bin_contents = np.concatenate((bin_contents,n))

    #plt.yscale('log')
    plt.ylim([0,np.max(bin_contents)*1.35])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if n_groups > 1:
        plt.legend()
    fig_file_name = 'figures/%s.jpg' % figure_name
    if not os.path.exists(os.path.join(os.getcwd(),'figures')): os.makedirs(os.path.join(os.getcwd(),'figures'))
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_ratio(data_num,data_denom,n_bins,axis_range,group_labels=None,xlabel='',ylabel='',title='',figure_name='',bin_vec=None):
    logger = Logger()
    color_vec = ['blue','green','red','black','purple']
    if type(data_num) is not dict:
        n_groups = 1
    else:
        n_groups = len(data_num)
    logger.log('Based on shape of input data, plotting %s groups against eachother' % n_groups)
    if type(data_num) is dict and group_labels == None:
        #group_labels = ['group %s' % i for i in np.arange(n_groups)]
        group_labels = {key:'group %s' % i for i,key in data_num.keys()}
    elif type(data_num) is dict:
        if len(group_labels) != n_groups:
            logger.warning('Label list does not match N_groups. Setting to default')
            group_labels = {key:'group %s' % i for i,key in data_num.keys()}            
    else:
        group_labels = ['']

    if figure_name == '':
        figure_name = 'figure'

    if not bin_vec:
        plot_variable_bin = False
        bin_vec = [axis_range[0] + float(i)*(axis_range[1]-axis_range[0])/float(n_bins) for i in np.arange(n_bins)]
        bin_vec.append(axis_range[1])
    else:
        plot_variable_bin = True
        axis_range[0] = bin_vec[0]
        axis_range[1] = bin_vec[-1]        

    # compute overflow

    if n_groups > 1:
        overflow_num = {key:0 for key in data_num.keys()}
        overflow_denom = {key:0 for key in data_denom.keys()}

        for key in data_num.keys():
            for cc in data_num[key]:
                if cc > axis_range[1]:
                    overflow_num[key] += 1
            for cc in data_denom[key]:
                if cc > axis_range[1]:
                    overflow_denom[key] += 1

    else:

        overflow_num = 0
        overflow_denom = 0
        for cc in data_num:
            if cc > axis_range[1]:
                overflow_num += 1
        for cc in data_denom:
            if cc > axis_range[1]:
                overflow_denom += 1
        
    bin_centers = []
    bin_width = []
    for ii,bin_bound in enumerate(bin_vec):
        if ii == len(bin_vec) - 1: break
        bin_center = float(bin_vec[ii]) + float(bin_vec[ii+1]-bin_vec[ii])/float(2)
        bin_centers.append(bin_center)
        bin_width.append(bin_center-bin_vec[ii])

    num_dict = {}
    denom_dict = {}
    ratio_dict = {}
    error_dict = {}

    ratio_list = []
    
    if n_groups > 1:
        for key in data_num.keys():
            if not plot_variable_bin:
                n_num,bins,patches = plt.hist(data_num[key],n_bins,normed=0,facecolor=color_vec[key],alpha=0.6,range=axis_range,label=str(group_labels[key]),histtype='step',fill=False,linewidth=1.5)
                n_denom,bins,patches = plt.hist(data_denom[key],n_bins,normed=0,facecolor=color_vec[key],alpha=0.6,range=axis_range,label=str(group_labels[key]),histtype='step',fill=False,linewidth=1.5)                
            else:
                n_num,bins,patches = plt.hist(data_num[key],bins=bin_vec,normed=0,facecolor=color_vec[key],alpha=0.6,range=axis_range,label=str(group_labels[key]),histtype='step',fill=False,linewidth=1.5)
                n_denom,bins,patches = plt.hist(data_denom[key],bins=bin_vec,normed=0,facecolor=color_vec[key],alpha=0.6,range=axis_range,label=str(group_labels[key]),histtype='step',fill=False,linewidth=1.5)

            n_num[-1] += overflow_num[key]
            n_denom[-1] += overflow_denom[key]
                
            num_dict[key] = n_num
            denom_dict[key] = n_denom
            ratio_dict[key] = [float(n_num[i])/float(n_denom[i]) if n_denom[i] != 0 else 0 for i in np.arange(len(n_num))]
            error_dict[key] = [float(n_num[i])/float(n_denom[i])*(1/float(n_num[i]) + 1/float(n_denom[i]))**0.5 if (n_denom[i] != 0 and n_num[i] != 0) else 0 for i in np.arange(len(n_num))]
            ratio_list += ratio_dict[key]

        plt.figure()
        for i,key in enumerate(ratio_dict.keys()):
            plt.errorbar(bin_centers,ratio_dict[key],xerr=bin_width,yerr=error_dict[key],label=str(group_labels[i]),fmt='o',color=color_vec[i],ecolor=color_vec[i],alpha=0.6)
            
    else:
        if not plot_variable_bin:
            n_num,bins,patches = plt.hist(data_num,n_bins,normed=0,facecolor=color_vec[0],alpha=0.6,range=axis_range,label=group_labels[0],histtype='step',fill=False,linewidth=1.5)
            n_denom,bins,patches = plt.hist(data_denom,n_bins,normed=0,facecolor=color_vec[0],alpha=0.6,range=axis_range,label=group_labels[0],histtype='step',fill=False,linewidth=1.5)            
        else:
            n_num,bins,patches = plt.hist(data_num,bins=bin_vec,normed=0,facecolor=color_vec[0],alpha=0.6,range=axis_range,label=group_labels[0],histtype='step',fill=False,linewidth=1.5)
            n_denom,bins,patches = plt.hist(data_denom,bins=bin_vec,normed=0,facecolor=color_vec[0],alpha=0.6,range=axis_range,label=group_labels[0],histtype='step',fill=False,linewidth=1.5)

        n_num[-1] += overflow_num
        n_denom[-1] += overflow_denom

        ratio_list = [float(n_num[i])/float(n_denom[i]) if n_denom[i] != 0 else 0 for i in np.arange(len(n_num))]
        error_list = [float(n_num[i])/float(n_denom[i])*(1/float(n_num[i]) + 1/float(n_denom[i]))**0.5 if (n_denom[i] != 0 and n_num[i] != 0) else 0 for i in np.arange(len(n_num))]

        plt.figure()
        plt.errorbar(bin_centers,ratio_list,xerr=bin_width,yerr=error_list,fmt='o',color=color_vec[0],ecolor=color_vec[0],alpha=0.6)

    plt.ylim([0,np.max(ratio_list)*1.35])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if n_groups > 1:
        plt.legend()
    fig_file_name = 'figures/%s.jpg' % figure_name
    if not os.path.exists(os.path.join(os.getcwd(),'figures')): os.makedirs(os.path.join(os.getcwd(),'figures'))        
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_scatter(xdata,ydata,xlabel,ylabel,title,figure_name,bin_bounds=[],do_nonparametric=True,dpi=500):

    bin_content = {}
    bin_error = {}    

    avg_rate = [list([]) for i in np.arange(len(bin_bounds)-1)]
    for i in np.arange(len(bin_bounds)):
        for indy,dd in enumerate(xdata):
            if i == len(bin_bounds) - 1:
                continue
            if i == len(bin_bounds) - 2:                    
                if dd >= bin_bounds[i]:
                    avg_rate[i].append(ydata[indy])
            else:
                if dd >= bin_bounds[i] and dd < bin_bounds[i+1]:
                    avg_rate[i].append(ydata[indy])

    if not do_nonparametric:
        mean_rate = [np.mean(avg_rate[ii]) if len(avg_rate[ii]) != 0 else 0 for ii in np.arange(len(bin_bounds)-1)]
        std_rate = [np.std(avg_rate[ii])/np.sqrt(len(avg_rate[ii])) if len(avg_rate[ii]) != 0 else np.max(avg_rate[ii]) for ii in np.arange(len(bin_bounds)-1)]
    else:
        mean_rate = [np.median(avg_rate[ii]) if len(avg_rate[ii]) != 0 else 0 for ii in np.arange(len(bin_bounds)-1)]
        error_low = [np.percentile(avg_rate[ii],25) if len(avg_rate[ii]) != 0 else 0 for ii in np.arange(len(bin_bounds)-1)]
        error_high = [np.percentile(avg_rate[ii],75) if len(avg_rate[ii]) != 0 else 0 for ii in np.arange(len(bin_bounds)-1)]
        std_rate = [error_low,error_high]

    n_points = [len(avg_rate[i]) for i in np.arange(len(bin_bounds)-1)]
                    
    bin_widths = []
    bin_centers = []

    for i in np.arange(len(bin_bounds)):
        if i == len(bin_bounds) - 1:
            continue
        bin_center = float(bin_bounds[i]) + float(bin_bounds[i+1]-bin_bounds[i])/float(2)
        bin_centers.append(bin_center)
        bin_widths.append(bin_center-bin_bounds[i])

    y_lim_high = np.max(map(lambda x: mean_rate[x] + error_high[x],np.arange(len(mean_rate))))*1.1
    dx = (bin_bounds[-1] - bin_bounds[0])/100.0
    dy = y_lim_high/100.0
    max_list = []
    plt.figure()
    plt.errorbar(bin_centers,mean_rate,xerr=bin_widths,yerr=std_rate,fmt='o',alpha=0.6,color='blue')
    for i in np.arange(len(bin_centers)):
        plt.annotate('%s' % n_points[i],xy=(bin_centers[i]+dx,mean_rate[i]+dy),xytext=(bin_centers[i]+dx,mean_rate[i]+dy),fontsize=8)
    plt.ylim([0,y_lim_high])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name,dpi=dpi)
    plt.close()

    return 1

def plot_2D_hist(xdata,ydata,xlabel,ylabel,title,figure_name,xlim=None,ylim=None,xbins=10,ybins=10,bin_bounds_x=[],bin_bounds_y=[]):

    xdata_temp = []
    ydata_temp = []            
    if len(bin_bounds_y) > 0 and len(bin_bounds_x) > 0:
        do_variable_bin = True
        upper_limit_x = bin_bounds_x[-1]
        upper_limit_y = bin_bounds_y[-1]
    else:
        do_variable_bin = False
        upper_limit_x = xlim[-1]
        upper_limit_y = ylim[-1]

    for i in np.arange(len(xdata)):
        if xdata[i] >= upper_limit_x:
            xdata_temp.append(upper_limit_x-0.001)
        else:
            xdata_temp.append(xdata[i])
    for i in np.arange(len(ydata)):
        if ydata[i] >= upper_limit_y:
            ydata_temp.append(upper_limit_y-0.001)
        else:
            ydata_temp.append(ydata[i])                
        
    if do_variable_bin:
        hist_array,x_bins,y_bins = np.histogram2d(xdata,ydata,bins=[bin_bounds_x,bin_bounds_y])
    else:
        hist_array,x_bins,y_bins = np.histogram2d(xdata,ydata,range=[list(xlim), list(ylim)],bins=[xbins,ybins])        

    bin_widths_x = map(lambda x: bin_bounds_x[x+1] - bin_bounds_x[x],np.arange(len(bin_bounds_x)-1))
    bin_widths_y = map(lambda y: bin_bounds_y[y+1] - bin_bounds_y[y],np.arange(len(bin_bounds_y)-1))
    
    bin_content_list = []
    xlab_list = []
    ylab_list = []
    
    for i in np.arange(hist_array.shape[0]):
        for j in np.arange(hist_array.shape[1]):

            bin_content = hist_array[i][j]
            bin_content_list.append(bin_content)
            xlab_list.append(round(x_bins[i]+bin_widths_x[i]/2.0,2))
            ylab_list.append(round(y_bins[j]+bin_widths_y[j]/2.0,2))


    df = pd.DataFrame({xlabel:xlab_list,ylabel:ylab_list,'counts':bin_content_list,'indy':np.arange(len(xlab_list))})
    #df = df.groupby(['x','y'])
    #df = pd.DataFrame({'counts':bin_content_list},index=[x_bins,ylab_list])

    df_pivot = pd.pivot_table(df,index=[ylabel],columns=[xlabel],aggfunc=np.sum,values='counts',fill_value=0.0)
    
    #print hist_array
    #print type(hist_array)

    plt.figure()
    seaborn.heatmap(df_pivot,cmap='YlGnBu',annot=True, fmt='d')
    plt.title(title)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_scatter_points(xdata,ydata,xlabel,ylabel,title,figure_name,xlim=None,ylim=None):

    plt.figure()
    plt.scatter(xdata,ydata,alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    fig_file_name = '%s.jpg' % figure_name
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_scatter_points_multiclass(xdata_dict,ydata_dict,yerr_dict,xlabel,ylabel,title,figure_name,xlim=None,ylim=None):

    plt.figure()
    for label,xdata in xdata_dict.iteritems():
        plt.errorbar(xdata_dict[label],ydata_dict[label],label=label,alpha=0.6,yerr=yerr_dict[label],fmt='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:    
        plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    fig_file_name = '%s.jpg' % figure_name
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_joint(xdata,ydata,xlabel,ylabel,title,figure_name,xlim=None,ylim=None):

    from scipy.stats import spearmanr

    df = pd.DataFrame({'x':xdata,'y':ydata,'indy':np.arange(len(xdata))}) 
    
    plt.figure()
    sns_plot = seaborn.jointplot(x='x', y='y',data=df,kind='hex', color='b',stat_func=spearmanr,xlim=xlim,ylim=ylim)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close('all')
    del sns_plot

    return 1

def plot_violin(xdata,ydata,xlabel,ylabel,title,figure_name,bin_bounds=[]):

    bin_category = []
    
    for i in np.arange(len(bin_bounds)):
        if i == len(bin_bounds) - 1:
            continue
        bin_center = float(bin_bounds[i]) + float(bin_bounds[i+1]-bin_bounds[i])/float(2)
        for x in xdata:
            if i == len(bin_bounds) - 2:
                if x >= float(bin_bounds[i]):
                    bin_category.append(bin_center)
                    continue
            else:
                if x < float(bin_bounds[i+1]) and x >= float(bin_bounds[i]):
                    bin_category.append(bin_center)
                    continue

    df = pd.DataFrame({'x':bin_category,'y':ydata})
            
    plt.figure()
    seaborn.factorplot(x='x',y='y',data=df,kind='violin')
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1


def plot_boxplot(xdata,ydata,xlabel,ylabel,title,figure_name,bin_bounds=[]):

    bin_category = []
    
    for i in np.arange(len(bin_bounds)):
        if i == len(bin_bounds) - 1:
            continue
        bin_center = float(bin_bounds[i]) + float(bin_bounds[i+1]-bin_bounds[i])/float(2)
        for x in xdata:
            if i == len(bin_bounds) - 2:
                if x >= float(bin_bounds[i]):
                    bin_category.append(bin_center)
                    continue
            else:
                if x < float(bin_bounds[i+1]) and x >= float(bin_bounds[i]):
                    bin_category.append(bin_center)
                    continue

    df = pd.DataFrame({'x':bin_category,'y':ydata})
            
    plt.figure()
    seaborn.boxplot(x='x',y='y',data=df)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_stripplot(xdata,ydata,xlabel,ylabel,title,figure_name,bin_bounds=[]):

    bin_category = []
    
    for i in np.arange(len(bin_bounds)):
        if i == len(bin_bounds) - 1:
            continue
        bin_center = float(bin_bounds[i]) + float(bin_bounds[i+1]-bin_bounds[i])/float(2)
        for x in xdata:
            if i == len(bin_bounds) - 2:
                if x >= float(bin_bounds[i]):
                    bin_category.append(bin_center)
                    continue
            else:
                if x < float(bin_bounds[i+1]) and x >= float(bin_bounds[i]):
                    bin_category.append(bin_center)
                    continue

    df = pd.DataFrame({'x':bin_category,'y':ydata})
            
    plt.figure()
    seaborn.stripplot(x='x',y='y',data=df,jitter=True)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_single_hist(xx,nbins,axis_range=[],xlabel='',title='',figure_name='figure',bin_bounds=[]):

    if len(bin_bounds) > 0:
        do_variable_binning = True
    else:
        do_variable_binning = False
    
    bin_widths = []
    bin_centers = []

    for i in np.arange(len(bin_bounds)):
        if i == len(bin_bounds) - 1:
            continue
        bin_center = float(bin_bounds[i]) + float(bin_bounds[i+1]-bin_bounds[i])/float(2)
        bin_centers.append(bin_center)
        bin_widths.append(bin_center-bin_bounds[i])
    
    xx_overflow = []
    for x in xx:
        if not do_variable_binning:
            if x >= axis_range[1]:
                xx_overflow.append(axis_range[1]-0.00001)
            else:
                xx_overflow.append(x)
        else:
            if x >= bin_bounds[-1]:
                xx_overflow.append(bin_bounds[-1]-0.00001)
            else:
                xx_overflow.append(x)

    plt.figure()
    if not do_variable_binning:
        bin_content,bins,patches = plt.hist(xx_overflow,nbins,normed=0,facecolor='g',color='g',alpha=0.4,range=axis_range,histtype='step',fill=True,linewidth=0.0)
    else:
        bin_content,_,_ = plt.hist(xx_overflow,bins=bin_bounds,normed=0,histtype='step',fill=True,linewidth=0.0)
        plt.close()
        counting_error = []
        for bc in bin_content:
            counting_error.append(bc**0.5)
        plt.errorbar(bin_centers,bin_content,xerr=bin_widths,yerr=counting_error,fmt='o',alpha=0.6)
        
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylim([0,np.max(bin_content)*1.2])
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_ranks_bar(variable_dict,number_displayed=-1,annotation_dict={},xlabel='',title='',xlim=None,ylim=None,figure_name='figure',fontsize=16):

    labels = variable_dict.keys()
    df = pd.DataFrame({'var':map(lambda x: variable_dict[x],labels)},index=labels)
    df = df.sort(columns=['var'],ascending=True)

    index_to_rank = {kkey:(len(df)-i-1) for i,kkey in enumerate(df.index)}
    
    if number_displayed != -1:
        df = df[len(df)-number_displayed:len(df)]

    hcell = 0.166
    wcell = 10
    fontscale = 344
    fontscale_small = 301
    
    plt.figure(figsize=[wcell*np.max(df['var'].values),hcell*len(df)],tight_layout=True)
    ax = df['var'].plot(kind='barh',x='var',alpha=0.5,grid=True)
    plt.title(title)
    plt.xlabel(xlabel)
    #plt.setp(ax.get_yticklabels(), fontsize=(float(fontscale)/len(df)))
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)    
    if annotation_dict != {}:
        for kkey,j in index_to_rank.iteritems():
            i = len(df) - j - 1
            if df['var'].values[i] >= 0.0:
                x_coord = 1.05*df['var'].values[i]
                y_coord = len(df['var'].values)-index_to_rank[kkey]-1.2
            else:
                x_coord = 0.0
                y_coord = len(df['var'].values)-index_to_rank[kkey]-1.2
                
            #plt.annotate("""%s""" % annotation_dict[kkey],
            #             xy=(x_coord,y_coord),
            #             xytext=(x_coord,y_coord),
            #             fontsize=(float(fontscale_small)/len(df)))

            plt.annotate("""%s""" % annotation_dict[kkey],
                         xy=(x_coord,y_coord),
                         xytext=(x_coord,y_coord),
                         fontsize=fontsize)
            
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_time_series(variable_list,time_index,sampling_freq='M',xlabel='',ylabel='',title='',ylim=None,figure_name='figure',do_error=False,count_mode=False):

    time_index = pd.DatetimeIndex(time_index)
    df = pd.DataFrame({'dummy':range(len(variable_list)),'var':variable_list,'ts':time_index})

    if sampling_freq == 'M':
        #df['month_year'] = map(lambda x: datetime.datetime.strptime('%s-%s-01' % (x.year(),x.month()),'%Y-%m-%d'),df.ts.values)
        df['month'] = time_index.month
        df['year'] = time_index.year
        df['month_year'] = map(lambda x: pd.to_datetime('%s-%s-01' % (df.year.values[x],df.month.values[x])),np.arange(len(df)))
        df_gb = df.groupby('month_year')
        std_list = df_gb.std()['var'].values
        count_list = df_gb.count()['dummy'].values
        error_list = [std_list[i]/math.sqrt(count_list[i]) for i in np.arange(len(count_list))]
        df_gb = df_gb.mean()
        var_list = df_gb['var']
        if count_mode:
            plot_var = count_list
        else:
            plot_var = var_list
        low_bound = [var_list[i]-error_list[i] for i in np.arange(len(var_list))]
        high_bound = [var_list[i]+error_list[i] for i in np.arange(len(var_list))]
        ts = pd.Series(plot_var,index=df_gb.index)
    else:
        logger.error('sampling frequency not implemented yet')
        raise

    plt.figure()
    ts.plot()
    if ylim is not None:
        plt.ylim(ylim)
    if do_error:
        plt.fill_between(ts.index,low_bound,high_bound,alpha=0.2,color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1

def plot_multi_hist(xx_list,nbins,axis_range=[],xlabel='',title='',figure_name='figure',bin_bounds=[],group_labels=[],do_overflow=True):

    if len(bin_bounds) > 0:
        do_variable_binning = True
    else:
        do_variable_binning = False

    if len(xx_list) != len(group_labels) and len(group_labels) != 0:
        logger.error('Number of labels must match number of groups supplied')
        raise

    if len(group_labels) == 0:
        for i in np.arange(len(xx_list)):
            group_labels.append('Group %s' % i)
        
    bin_widths = []
    bin_centers = []

    for i in np.arange(len(bin_bounds)):
        if i == len(bin_bounds) - 1:
            continue
        bin_center = float(bin_bounds[i]) + float(bin_bounds[i+1]-bin_bounds[i])/float(2)
        bin_centers.append(bin_center)
        bin_widths.append(bin_center-bin_bounds[i])

    if do_overflow:
        xx_list_overflow = []
        for xx in xx_list:
            xx_overflow = []
            for x in xx:
                if not do_variable_binning:
                    if x >= axis_range[1]:
                        xx_overflow.append(axis_range[1]-0.00001)
                    else:
                        xx_overflow.append(x)
                else:
                    if x >= bin_bounds[-1]:
                        xx_overflow.append(bin_bounds[-1]-0.00001)
                    else:
                        xx_overflow.append(x)
            xx_list_overflow.append(xx_overflow)
    else:
        xx_list_overflow = xx_list

    color_list = ['green','blue','red','purple','black']
    plt.figure()
    bin_content_list = []
    if not do_variable_binning:
        for i,xx_overflow in enumerate(xx_list_overflow):
            bin_content,bins,patches = plt.hist(xx_overflow,nbins,normed=0,facecolor=color_list[i],color=color_list[i],alpha=0.6,range=axis_range,histtype='step',fill=False,linewidth=2.0,label=group_labels[i])
            bin_content_list += list(bin_content)
    else:
        for i,xx_overflow in enumerate(xx_list_overflow):
            bin_content,_,_ = plt.hist(xx_overflow,bins=bin_bounds,normed=0,histtype='step',fill=True,linewidth=0.0,alpha=0.0)
            #plt.close()
            counting_error = []
            bin_content_list += list(bin_content)
            for bc in bin_content:
                counting_error.append(bc**0.5)
            plt.errorbar(bin_centers,bin_content,xerr=bin_widths,yerr=counting_error,fmt='o',alpha=0.6,color=color_list[i],facecolor=color_list[i],label=group_labels[i])
        
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylim([0,np.max(bin_content_list)*1.2])
    plt.legend()
    fig_file_name = '%s.jpg' % figure_name
    logger.log('Generating figure %s' % fig_file_name)
    plt.savefig(fig_file_name)
    plt.close()

    return 1
