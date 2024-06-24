


################################################################
#                                                              #
#                                                              #
#          /////////  // // // // //   //  ////// //////       #
#         //         // //  //// ///  //  //       //          #
#        /////////  ////    /// // / //  ////     //           #
#              //  ////     // //  ///  //       //            #
#       ////////  // //    // //   //  ///////  //             #
#                                                              #
#                                                              #
################################################################



# some functions that will be useful for preparing data



import h5py
import torch
import scipy
import random
import scipy.stats
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import sosfilt,iirfilter
from scipy.signal import zpk2sos, find_peaks

from obspy import Trace,Stream
from obspy.core import UTCDateTime as UT
from obspy.geodetics import calc_vincenty_inverse
from obspy.taup import TauPyModel

from torch_geometric.data import Data


import matplotlib.gridspec as gridspec
#from IPython.display import Image
from collections import Counter


plt.rcParams['font.family']='Helvetica'
plt.rcParams['font.size']=18


#  combine multiple examples

#def create_multiple


def open(fname):
	"""
	opens a hf5 file
	"""
	f = h5py.File(fname,'r')
	data = f['data']
	return data


def superposition(examples,delays,scalars=None):
	"""
	function superposition: superposes a list of examples, in terms of the waveforms as they are, 
	given a sligth delay
	:input:

	examples: a python list of hdf5 examples
	delays: a list of delays in seconds 
	scalars: scales amplitudes of examples
	"""

	# check there are examples-1 delays

	#if not len(delays)==len(examples)-1:
	#	print('number of delays not matching waveforms')
		#break

		
	# add on all channels, from the delay onwards or just a chunk?
	example_shape = examples[0][()].shape
	result = np.zeros(shape=example_shape)
	result += examples[0][()]
	if scalars:
		for i in range(1,len(examples)):
			result[:,delays[i-1]:] += examples[i][()][:,0:example_shape[1]-delays[i-1]]*scalars[i-1]

	return result

def plot_example(example):
    traces=example[()]
    fig=plt.figure(figsize=(12,4),tight_layout=True)
    channels=['E','N','Z']

    for i in range(len(traces)):
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(28300,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))

        if example.attrs['Pg_arrival_sample'] != 0:
            pg_pos = example.attrs['Pg_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            if i==0:plt.text(pg_pos-800,-2*i+1,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 0:
            pn_pos = example.attrs['Pn_arrival_sample']
            plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn')
            if i==0:plt.text(pn_pos-800,-2*i+1,'Pn',c='r')
        if example.attrs['Sg_arrival_sample'] != 0:
            sg_pos = example.attrs['Sg_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            if i==0:plt.text(sg_pos,-2*i+1,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 0:
            sn_pos = example.attrs['Sn_arrival_sample']
            plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            if i==0:plt.text(sn_pos,-2*i+1,'Sn',c='b')

        if example.attrs['P_arrival_sample'] != 0:
            p_pos = example.attrs['P_arrival_sample']
            plt.scatter(p_pos,-2*i,s=4000,c='r',marker='|',label='P')
            if i==0:plt.text(p_pos,-2*i+1,'P',c='r')
        if example.attrs['S_arrival_sample'] != 0:
            s_pos = example.attrs['S_arrival_sample']
            plt.scatter(s_pos,-2*i,s=4000,c='b',marker='|',label='S')
            if i==0:plt.text(s_pos,-2*i+1,'S',c='b')



    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    #plt.show()
    return fig


def calculate_network_centroid(lons,lats,codes):
    """
    calcualtes the centroid from a network of stations
    """
    m_lon = np.mean(lons)
    m_lat = np.mean(lats)
    
    return m_lon,m_lat

def sort_stations_to_centroid(lons,lats,codes):
    m_lon,m_lat = calculate_network_centroid(lons,lats,codes)
    distances   = np.sqrt( (lons-m_lon)**2 + (lats-m_lat)**2 )
    sorted_codes = codes[np.argsort(distances)]

    return sorted_codes


def create_ps_box_label(example):
    """
    creates a box label from the p arrival to the s arrival
    
    do it from the earliest starting P? and the earliest arriving S?
    
    """
    
    p_times = np.asarray([example.attrs['Pn_arrival_sample'],example.attrs['Pg_arrival_sample'],example.attrs['P_arrival_sample']])
    s_times = np.asarray([example.attrs['Sn_arrival_sample'],example.attrs['Sg_arrival_sample'],example.attrs['S_arrival_sample']])
    
    p_times = p_times[p_times!='NaN'] #changed from 0. to 'NaN'
    s_times = s_times[s_times!='NaN']
    
    #print(p_times)
    #print(s_times)
    
    p_arrival = min(p_times)
    s_arrival = min(s_times)
    
    y  = np.zeros((1,30000))
    p = float(p_arrival) # changed from int to float
    s = float(s_arrival)
    y[0,p:s] = 1
    
    return y




def create_ps_label_batch(dataset,indices):
    """
    creates a batch of labels from the data examples whose indices are passed,
    puts everything into a pytorch tensor
    
    """
    n_examples = len(indices)
    y_batch = torch.zeros((n_examples,1,30000))
    x_batch = torch.zeros((n_examples,3,30000))
    
    names = []    
   
    for i,index in enumerate(indices):
        example    = dataset[list(dataset.keys())[index]]
        #print(example[()].shape)
        temp_label = torch.tensor(create_ps_box_label(example))
        
        y_batch[i,:,:] = temp_label
        x_batch[i,:,:] = torch.tensor(example[()])        
        names.append(list(dataset.keys())[index])
    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names


def create_ps_label_batch_numpy(dataset,indices,filtered=False):
    """
    creates a batch of labels from the data examples whose indices are passed,
    puts everything into a pytorch tensor
    
    """
    n_examples = len(indices)
    y_batch = np.zeros((n_examples,1,30000))
    x_batch = np.zeros((n_examples,3,30000))
    
    names = []

    for i,index in enumerate(indices):
        example    = dataset[list(dataset.keys())[index]]
        #print(example[()].shape)
        temp_label = create_ps_box_label(example)
        
        y_batch[i,:,:] = temp_label
        if filtered==True:
            x_batch[i,:,:] = bandpass_data(example[()])

        else:
            x_batch[i,:,:] = example[()]        
        names.append(list(dataset.keys())[index])
   
    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names


def random_batch(dataset,n):
    """
    creates a list of random numbers which will serve as indices when calling
    create_ps_label_batch
    """
    n_examples = len(list(dataset.keys()))
    
    indices    = np.random.choice(n_examples,n,replace=False)
    
    x_batch,y_batch,names      = create_ps_label_batch(dataset,indices)
    
    return x_batch,y_batch,names


def random_batch_numpy(dataset,n,filtered=False):
    """
    creates a list of random numbers which will serve as indices when calling
    create_ps_label_batch
    """
    n_examples = len(list(dataset.keys()))
    
    indices    = np.random.choice(n_examples,n,replace=False)
    
    x_batch,y_batch,names      = create_ps_label_batch_numpy(dataset,indices,filtered=filtered)
    
    return x_batch,y_batch,names

def bandpass_data(data,freqmin=0.5,freqmax=20,corners=4):
    """
    apply a bandpass to an example
    first apply a taper, using a Tukey window (very narrow)
    """
    dataf    = np.copy(data)
    len_data = dataf.shape[-1]
    window   = signal.windows.tukey(len_data,alpha=0.05)
    dataf    = dataf*window
    fe       = 0.5 * 100
    low      = freqmin / fe
    high     = freqmax / fe
    
    z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    
    firstpass = sosfilt(sos, dataf)
    return sosfilt(sos, firstpass[::-1])[::-1]


def plot_example_long(example,filtered=False,seisbench=True):
    traces=example[()]
    if filtered:
        traces  = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))
    
    fig = plt.figure(figsize=(16,4),tight_layout=True)
    gs  = gridspec.GridSpec(1,4)
    
    
    
    ax = fig.add_subplot(gs[0,0:3])
    for i in range(len(traces)):
        #x = np.arange(0,300,0.01)
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(28000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        #plt.hlines(-3,0,30000)
        #plt.hlines(-1,0,30000)
        

        if example.attrs['P_arrival_sample'] != 'NaN':
            pg_pos = example.attrs['P_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='P',linewidth=1)
            if i==0:plt.text(pg_pos-600,-5.2,'P',c='r')
        if example.attrs['Pg_arrival_sample'] != 'NaN':
            pg_pos = example.attrs['Pg_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=1)
            if i==0:plt.text(pg_pos,1.2,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 'NaN':
            pn_pos = example.attrs['Pn_arrival_sample']
            plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            if i==0:plt.text(pn_pos-1000,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != 'NaN':
            sg_pos = example.attrs['S_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            if i==0:plt.text(sg_pos,-5.2,'S',c='b')                
        if example.attrs['Sg_arrival_sample'] != 'NaN':
            sg_pos = example.attrs['Sg_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            if i==0:plt.text(sg_pos,1.2,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 'NaN':
            sn_pos = example.attrs['Sn_arrival_sample']
            plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            if i==0:plt.text(sn_pos-1000,1.2,'Sn',c='b')
        
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km'],1)))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    if filtered:
        plt.text(0,-5.9,'Bandpassed',fontsize=12,color='red')

    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)


    
    return fig


def plot_example_seisbench(example,filtered=False,polarity=False):
    traces=example[()]
    length=traces.shape[1]
    if filtered:
        traces  = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))
    
    fig = plt.figure(figsize=(16,4),tight_layout=True)
    gs  = gridspec.GridSpec(1,4)
    
    
    
    ax = fig.add_subplot(gs[0,0:3])
    for i in range(len(traces)):
        #x = np.arange(0,300,0.01)
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(length-1000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        #plt.hlines(-3,0,30000)
        #plt.hlines(-1,0,30000)
        


        if example.attrs['trace_p_arrival_sample'] != 0:
            pg_pos = example.attrs['trace_p_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='P',linewidth=1)
            if i==0:plt.text(pg_pos-600,-5.2,'P',c='r')
                
                
        if example.attrs['trace_s_arrival_sample'] != 0:
            sg_pos = example.attrs['trace_s_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            if i==0:plt.text(sg_pos,-5.2,'S',c='b')                

        if example.attrs['polarity']=='positive':
            plt.annotate("",xy=(pg_pos-200,-3),xytext=(pg_pos-200,-4),arrowprops=dict(arrowstyle="->",color='red'))
            #plt.arrow(pg_pos-200,-4,0,1,color='r',head_width=10)
        if example.attrs['polarity']=='negative':
            plt.annotate("",xy=(pg_pos-200,-5),xytext=(pg_pos-200,-4),arrowprops=dict(arrowstyle="->",color='red'))
            #plt.arrow(pg_pos-200,-4,0,-1,color='r',head_width=10)        

    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,length)
    plt.xticks(np.arange(0,length+1,6000),np.arange(0,(length/100) +1,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']#+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km']/1000)))
    try:
        plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    except Exception as e:
        plt.text(0,-4,'Magnitude          = NaN')
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    if filtered:
        plt.text(0,-5.9,'Bandpassed',fontsize=12,color='red')

    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)


    
    return fig

def plot_example_both(example,figsize=(16,8)):
    """
    plots both the raw waveforms and the filtered waveforms
    """
    traces=example[()]
    f_traces = bandpass_data(traces,freqmin=1)
    f_traces = f_traces / np.max(np.abs(f_traces))

    
    fig = plt.figure(figsize=figsize,tight_layout=True)
    gs  = gridspec.GridSpec(2,4)
    
    
    
    ax = fig.add_subplot(gs[0,0:3])
    for i in range(len(traces)):
        #x = np.arange(0,300,0.01)
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(28000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        #plt.hlines(-3,0,30000)
        #plt.hlines(-1,0,30000)
        


        if example.attrs['P_arrival_sample'] != 0:
            pg_pos = example.attrs['P_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='darkorange',marker='|',label='P',linewidth=0.5)
            if i==0:plt.text(pg_pos-400,-5.2,'P',c='darkorange')
        if example.attrs['Pg_arrival_sample'] != 0:
            pg_pos = example.attrs['Pg_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            if i==0:plt.text(pg_pos,1.2,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 0:
            pn_pos = example.attrs['Pn_arrival_sample']
            plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            if i==0:plt.text(pn_pos-800,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != 0:
            sg_pos = example.attrs['S_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            if i==0:plt.text(sg_pos,-5.2,'S',c='b')                
        if example.attrs['Sg_arrival_sample'] != 0:
            sg_pos = example.attrs['Sg_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            if i==0:plt.text(sg_pos,1.2,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 0:
            sn_pos = example.attrs['Sn_arrival_sample']
            plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            if i==0:plt.text(sn_pos-800,1.2,'Sn',c='b')
        
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[1,0:3])
    for i in range(len(f_traces)):
        #x = np.arange(0,300,0.01)
        plt.plot(f_traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(28000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        #plt.hlines(-3,0,30000)
        #plt.hlines(-1,0,30000)
        


        if example.attrs['P_arrival_sample'] != 0:
            pg_pos = example.attrs['P_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='darkorange',marker='|',label='P',linewidth=0.5)
            if i==0:plt.text(pg_pos-400,-5.2,'P',c='darkorange')
        if example.attrs['Pg_arrival_sample'] != 0:
            pg_pos = example.attrs['Pg_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            if i==0:plt.text(pg_pos,1.2,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 0:
            pn_pos = example.attrs['Pn_arrival_sample']
            plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            if i==0:plt.text(pn_pos-800,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != 0:
            sg_pos = example.attrs['S_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            if i==0:plt.text(sg_pos,-5.2,'S',c='b')                
        if example.attrs['Sg_arrival_sample'] != 0:
            sg_pos = example.attrs['Sg_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            if i==0:plt.text(sg_pos,1.2,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 0:
            sn_pos = example.attrs['Sn_arrival_sample']
            plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            if i==0:plt.text(sn_pos-800,1.2,'Sn',c='b')
        
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    #plt.title(example.name)
    
    ax = fig.add_subplot(gs[0,3])
    ax.axis('off')
    
    id_ = example.attrs['network']+'.'+example.attrs['station']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['event_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['event_origin_longitude'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['event_origin_latitude'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['event_origin_depth']/1000)))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['distance'],2)) )
    
    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)
    
    return fig

# noise generator



def create_noise_example(size):
    """
    creates a random noise waveform
    """
    x = np.random.random(size)
    #x = torch.rand(size)
    return x 
    
def create_noise_with_spikes(size,n_spikes=10,max_amplitude=100):
    """
    creates a random noise waveform with some spikes
    """
    x = np.random.random(size)
    # add some spikes here an here randomly
    # get the indices
    spike_indices    = random.sample(set(np.arange(0,size[0])),n_spikes)
    # get some amplitudes
    spike_amplitudes = np.random.randint(-max_amplitude,max_amplitude,n_spikes)
    
    for i in range(0,size[1]):
        x[spike_indices,i] += spike_amplitudes
    
    x = x / np.max(np.abs(x))
    return x 

def create_noise_sinusoid(size,freqmin=1,freqmax=10000,add_noise=False,noise_intensity=0.5):
    t = np.arange(size[0])
    sinusoid = np.sin(2*np.pi*t*(freqmin/freqmax))
    #sinusoid = np.sin(t/freqmin)
    
    x = np.zeros(size)
    for i in range(0,size[1]):
        x[:,i] = sinusoid
        
    if add_noise:
        x = x + create_noise_example(size)*noise_intensity
        x = x / np.max(np.abs(x))

    return x

def add_vertical_shift(x,magnitude=0.5):
    """
    add a baseline change at a random location
    """
    size = x.shape
    x_new = x.copy()
    change_loc = np.random.randint(0,size[0])
    x_new[change_loc:,:] = x_new[change_loc:,:]+magnitude
    
    x_new = x_new/np.max(np.abs(x_new))
    
    return x_new
   
def create_complex_noise(size):
    """
    complex in the sense of complicated
    """
    # pure noise + multiple sines + multiple spikes + multiple shifts
    x        = create_noise_example(size)
    n_spikes = np.random.randint(size[0]/1000)
    x        = x + create_noise_with_spikes(size,n_spikes=n_spikes)
    n_sines  = np.random.randint(100) # how many to use?
    for i in range(0,n_sines):
        freqmax=np.random.randint(low=50,high=size[0])
        freqmin=np.random.randint(low=10,high=freqmax)
        intensity = np.random.uniform(low=0.5,high=1)
        x = x + create_noise_sinusoid(size,freqmax=freqmax,add_noise=True,noise_intensity=intensity)
        
    x = x/np.max(np.abs(x))
    return x
    

    
def create_complex_steppy_noise(size):
    n_steps = np.random.randint(1,10)
    x = create_complex_noise(size)
    for i in range(n_steps):
        shift = np.random.uniform(low=-0.5,high=0.5)
        x = add_vertical_shift(x,magnitude=shift)
        
    x = x / np.max(np.abs(x))
    return x
 

def create_noise_batch(m,size):
    batch = np.zeros((m,size[0],size[1]))
    for i in range(0,int(m/2)):
        batch[i,:,:] = create_complex_noise(size)
    for i in range(int(m/2),m):
        batch[i,:,:] = create_complex_steppy_noise(size)
    return batch
   


def plot_example_predictions(example,predictions,augmented=False,filtered=False):
    traces=example[()]
    if augmented:
        traces = augmented    
    if filtered:
        traces = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))
    
    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
    
    
    
    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(len(traces)):
        #x = np.arange(0,300,0.01)
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(1000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        #plt.hlines(-3,0,30000)
        #plt.hlines(-1,0,30000)
        


        if example.attrs['P_arrival_sample'] != 'NaN':
            pg_pos = example.attrs['P_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='darkorange',marker='|',label='P',linewidth=0.5)
            if i==0:plt.text(pg_pos-600,-5.2,'P',c='darkorange')
        if example.attrs['Pg_arrival_sample'] != 'NaN':
            pg_pos = example.attrs['Pg_arrival_sample']
            plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            if i==0:plt.text(pg_pos,1.2,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 'NaN':
            pn_pos = example.attrs['Pn_arrival_sample']
            plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            if i==0:plt.text(pn_pos-800,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != 'NaN':
            sg_pos = example.attrs['S_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            if i==0:plt.text(sg_pos+100,-5.2,'S',c='b')                
        if example.attrs['Sg_arrival_sample'] != 'NaN':
            sg_pos = example.attrs['Sg_arrival_sample']
            plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            if i==0:plt.text(sg_pos,1.2,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 'NaN':
            sn_pos = example.attrs['Sn_arrival_sample']
            plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            if i==0:plt.text(sn_pos-800,1.2,'Sn',c='b')
        
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0:2,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km'])))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    if filtered:
        plt.text(0,-5.9,'Bandpassed',fontsize=16,color='red')
    
    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)
    
    ax = fig.add_subplot(gs[2,0:3])
    label = create_ps_box_label(example)
    plt.plot(label[0,:],label='Label',c='r')
    plt.plot(predictions,label='Predictions',c='b')
    plt.xlim(0,30000)
    ax.fill_between(np.arange(0,30000), label[0,:], 0, color='r', alpha=.1,zorder=0)
    plt.xlabel('sample points')
    plt.legend()
    # shade the region under the label
    #ax.fill_between(np.arange(0,30000), label[0,:], 0, color='blue', alpha=.1,zorder=0)


    print(label.shape,predictions.shape)
    bce = estimate_bce(label[0,:],predictions)

    ax = fig.add_subplot(gs[2,3])
    ax.axis('off')
    bce_line = 'BCE = '+str(np.round(bce,4))
    plt.text(0,0,bce_line,fontsize=30)
    plt.yticks([]);plt.xticks([])



    return fig


def plot_example_augmented(example,augmentation_x,augmentation_y,predictions,threshold=0.5,legend=True):
    traces=augmentation_x

    
    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
 
    if type(predictions)==np.ndarray:
        #predictions = np.asarray(predictions)
        #print(predictions)
        p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100)
        s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)   
    
    
    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(len(traces)):
        #x = np.arange(0,300,0.01)
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(28000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        #plt.hlines(-3,0,30000)
        #plt.hlines(-1,0,30000)
        


        if example.attrs['P_arrival_sample'] != 'NaN':
            pg_pos = example.attrs['P_arrival_sample']
            #plt.scatter(pg_pos,-2*i,s=4000,c='darkorange',marker='|',label='P',linewidth=0.5)
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-800,-5.2,'P',c='r')
        if example.attrs['Pg_arrival_sample'] != 'NaN':
            pg_pos = example.attrs['Pg_arrival_sample']
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-1000,1.2,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 'NaN':
            pn_pos = example.attrs['Pn_arrival_sample']
            #plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            plt.axvline(pn_pos,c='r',linestyle='--')
            if i==0:plt.text(pn_pos-1000,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != 'NaN':
            sg_pos = example.attrs['S_arrival_sample']
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos+200,-5.2,'S',c='b')                
        if example.attrs['Sg_arrival_sample'] != 'NaN':
            sg_pos = example.attrs['Sg_arrival_sample']
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos+200,1.2,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 'NaN':
            sn_pos = example.attrs['Sn_arrival_sample']
            #plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            plt.axvline(sn_pos,c='b',linestyle='--')
            if i==0:plt.text(sn_pos-1000,1.2,'Sn',c='b')
 
    try:
        if len(p_picks)>0:
            for p_pick in p_picks:
                plt.axvline(p_pick,c='r')
        if len(s_picks)>0:
            for s_pick in s_picks:
                plt.axvline(s_pick,c='b')
    except Exception as e:
        print(e)

       
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0:2,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km'])))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)
    
    ax = fig.add_subplot(gs[2,0:3])


    label = augmentation_y
    if label.shape[0]==1:
        plt.plot(label[0,:],label='Label')
    if label.shape[0]==3:
        plt.plot(label[0,:],'r',linestyle='--',label='P label')
        plt.plot(label[1,:],'b',linestyle='--',label='S label')
        ax.fill_between(np.arange(0,30000), label[0,:], 0, color='r', alpha=.09,zorder=0)
        ax.fill_between(np.arange(0,30000), label[1,:], 0, color='b', alpha=.09,zorder=0)


    if predictions.shape[0]>1:
        plt.plot(predictions[0,:],'r',label='P Predictions')
        plt.plot(predictions[1,:],'b',label='S Predictions')
        # calculate and print binary cross entropy
        #bce = estimate_bce(label[0,:],predictions[0,:])
        plt.xlim(0,30000)
        plt.xlabel('sample points')
        #ax.legend(bbox_to_anchor=(1.04, 0.5),loc="center left",ncol=2,fontsize=10)
        #ax.legend(bbox_to_anchor=(0,1.02,1,0.2),loc="lower left",mode="expand",ncol=4,fontsize=12)
        #ax.legend(ncol=2,fontsize=14,loc="best",numpoints=3,columnspacing=1)
        #ax = fig.add_subplot(gs[2,3])
        #ax.axis('off')
        #bce_line = 'BCE = '+str(np.round(bce,4))
        #plt.text(0,0,bce_line)
        #plt.yticks([]);plt.xticks([])
    if legend:
        ax.legend(ncol=2,fontsize=14,loc="best",numpoints=3,columnspacing=1)

    try:
        if len(p_picks)>0:
            for p_pick in p_picks:
                plt.axvline(p_pick,c='r')
        if len(s_picks)>0:
            for s_pick in s_picks:
                plt.axvline(s_pick,c='b')
    except Exception as e:
        print(e)


    plt.xlim(0,30000)
    plt.xlabel('sample points')
    #plt.legend(bbox_to_anchor=(1.04, 0.5),loc="center left",ncol=2,fontsize=10)


    return fig

def get_min_max_times(example):
    """
    returns the arliest and the latest arrivals
    """
    
    p_times = np.asarray([example.attrs['Pn_arrival_sample'],example.attrs['Pg_arrival_sample'],example.attrs['P_arrival_sample']],dtype=float)
    s_times = np.asarray([example.attrs['Sn_arrival_sample'],example.attrs['Sg_arrival_sample'],example.attrs['S_arrival_sample']],dtype=float)
    #print(p_times,type(p_times),s_times,type(s_times))

    #p_times = p_times[p_times!='NaN']
    #s_times = s_times[s_times!='NaN']
    p_times = p_times[~(np.isnan(p_times))]
    s_times = s_times[~(np.isnan(s_times))]
    #print(p_times)
    #print(s_times)
    #p_times=np.asarray(p_times)
    #s_times=np.asarray(s_times)


    mintime = min(p_times)
    maxtime = max(s_times)
 
    #mintime=float(mintime)
    #maxtime=float(maxtime)
   
    return mintime,maxtime


def get_first_arrivals(example,seisbench=False):
    """
    gets the arrival samples of the first {P,Pg,Pn} and the first of {S,Sg,S}
    """
    if seisbench==True:
        first_p = example.attrs['trace_p_arrival_sample']
        first_s = example.attrs['trace_s_arrival_sample']
        return first_p,first_s

    else:
        p_times = np.asarray([example.attrs['Pn_arrival_sample'],example.attrs['Pg_arrival_sample'],example.attrs['P_arrival_sample']])
        s_times = np.asarray([example.attrs['Sn_arrival_sample'],example.attrs['Sg_arrival_sample'],example.attrs['S_arrival_sample']])

        p_times = p_times[p_times!='NaN']
        s_times = s_times[s_times!='NaN']

        first_p = min(p_times)
        first_s = min(s_times)
    
        return first_p,first_s


def create_mixed_multiplet(examples,label_type='traingle',half_width=500,mix_type=None):
    """
    creates augmented data by mixing waveforms from different examples
    """
    
    labels      = []
    sptimes     = []
    minmaxtimes = []
    for example in examples:
        label = generate_triangle_label(example,half_width=half_width)
        labels.append(label)
        mintime,maxtime = get_min_max_times(example)
        mintime   = np.asarray(mintime);maxtime=np.asarray(maxtime)
        mintime   = float(mintime);maxtime=float(maxtime)
        minmaxtimes.append((mintime,maxtime))

        sptime    = maxtime-mintime
        sptimes.append(sptime)

    # assemble non overlap, all short duration signals
    shifts = []
    shift            = np.random.randint(7000,9000)
    amplitude_factor = np.random.uniform(low=0.5,high=2)
    x = examples[0][()] + np.roll(examples[1][()],shift)*amplitude_factor
    y = labels[0] + np.roll(labels[1],shift,axis=1)

    if len(examples)==3:
        shift1 = np.random.randint(7000,9000)
        shift2 = np.random.randint(15000,16000)
        amplitude_factor1 = np.random.uniform(low=0.5,high=2)
        amplitude_factor2 = np.random.uniform(low=0.5,high=2)
        x = examples[0][()] + np.roll(examples[1][()],shift1,axis=1)*amplitude_factor1 + np.roll(examples[2][()],shift2,axis=1)*amplitude_factor2
        y = labels[0] + np.roll(labels[1],shift1,axis=1) + np.roll(labels[1],shift2,axis=1)


    # assemble long + short example with one quake inside the other
    if mix_type=='short_inside':
        # one possiblity
        random_start     = np.random.uniform(low=0.35,high=0.6)
        amplitude_factor = np.random.uniform(low=0.24,high=4)
        desired_start    = (minmaxtimes[0][1] - minmaxtimes[0][0])*random_start +  minmaxtimes[0][0]        
        shift1 = int(desired_start - minmaxtimes[1][0])
        x = examples[0][()] + np.roll(examples[1][()],shift1,axis=1)*amplitude_factor
        y = labels[0] + np.roll(labels[1],shift1,axis=1) 

    if mix_type=='shorts_inside':
        # two short quakes inside a long one
        amplitude_factors = np.random.uniform(low=0.5,high=2,size=2)
        random_start1     = np.random.uniform(low=0.18,high=0.35)
        random_start2     = np.random.uniform(low=0.6,high=0.75)
        desired_start1    = (minmaxtimes[0][1] - minmaxtimes[0][0])*random_start1 +  minmaxtimes[0][0]         
        desired_start2    = (minmaxtimes[0][1] - minmaxtimes[0][0])*random_start2 +  minmaxtimes[0][0] 
        shift1 = int(desired_start1 - minmaxtimes[1][0])
        shift2 = int(desired_start2 - minmaxtimes[2][0])
        x = examples[0][()] + np.roll(examples[1][()],shift1,axis=1)*amplitude_factors[0] + np.roll(examples[2][()],shift2,axis=1)*amplitude_factors[1]
        y = labels[0] + np.roll(labels[1],shift1,axis=1) + np.roll(labels[2],shift2,axis=1)

    if mix_type=='short_outside':
        desired_start    = np.random.randint(1000,3000)
        amplitude_factor = np.random.uniform(low=0.24,high=4)
        shift1           = -1*int(minmaxtimes[1][0] - desired_start)
        x = examples[0][()] + np.roll(examples[1][()],shift1,axis=1)*amplitude_factor
        y = labels[0] + np.roll(labels[1],shift1,axis=1)

    if mix_type=='mid_ppss':
        desired_start    = (minmaxtimes[0][1] - minmaxtimes[0][0])/2  + minmaxtimes[0][0]
        shift1            = np.random.randint(int(desired_start - minmaxtimes[1][0])-1000,int(desired_start - minmaxtimes[1][0])+1000)
        amplitude_factor = np.random.uniform(low=0.25,high=4)
        x = examples[0][()] + np.roll(examples[1][()],shift1,axis=1)*amplitude_factor
        y = labels[0] + np.roll(labels[1],shift1,axis=1)

    if mix_type=='asmany':
        x = examples[0][()]
        y = labels[0]
        
        n_examples=len(examples)
        shifts = [0]
        for i in range(0,n_examples-1):
            shift         = np.random.randint(2000,3000)
            desired_start = minmaxtimes[i][1] + shift + shifts[-1]
            shift =  int(desired_start - minmaxtimes[i+1][0])

            shifts.append(shift)
        

        amplitude_factors = np.random.uniform(low=0.5,high=2,size=n_examples)
        for example,label,shift,amplitude_factor in zip(examples[1:],labels[1:],shifts[1:],amplitude_factors[1:]):
    
            x += np.roll(example[()],shift,axis=1)*amplitude_factor
            y += np.roll(label,shift,axis=1)
    return x,y
    
def prepare_mixes(df):
    long_names  = df[(df['path_ep_distance_deg']>12)]['example_name'].to_list()
    short_names = df[(df['path_ep_distance_deg']>1)&(df['path_ep_distance_deg']<1.1)]['example_name'].to_list()
    mid_names   = df[(df['path_ep_distance_deg']>6)&(df['path_ep_distance_deg']<8)]['example_name'].to_list()
    
    return short_names,mid_names,long_names

def get_examples_from_names(names,data):
    examples=[]
    for name in names:
        examples.append(data[name])
    return examples

def prepare_complex_batch(data,metadata,batch_size=32,half_width=200):
    
    # filter based on snr to create the most complex examples
    snr_threshold=4
    metadata['snr'] = metadata['p_rms']/metadata['noise_rms']
    tdf             = metadata[metadata['snr']>snr_threshold]
    
    temp_x     = []
    temp_y     = []
    temp_names = []
    
    # especailly for the very short ones in 'asmany' mix mode
    
    n_asmany    = 3 # how many examples to prepare
    short_names,mid_names,long_names = prepare_mixes(tdf)
    short_names = np.asarray(short_names)
    mid_names   = np.asarray(mid_names)
    long_names  = np.asarray(long_names)
    
    for i in range(n_asmany):
    
        indices     = np.random.permutation(len(short_names))[:5]
        tnames      = short_names[indices]
        texamples   = get_examples_from_names(tnames,data)
        x,y         = create_mixed_multiplet(texamples,mix_type='asmany',half_width=half_width)
        temp_x.append(x)
        temp_y.append(y)
        temp_names.append(tnames)
        
    # make one of each type of example for: short-long combinations
    longname      = long_names[np.random.permutation(len(long_names))[0]]
    shortnames    = short_names[np.random.permutation(len(short_names))[:2]]
    
    longexample   = get_examples_from_names([longname],data)
    shortexamples = get_examples_from_names(shortnames,data)
    
    x,y = create_mixed_multiplet([longexample[0],shortexamples[0]],mix_type='short_outside',half_width=half_width)
    temp_x.append(x)
    temp_y.append(y)
    
    x,y = create_mixed_multiplet([longexample[0],shortexamples[1]],mix_type='short_inside',half_width=half_width)
    temp_x.append(x)
    temp_y.append(y)
    
    x,y = create_mixed_multiplet([longexample[0],shortexamples[0],shortexamples[1]],mix_type='shorts_inside',half_width=half_width)
    temp_x.append(x)
    temp_y.append(y)
    
    # add some ppss examples
    
    indices     = np.random.permutation(len(mid_names))[:2]
    midnames    = mid_names[indices]
    midexamples = get_examples_from_names(midnames,data)
    x,y         = create_mixed_multiplet(midexamples,mix_type='mid_ppss',half_width=half_width)
    temp_x.append(x)
    temp_y.append(y)
    
    # cast temp_x and temp_y into torch tensors
    X = torch.zeros((len(temp_x),3,30000))
    Y = torch.zeros((len(temp_x),3,30000))
    for i in range(len(temp_x)):
        X[i,:,:] = torch.from_numpy(temp_x[i])
        Y[i,:,:] = temp_y[i]

    Y[:,2,:] = 1 - Y[:,0,:]-Y[:,1,:]
    
    # add regular augmented data
    Xr,Yr,names = assemble_augmented_batch(data,batch_size,random=True,label_type='triangle',half_width=half_width)
    Xr = torch.from_numpy(Xr)
    Yr = torch.from_numpy(Yr)
    
    # add noise
    X_noise = create_noise_batch(7,(30000,3))
    X_noise = torch.from_numpy(X_noise)
    X_noise = torch.transpose(X_noise,1,2)
    Y_temp  = torch.zeros((7,3,30000))
    Y_temp[:,2,:] = 1
    
    for i in range(X_noise.shape[0]):
        X_noise[i,0,:] = X_noise[i,0,:] - torch.mean(X_noise[i,0,:])
        X_noise[i,1,:] = X_noise[i,1,:] - torch.mean(X_noise[i,1,:])
        X_noise[i,2,:] = X_noise[i,2,:] - torch.mean(X_noise[i,2,:])


    # stack the three tensors
    X = torch.cat((X,Xr,X_noise),dim=0)
    Y = torch.cat((Y,Yr,Y_temp),dim=0)
    
    return X,Y


def create_multiplet(example,label_type='triangle',half_width=500):
    """
    creates an augmentation sample given an example by copying shifted versions
    """
    #make sure the s-p is short enough so multiple copies will fit
    
    # how to decide netween two or three copies?
    #need to identify earliest and latest arrival
    #print(example.name)
    mintime,maxtime = get_min_max_times(example)
    mintime=np.asarray(mintime);maxtime=np.asarray(maxtime)

    #print('mintime',mintime,'maxtime',maxtime)
    #print('lens',len(mintime),len(maxtime))
    #if len(mintime)>1:
    #    mintime=mintime[0]
    #if len(maxtime)>1:
    #    maxtime=maxtime[0]

    mintime=float(mintime);maxtime=float(maxtime)
    #print(example.name)
    #print('actual time',mintime,maxtime)
    #print(type(mintime),type(maxtime))

    if label_type=='box':
        label = create_ps_box_label(example)
    if label_type=='triangle':
        label = generate_triangle_label(example,half_width=half_width)
    #print(label_type)
    # if maxtime is less thatn 8000, it would definely fit 3 copies
    # draw a random number to decide if two or three copies
 
    sptime = maxtime-mintime   
    
    if maxtime < 11000:
        number = np.random.uniform()
        if number < 0.1:
            # two copies
            shift = np.random.randint(low=10000,high=15000)
            # with some probability flip polarity
            flip = np.random.uniform()
            if flip < 0.5:
                x = example[()] + np.roll(example[()],shift)
            else:
                x = example[()] + np.roll(example[()],shift)*-1
            y = label + np.roll(label,shift,axis=1)
            if label_type=='triangle':
                y[2,:]=1-y[1,:]-y[0,:]           
            return x,y
        else:
            # three copies 
            min1 = (maxtime-mintime)*2
            max1 = (maxtime-mintime)*2 + 6000
            
            min2 = (maxtime-mintime)*2 + 6000 + (maxtime-mintime)*2
            max2 = min2 + 3000
            
            shift1 = np.random.randint(low=min1,high=max1)
            shift2 = np.random.randint(low=min2,high=max2)
            #print(min1,max1)
            #print(min2,max2)
            #print(shift1,shift2)
            
            x = example[()] + np.roll(example[()],shift1) + np.roll(example[()],shift2)*-1
            y = label + np.roll(label,shift1,axis=1) +  np.roll(label,shift2,axis=1)
 
            if label_type=='triangle':
                y[2,:]=1-y[1,:]-y[0,:]
            return x,y
            
    elif (maxtime>9000)&(maxtime<15000):
        #just two copies
        #mins = (maxtime-mintime)*2.5
        #maxs = 30000-maxtime- ((maxtime-mintime)*1.5)
        mins = (maxtime-mintime)*2
        maxs = (maxtime-mintime)*3
        #print('mins',mins,maxs)
        # random polarity shift        
        flip = np.random.uniform()
        shift = np.random.randint(low=int(mins),high=int(maxs))
        if flip < 0.5:
            x = example[()] + np.roll(example[()],shift)
        else:
            x = example[()] + np.roll(example[()],shift)*-1

        y = label + np.roll(label,shift,axis=1)
        if label_type=='triangle':
             y[2,:]=1-y[1,:]-y[0,:]
        
        return x,y
    
    #if the s-p time is too long, do not copy
    else:
        return example,label
    
 
def assemble_augmented_batch(data,n,random=True,label_type='triangle',half_width=500):
    """
    n examples from data will be augmented by stitching together copies of the examples with themselves
    
    """
    X = np.zeros((n,3,30000))
    Y = np.zeros((n,1,30000))
    if label_type=='triangle':
        Y =np.zeros((n,3,30000))
    outnames = []    

    #draw a random set of indices
    indices = np.random.choice(len(data),n,replace=False)
    names = list(data.keys())
    for j,index in enumerate(indices):
        
        try:
            example = data[names[index]]
            x,y = create_multiplet(example,label_type=label_type,half_width=half_width)
            outnames.append(names[index])        

            X[j,:,:] = x
            Y[j,:,:] = y
        
        except Exception as e:
            X[j,:,:] = example
            Y[j,:,:] = create_ps_box_label(example)
            if label_type=='triangle':
                Y[j,:,:] = generate_triangle_label(example,half_width=half_width)
            outnames.append(names[index])
    return X,Y,outnames
        
       
def plot_noise_example(data):
    """
    plots the data from a noise example
    """ 
    traces = data
    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(3):
        x = np.arange(0,300,0.01)
        plt.plot(traces[:,i]-i*2,linewidth=0.5,c='k')

    plt.xlim(0,30000)
    plt.yticks([])
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title('noise example')

    return fig

def estimate_bce(y_true,y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -(1/y_pred.shape[0])*np.sum((y_true*np.log10(y_pred)) + ((1-y_true)*np.log10(1-y_pred)))


def generate_gaussian(mean,std=100,length=30000):
    """
    generates truncated gaussian labels
    """
    x_values = np.arange(0,length)
    y_values = scipy.stats.norm(mean,std)
    y_values = y_values.pdf(x_values)
    #re-escale to peak at one
    y_values = y_values/np.max(y_values)
    # truncate at 2 sigmas
    y_truncated = np.zeros((30000))
    y_truncated[int(mean-2*std):int(mean+2*std)] = y_values[int(mean-2*std):int(mean+2*std)]

    return y_truncated

def generate_triangle(position,half_width=500,length=30000):
    
    p=int(position)
    #print(p,position)
    y_values           = torch.zeros((1,length))
    y_values[0,int(position)] = 1
    y_values[0,p-half_width+1:p+1] = torch.linspace(0,1,half_width)   
    y_values[0,p:p+half_width]     = torch.linspace(1,0,half_width)

    
    return y_values

def create_phasenet_label(example,std=100):
    """
    creates a Phsenet style label, 3 channels of P arrival, S and noise
    
    """
    # find the right times:
    p_times = np.asarray([example.attrs['Pn_arrival_sample'],example.attrs['Pg_arrival_sample'],example.attrs['P_arrival_sample']])
    s_times = np.asarray([example.attrs['Sn_arrival_sample'],example.attrs['Sg_arrival_sample'],example.attrs['S_arrival_sample']])

    p_times = p_times[p_times!='NaN']
    s_times = s_times[s_times!='NaN']


    p_arrival = min(p_times)
    s_arrival = min(s_times)
    
    
    label = torch.zeros((3,30000))
    label[0,:] = torch.from_numpy(generate_gaussian(p_arrival,std=std))
    label[1,:] = torch.from_numpy(generate_gaussian(s_arrival,std=std))
    label[2,:] = 1 - label[0,:] - label[1,:]
    
    return label

def create_multiphase_label(example,half_width=500,length=30000):
    """
    creates labels for the 4 phases Pn,Pg,Sn and Sg 
    """
    pn = example.attrs['Pn_arrival_sample']
    pg = example.attrs['Pg_arrival_sample']
    sn = example.attrs['Sn_arrival_sample']
    sg = example.attrs['Sg_arrival_sample']

    label = torch.zeros((5,30000))
    label[0,:] = generate_triangle(pn,half_width=half_width,length=length)
    label[1,:] = generate_triangle(pg,half_width=half_width,length=length)
    label[2,:] = generate_triangle(sn,half_width=half_width,length=length)
    label[3,:] = generate_triangle(sg,half_width=half_width,length=length)
    label[4,:] = 1 - label[0,:] - label[1,:] - label[2,:] - label[3,:]

    label = torch.clamp(label,min=0,max=1)

    return label

def create_multiphase_batch(dataset,indices,half_width=500,filtered=False):


    n_examples = len(indices)
    y_batch = torch.zeros((n_examples,5,30000))
    x_batch = torch.zeros((n_examples,3,30000))

    names = []
    names = list(dataset.keys())
    for i,index in enumerate(indices):
        example    = dataset[names[index]]
        #print(example[()].shape)
        temp_label = create_multiphase_label(example,half_width=half_width)

        y_batch[i,:,:] = temp_label
        x_batch[i,:,:] = torch.from_numpy(example[()])
        names.append(list(dataset.keys())[index])

    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names



def create_phasenet_batch(dataset,indices,std,filtered=False):
    """
    creates a batch of labels from the data examples whose indices are passed,
    puts everything into a pytorch tensor
    
    """
    n_examples = len(indices)
    y_batch = torch.zeros((n_examples,3,30000))
    x_batch = torch.zeros((n_examples,3,30000))

    names = []

    for i,index in enumerate(indices):
        example    = dataset[list(dataset.keys())[index]]
        #print(example[()].shape)
        temp_label = create_phasenet_label(example,std).reshape((1,3,30000))

        y_batch[i,:,:] = temp_label

        traces = example[()]
        traces = np.copy(traces)
        #print(traces.shape,type(traces))
        if filtered:
            traces = bandpass_data(traces)
            #print(traces.shape,type(traces))

        x_batch[i,:,:] = torch.from_numpy(traces)
        names.append(list(dataset.keys())[index])

    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names

def random_phasenet_batch(dataset,n,std):
    """
    creates a list of random numbers which will serve as indices when calling
    create_ps_label_batch
    """
    n_examples = len(list(dataset.keys()))

    indices    = np.random.choice(n_examples,n,replace=False)

    x_batch,y_batch,names      = create_phasenet_batch(dataset,indices,std)

    return x_batch,y_batch,names


def generate_triangle_label(example,half_width=100,length=30000,seisbench=False):
    
    y_values = torch.zeros((3,length))
    
    p,s = get_first_arrivals(example,seisbench=seisbench)

    try:
        p = int(float(p)); s = int(float(s))
        y_values[0,p] = 1
        y_values[0,p-half_width+1:p+1] = torch.linspace(0,1,half_width)
        y_values[0,p:p+half_width]     = torch.linspace(1,0,half_width)  
    
        y_values[1,s] = 1
        y_values[1,s-half_width+1:s+1] = torch.linspace(0,1,half_width)
        if s+half_width>30000:
            y_values[1,s:]     = torch.linspace(1,0,half_width)[:30000-s]
        else:
            y_values[1,s:s+half_width]     = torch.linspace(1,0,half_width)   
    
        y_values[2,:] = 1 - y_values[0,:] - y_values[1,:]
    
        y_values = torch.clamp(y_values,min=0,max=1)    
    except Exception as e:
        # for the examples where there are waveforms but no arrivals, return labels for noise only
        y_values[2,:] = 1
        y_values[1,:] = 0
        y_values[0,:] = 0

    return y_values


def seisbench_batch_phasenet(dataset,indices,half_width):
    # check the dataset to ghet the shape
    tnames = list(dataset.keys())
    example = dataset[tnames[0]]
    shape   = example[()].shape
    n_examples = len(indices)
    x_batch = torch.zeros((n_examples,shape[0],shape[1]))
    y_batch = torch.zeros((n_examples,3,shape[1]))

    names = []
 
    for i,index in enumerate(indices):
        example    = dataset[tnames[index]]
        #print(example[()].shape)
        temp_label = generate_triangle_label(example,half_width,length=shape[1],seisbench=True).reshape((1,3,shape[1]))

        y_batch[i,:,:] = temp_label
        x_batch[i,:,:] = torch.from_numpy(example[()])
        names.append(list(dataset.keys())[index])

    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names

def batch_from_filename(filename,batch_size=20,half_width=500):
    data      = open(filename)
    len_data  = len(list(data.keys()))
    
    indices   = np.random.choice(len_data,batch_size,replace=False)
    X,Y,names = create_phasenet_t_batch(dataset=data,indices=indices,half_width=half_width)
    
    return X,Y,names

def create_phasenet_t_batch(dataset,indices,half_width):
    """
    creates a batch of labels from the data examples whose indices are passed,
    puts everything into a pytorch tensor
    
    # THIS IS FOR TRIANGLE LABELS
    """
    n_examples = len(indices)
    y_batch = torch.zeros((n_examples,3,30000))
    x_batch = torch.zeros((n_examples,3,30000))

    names = []

    for i,index in enumerate(indices):
        example    = dataset[list(dataset.keys())[index]]
        #print(example[()].shape)
        temp_label = generate_triangle_label(example,half_width).reshape((1,3,30000))

        y_batch[i,:,:] = temp_label
        x_batch[i,:,:] = torch.from_numpy(example[()])
        names.append(list(dataset.keys())[index])

    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names


def random_phasenet_t_batch(dataset,n,half_width):
    """
    creates a list of random numbers which will serve as indices when calling
    create_ps_label_batch
    with TRIANGLE LABELS
    """
    n_examples = len(list(dataset.keys()))

    indices    = np.random.choice(n_examples,n,replace=False)

    x_batch,y_batch,names      = create_phasenet_t_batch(dataset,indices,half_width)

    return x_batch,y_batch,names

def plot_phasenet_example_predictions(example,predictions=False,picks=False,filtered=False,std=100,label_type='triangle',half_width=500,threshold=0.5,legend=True):
    """
    plots the waveforms along with predicitons and desired labels
    choose from gaussian for truncated gaussian labels
    or triangle for triangle labels
    """

    traces=example[()]
    
    if filtered:
        traces = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))

    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
 
    # extract the picks
    # first if the array predictions are passed
    #print('change',predictions)
    if type(predictions)==np.ndarray:
        #predictions = np.asarray(predictions)
        #print(predictions)
        p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
        s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)
    # second, pass the picks and plot them
    if picks:
       p_picks = picks[0]
       s_picks = picks[1]

    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(len(traces)):
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(1000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])


        if example.attrs['P_arrival_sample'] != 0:
            pg_pos = float(example.attrs['P_arrival_sample'])
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='P',linewidth=0.5)
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-800,-5.2,'P',c='darkorange')
        if example.attrs['Pg_arrival_sample'] != 0:
            pg_pos = float(example.attrs['Pg_arrival_sample'])
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-800,1.2,'Pg',c='r')
        if example.attrs['Pn_arrival_sample'] != 0:
            pn_pos = float(example.attrs['Pn_arrival_sample'])
            #plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            plt.axvline(pn_pos,c='r',linestyle='--')
            if i==0:plt.text(pn_pos-800,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != 0:
            sg_pos = float(example.attrs['S_arrival_sample'])
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos+200,-5.2,'S',c='b')                
        if example.attrs['Sg_arrival_sample'] != 0:
            sg_pos = float(example.attrs['Sg_arrival_sample'])
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos,1.2,'Sg',c='b')
        if example.attrs['Sn_arrival_sample'] != 0:
            sn_pos = float(example.attrs['Sn_arrival_sample'])
            #plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            plt.axvline(sn_pos,c='b',linestyle='--')
            if i==0:plt.text(sn_pos-800,1.2,'Sn',c='b')

    # add the predicted picks
    try:
        if len(p_picks)>0:
            for p_pick in p_picks:
                plt.axvline(p_pick,c='r')
        if len(s_picks)>0:
            for s_pick in s_picks:
                plt.axvline(s_pick,c='b')     
    except Exception as e:
        print(e)
   
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,30000)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0:2,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km'])))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)

    # create phasenet label
    if label_type == 'gaussian':
        label = create_phasenet_label(example,std=std)
    if label_type == 'triangle':
        label = generate_triangle_label(example,half_width=half_width).detach().numpy()

    print('label shape',label.shape)
    ax = fig.add_subplot(gs[2,0:3])
    plt.plot(label[0,:],label='P label',c='r',linestyle='--')
    plt.plot(label[1,:],label='S label',c='b',linestyle='--')
    ax.fill_between(np.arange(0,30000), label[0,:], 0, color='r', alpha=.09,zorder=0)
    ax.fill_between(np.arange(0,30000), label[1,:], 0, color='b', alpha=.09,zorder=0)

    if type(predictions)==np.ndarray:
        plt.plot(predictions[0,:],label='P predictions',c='r')
        plt.plot(predictions[1,:],label='S predictions',c='b')

    # estimate the peaks of the predictions, which will be the picks
    #p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
    #s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)


    try:
        if len(p_picks)>0:
            for p_pick in p_picks:
                plt.axvline(p_pick,c='r')
        if len(s_picks)>0:
            for s_pick in s_picks:
                plt.axvline(s_pick,c='b')     
    except Exception as e:
        print(e)
    plt.xlim(0,30000)
    plt.ylim(-0.05,1.05)
    plt.xlabel('sample points')
    #plt.legend()
    if legend:
       ax.legend(loc='best',ncol=2,fontsize=14)#bbox_to_anchor=(0,0),ncol=2)
    #ax.legend(ncol=2)


    p,s        = get_first_arrivals(example)
    #print('first arrivals',p,s)
    p=float(p)
    s=float(s) 
   
    ax = fig.add_subplot(gs[2,3])
    ax.axis('off')
    try:
        print(p,p_picks[0])
        p_residual = (p-p_picks[0])/100
        plt.text(0,1,'P residual = '+str(np.round(np.abs(p_residual),3))+' s',color='r')
        print(s,s_picks[0])
        s_residual = (s-s_picks[0])/100
        plt.text(0,0,'S residual = '+str(np.round(np.abs(s_residual),3))+' s',color='b')
        #plt.axvline(p_picks[0],c='r')
        #plt.axvline(s_picks[0],c='b')
        plt.ylim(0,2)
    except Exception as e:
        print(e)


    return fig


def plot_seisbench_example_predictions(example,predictions=False,picks=False,filtered=False,label_type='triangle',half_width=250,threshold=0.5,legend=True):
    """
    plots the waveforms along with predicitons and desired labels
    choose from gaussian for truncated gaussian labels
    or triangle for triangle labels
    """

    traces=example[()]
    xlim = traces.shape[-1]    


    if filtered:
        traces = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))

    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
 
    # extract the picks
    # first if the array predictions are passed
    print('change',predictions)
    if type(predictions)==np.ndarray:
        #predictions = np.asarray(predictions)
        #print(predictions)
        p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
        s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)
    # second, pass the picks and plot them
    if picks:
       p_picks = picks[0]
       s_picks = picks[1]

    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(len(traces)):
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(1000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])


        if example.attrs['trace_p_arrival_sample'] != 0:
            pg_pos = example.attrs['trace_p_arrival_sample']
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-400,-5.2,'P',c='darkorange')
                
                
        if example.attrs['trace_s_arrival_sample'] != 0:
            sg_pos = example.attrs['trace_s_arrival_sample']
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos,-5.2,'S',c='b')                

    # add the predicted picks
    try:
        if len(p_picks)>0:
            for p_pick in p_picks:
                plt.axvline(p_pick,c='r')
        if len(s_picks)>0:
            for s_pick in s_picks:
                plt.axvline(s_pick,c='b')     
    except Exception as e:
        print(e)
   
    plt.ylim(-6,2)
    plt.yticks([])
    plt.xlim(0,xlim)
    plt.xticks(np.arange(0,12001,6000),np.arange(0,121,60))
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0:2,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km']/1000)))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)

    # create phasenet label
    if label_type == 'gaussian':
        label = create_phasenet_label(example,std=std)
    if label_type == 'triangle':
        label = generate_triangle_label(example,half_width=half_width,seisbench=True).detach().numpy()


    ax = fig.add_subplot(gs[2,0:3])
    plt.plot(label[0,:],label='P label',c='r',linestyle='--')
    plt.plot(label[1,:],label='S label',c='b',linestyle='--')

    if type(predictions)==np.ndarray:
        plt.plot(predictions[0,:],label='P predictions',c='r')
        plt.plot(predictions[1,:],label='S predictions',c='b')

    # estimate the peaks of the predictions, which will be the picks
    #p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
    #s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)
    #p,s        = get_first_arrivals(example,seisbench=True)
    try:
        print(p,p_picks[0])
        print(s,s_picks[0])
        #plt.axvline(p_picks[0],c='r')
        #plt.axvline(s_picks[0],c='b')
    except Exception as e:
        print(e)

    try:
        if len(p_picks)>0:
            for p_pick in p_picks:
                plt.axvline(p_pick,c='r')
        if len(s_picks)>0:
            for s_pick in s_picks:
                plt.axvline(s_pick,c='b')     
    except Exception as e:
        print(e)
    plt.xlim(0,xlim)
    plt.xlabel('sample points')
    #plt.legend()
    if legend:
       ax.legend(bbox_to_anchor=(1,1),ncol=2)
    #ax.legend(ncol=2)
    return fig

def plot_multiphase_predictions(example,predictions=False,picks=False,filtered=False,std=100,label_type='triangle',half_width=500,threshold=0.5,legend=True,labels=False,xlims=(0,30000)):
    """
    plots the waveforms along with predicitons and desired labels
    choose from gaussian for truncated gaussian labels
    or triangle for triangle labels
    """

    traces=example[()]
    
    if filtered:
        traces = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))

    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
 
    # extract the picks
    # first if the array predictions are passed
    #print('change',predictions)
    if type(predictions)==np.ndarray:
        #predictions = np.asarray(predictions)
        #print(predictions)
        pn_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
        pg_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)
        sn_picks,_ = find_peaks(predictions[2,:], distance=150,height=threshold,width=100)
        sg_picks,_ = find_peaks(predictions[3,:], distance=150,height=threshold,width=100)
    # second, pass the picks and plot them
    if picks:
       pn_picks = picks[0]
       pg_picks = picks[1]
       sn_picks = picks[2]
       sg_picks = picks[3]

    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(len(traces)):
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        plt.text(1000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        
        compkey='NaN'
        if example.attrs['P_arrival_sample'] != compkey:
            pg_pos = example.attrs['P_arrival_sample']
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='P',linewidth=0.5)
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-800,-5.2,'P',c='red')
        if example.attrs['Pg_arrival_sample'] != compkey:
            pg_pos = example.attrs['Pg_arrival_sample']
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            plt.axvline(pg_pos,c='purple',linestyle='--')
            if i==0:plt.text(pg_pos+100,1.2,'Pg',c='purple')
        if example.attrs['Pn_arrival_sample'] != compkey:
            pn_pos = example.attrs['Pn_arrival_sample']
            #plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            plt.axvline(pn_pos,c='r',linestyle='--')
            if i==0:plt.text(pn_pos-1200,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != compkey:
            sg_pos = example.attrs['S_arrival_sample']
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos+500,-5.2,'S',c='blue')                
        if example.attrs['Sg_arrival_sample'] != compkey:
            sg_pos = example.attrs['Sg_arrival_sample']
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            plt.axvline(sg_pos,c='dodgerblue',linestyle='--')
            if i==0:plt.text(sg_pos+100,1.2,'Sg',c='dodgerblue')
        if example.attrs['Sn_arrival_sample'] != compkey:
            sn_pos = example.attrs['Sn_arrival_sample']
            #plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            plt.axvline(sn_pos,c='b',linestyle='--')
            if i==0:plt.text(sn_pos-1200,1.2,'Sn',c='b')

    # add the predicted picks
    try:
        if len(pn_picks)>0:
            for p_pick in pn_picks:
                plt.axvline(p_pick,c='r')
        if len(sn_picks)>0:
            for s_pick in sn_picks:
                plt.axvline(s_pick,c='b')   

        if len(pg_picks)>0:
            for p_pick in pg_picks:
                plt.axvline(p_pick,c='purple')
        if len(sg_picks)>0:
            for s_pick in sg_picks:
                plt.axvline(s_pick,c='dodgerblue')   

  
    except Exception as e:
        print(e)
   
    plt.ylim(-6,2)
    plt.yticks([])
    #plt.xlim(xlims)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlim(xlims)
    plt.xlabel('seconds')
    plt.title(example.name)
    
    
    ax = fig.add_subplot(gs[0:2,3])
    ax.axis('off')
    
    id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    
    plt.text(0,1,id_)
    plt.text(0,0,example.attrs['source_origin_time'])
    plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km']/1000)))
    plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    if filtered:
        plt.text(0,-5.9,'Bandpassed',color='red')

    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)


    ax = fig.add_subplot(gs[2,0:3])

    if labels:
        # create phasenet label
        if label_type == 'gaussian':
            label = create_phasenet_label(example,std=std)
        if label_type == 'triangle':
            #label = generate_triangle_label(example,half_width=half_width).detach().numpy()
            label = create_multiphase_label(example,half_width=half_width,length=30000).detach().numpy()

        #ax = fig.add_subplot(gs[2,0:3])
        #plt.plot(label[0,:],label='P label',c='r',linestyle='--')
        #plt.plot(label[1,:],label='S label',c='b',linestyle='--')
        #'_nolegend_'
        plt.plot(label[0,:],label='_nolegend_',c='r',linestyle='--')
        plt.plot(label[1,:],label='_nolegend_',c='purple',linestyle='--')

        plt.plot(label[2,:],label='_nolegend_',c='b',linestyle='--')
        plt.plot(label[3,:],label='_nolegend_',c='dodgerblue',linestyle='--')    

    if type(predictions)==np.ndarray:
        #plt.plot(predictions[0,:],label='P predictions',c='r')
        #plt.plot(predictions[1,:],label='S predictions',c='b')
        plt.plot(predictions[0,:],label='Pn predictions',c='r')
        plt.plot(predictions[1,:],label='Pg predictions',c='purple')
        plt.plot(predictions[2,:],label='Sn predictions',c='b')
        plt.plot(predictions[3,:],label='Sg predictions',c='dodgerblue')

    # estimate the peaks of the predictions, which will be the picks
    #p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
    #s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)

    p,s        = get_first_arrivals(example)
    try:
        print(p,p_picks[0])
        print(s,s_picks[0])
        #plt.axvline(p_picks[0],c='r')
        #plt.axvline(s_picks[0],c='b')
    except Exception as e:
        print(e)

    try:
        if len(pn_picks)>0:
            for p_pick in pn_picks:
                plt.axvline(p_pick,c='r')
        if len(sn_picks)>0:
            for s_pick in sn_picks:
                plt.axvline(s_pick,c='b')     
        if len(pg_picks)>0:
            for p_pick in pg_picks:
                plt.axvline(p_pick,c='purple')
        if len(sg_picks)>0:
            for s_pick in sg_picks:
                plt.axvline(s_pick,c='dodgerblue')     
    except Exception as e:
        print(e)
    plt.xlim(xlims)
    plt.ylim(-0.05,1.05)
    plt.xlabel('sample points')
    #plt.legend()
    if legend:
       ax.legend(bbox_to_anchor=(1,1),ncol=2,fontsize=14)
    #ax.legend(ncol=2)
    return fig

def plot_multiphase_predictions_from_stream(st,predictions=False,picks=False,filtered=False,std=100,label_type='triangle',half_width=500,threshold=0.5,legend=True,labels=False,xlims=(0,30000)):
    """
    plots the waveforms along with predicitons from streams of data, this one is for 5 minutes only
    no labels
    """

    traces = np.zeros((3,30000))
    st = st.sort() # get the Z channel last
    traces[0,:] = st[0].data[:30000] 
    traces[1,:] = st[1].data[:30000]
    traces[2,:] = st[2].data[:30000]
    traces = traces/np.max(np.abs(traces))

    if filtered:
        traces = bandpass_data(traces)
        traces = traces / np.max(np.abs(traces))

    fig = plt.figure(figsize=(16,6),tight_layout=True)
    gs  = gridspec.GridSpec(3,4)
 
    # extract the picks
    # first if the array predictions are passed
    #print('change',predictions)
    if type(predictions)==np.ndarray:
        #predictions = np.asarray(predictions)
        #print(predictions)
        pn_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
        pg_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)
        sn_picks,_ = find_peaks(predictions[2,:], distance=150,height=threshold,width=100)
        sg_picks,_ = find_peaks(predictions[3,:], distance=150,height=threshold,width=100)
    # second, pass the picks and plot them
    if picks:
       pn_picks = picks[0]
       pg_picks = picks[1]
       sn_picks = picks[2]
       sg_picks = picks[3]

    ax = fig.add_subplot(gs[0:2,0:3])
    for i in range(len(traces)):
        plt.plot(traces[i]-i*2,linewidth=0.5,c='k')
        #plt.text(1000,-i*2+0.3,example.attrs['channels'][i],bbox=dict(boxstyle="round",fc='white'))#+channels[i])
        """
        compkey='NaN'
        if example.attrs['P_arrival_sample'] != compkey:
            pg_pos = example.attrs['P_arrival_sample']
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='P',linewidth=0.5)
            plt.axvline(pg_pos,c='r',linestyle='--')
            if i==0:plt.text(pg_pos-800,-5.2,'P',c='red')
        if example.attrs['Pg_arrival_sample'] != compkey:
            pg_pos = example.attrs['Pg_arrival_sample']
            #plt.scatter(pg_pos,-2*i,s=4000,c='r',marker='|',label='Pg',linewidth=0.5)
            plt.axvline(pg_pos,c='purple',linestyle='--')
            if i==0:plt.text(pg_pos+100,1.2,'Pg',c='purple')
        if example.attrs['Pn_arrival_sample'] != compkey:
            pn_pos = example.attrs['Pn_arrival_sample']
            #plt.scatter(pn_pos,-2*i,s=4000,c='r',marker='|',label='Pn') 
            plt.axvline(pn_pos,c='r',linestyle='--')
            if i==0:plt.text(pn_pos-1200,1.2,'Pn',c='r')
                
                
        if example.attrs['S_arrival_sample'] != compkey:
            sg_pos = example.attrs['S_arrival_sample']
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='S')
            plt.axvline(sg_pos,c='b',linestyle='--')
            if i==0:plt.text(sg_pos+500,-5.2,'S',c='blue')                
        if example.attrs['Sg_arrival_sample'] != compkey:
            sg_pos = example.attrs['Sg_arrival_sample']
            #plt.scatter(sg_pos,-2*i,s=4000,c='b',marker='|',label='Sg')
            plt.axvline(sg_pos,c='dodgerblue',linestyle='--')
            if i==0:plt.text(sg_pos+100,1.2,'Sg',c='dodgerblue')
        if example.attrs['Sn_arrival_sample'] != compkey:
            sn_pos = example.attrs['Sn_arrival_sample']
            #plt.scatter(sn_pos,-2*i,s=4000,c='b',marker='|',label='Sn')
            plt.axvline(sn_pos,c='b',linestyle='--')
            if i==0:plt.text(sn_pos-1200,1.2,'Sn',c='b')
        """
    # add the predicted picks
    try:
        if len(pn_picks)>0:
            for p_pick in pn_picks:
                plt.axvline(p_pick,c='r')
        if len(sn_picks)>0:
            for s_pick in sn_picks:
                plt.axvline(s_pick,c='b')   

        if len(pg_picks)>0:
            for p_pick in pg_picks:
                plt.axvline(p_pick,c='purple')
        if len(sg_picks)>0:
            for s_pick in sg_picks:
                plt.axvline(s_pick,c='dodgerblue')   

  
    except Exception as e:
        print(e)
   
    plt.ylim(-6,2)
    plt.yticks([])
    #plt.xlim(xlims)
    plt.xticks(np.arange(0,30001,6000),np.arange(0,301,60))
    plt.xlim(xlims)
    plt.xlabel('seconds')
    ttitle = st[0].id[:-1]+'?'+'   '+ str(st[0].stats.starttime)

    plt.title(ttitle)
    
    
    ax = fig.add_subplot(gs[0:2,3])
    ax.axis('off')
    
    #id_ = example.attrs['station_network_code']+'.'+example.attrs['station_code']+'..'+example.attrs['channels'][0][:2]+'*'
    #id_ = st[0].id
    #plt.text(0,1,id_)
    #plt.text(0,-2,str(st[0].stats.starttime))
    #plt.text(0,0,example.attrs['source_origin_time'])
    #plt.text(0,-1,'Origin Longitude = '+str(np.round(example.attrs['source_longitude_deg'],2)))
    #plt.text(0,-2,'Origin Latitude    = '+str(np.round(example.attrs['source_latitude_deg'],2)))
    #plt.text(0,-3,'Origin Depth       = '+str(np.round(example.attrs['source_depth_km']/1000)))
    #plt.text(0,-4,'Magnitude          = '+str(np.round(example.attrs['source_magnitude'],1)))
    #plt.text(0,-5,'Distance             = '+str(np.round(example.attrs['path_ep_distance_deg'],2)) )
    
    if filtered:
        plt.text(0,-5.9,'Bandpassed',color='red')

    plt.yticks([]);plt.xticks([])
    plt.ylim(-6,2)


    ax = fig.add_subplot(gs[2,0:3])

    if labels:
        # create phasenet label
        if label_type == 'gaussian':
            label = create_phasenet_label(example,std=std)
        if label_type == 'triangle':
            #label = generate_triangle_label(example,half_width=half_width).detach().numpy()
            label = create_multiphase_label(example,half_width=half_width,length=30000).detach().numpy()

        #ax = fig.add_subplot(gs[2,0:3])
        #plt.plot(label[0,:],label='P label',c='r',linestyle='--')
        #plt.plot(label[1,:],label='S label',c='b',linestyle='--')
        #'_nolegend_'
        plt.plot(label[0,:],label='_nolegend_',c='r',linestyle='--')
        plt.plot(label[1,:],label='_nolegend_',c='purple',linestyle='--')

        plt.plot(label[2,:],label='_nolegend_',c='b',linestyle='--')
        plt.plot(label[3,:],label='_nolegend_',c='dodgerblue',linestyle='--')    

    if type(predictions)==np.ndarray:
        #plt.plot(predictions[0,:],label='P predictions',c='r')
        #plt.plot(predictions[1,:],label='S predictions',c='b')
        plt.plot(predictions[0,:],label='Pn predictions',c='r')
        plt.plot(predictions[1,:],label='Pg predictions',c='purple')
        plt.plot(predictions[2,:],label='Sn predictions',c='b')
        plt.plot(predictions[3,:],label='Sg predictions',c='dodgerblue')

    # estimate the peaks of the predictions, which will be the picks
    #p_picks,_ = find_peaks(predictions[0,:], distance=150,height=threshold,width=100) 
    #s_picks,_ = find_peaks(predictions[1,:], distance=150,height=threshold,width=100)

    #p,s        = get_first_arrivals(example)
    try:
        print(p,p_picks[0])
        print(s,s_picks[0])
        #plt.axvline(p_picks[0],c='r')
        #plt.axvline(s_picks[0],c='b')
    except Exception as e:
        print(e)

    try:
        if len(pn_picks)>0:
            for p_pick in pn_picks:
                plt.axvline(p_pick,c='r')
        if len(sn_picks)>0:
            for s_pick in sn_picks:
                plt.axvline(s_pick,c='b')     
        if len(pg_picks)>0:
            for p_pick in pg_picks:
                plt.axvline(p_pick,c='purple')
        if len(sg_picks)>0:
            for s_pick in sg_picks:
                plt.axvline(s_pick,c='dodgerblue')     
    except Exception as e:
        print(e)
    plt.xlim(xlims)
    plt.ylim(-0.05,1.05)
    plt.xlabel('sample points')
    #plt.legend()
    if legend:
       ax.legend(bbox_to_anchor=(1,1),ncol=2,fontsize=14)
    #ax.legend(ncol=2)
    return fig


def to_obspy(example):
    """
    casts the example into an obspy strea, only the traces and station metadata, no arrival metadata
    """
    st = Stream()
    for i in range(3):
        trace = Trace(data=example[()][i,:])
        trace.stats.station = example.attrs['station_code']
        trace.stats.network = example.attrs['station_network_code']
        trace.stats.channel = example.attrs['channels'][i]
        trace.stats.sampling_rate = 100.
        trace.stats.starttime = example.attrs['trace_start_time']

        st += trace
    return st


def create_picks_batch(dataset,indices):
    """
    creates a batch of X,Y where X are the ususal waveforms,
    and Y are the earliest p and s arrivals
    X is of shape (n,3,30000)
    Y is of shape (n,2)
    """
    n_examples = len(indices)
    y_batch = torch.zeros((n_examples,2))
    x_batch = torch.zeros((n_examples,3,30000))

    names = []

    for i,index in enumerate(indices):
        example    = dataset[list(dataset.keys())[index]]
        p,s        = get_first_arrivals(example)
        y_batch[i,0] = p
        y_batch[i,1] = s
        x_batch[i,:,:] = torch.from_numpy(example[()])
        names.append(list(dataset.keys())[index])

    #print(x_batch.shape,y_batch.shape)

    return x_batch,y_batch,names
"""
usage examples
x = create_complex_steppy_noise(size=(30000,3))
plt.plot(x[:,0])


x=create_noise_with_spikes((30000,3),n_spikes=200,max_amplitude=50)
x += create_noise_sinusoid(x.shape,noise_intensity=10,add_noise=True)
x = add_vertical_shift(x,0.3)

"""


def predict_from_stream(st,model,overlap=0.2,threshold=0.5,normalize=True,stack=False):
    """
    run predictions on obspy streams with an sliding window
    st should be sampled at 100 Hz

    when single station streams are passed, the channels are duplicated to make three channels

    """
    if len(st)==1:
        data = np.stack((st[0].data,st[0].data,st[0].data),axis=0)
    elif len(st)==2:
        data = np.stack((st[0].data,st[1].data,st[1].data),axis=0)
    else:
        data = np.stack((st[0].data,st[1].data,st[2].data),axis=0)    
    data = torch.from_numpy(data)
    print('data shape ',data.shape)
    n_samples   = len(st[0].data)
    print('n_samples  ', n_samples)
    slide_step  = int(overlap*30000)
    step_size   = int(30000 - (30000*overlap))
    n_steps     = int(np.ceil(n_samples/step_size))
    X           = torch.zeros((n_steps,3,30000))
   
    if n_samples==30000:
        print('only one window')
        X           = torch.zeros((1,3,30000))
        X[0,:,:]    = data
        X[0,:,:] = X[0,:,:] / torch.max(torch.abs(X[0,:,:]))


    else:

        for i,step in enumerate(range(0,n_samples,step_size)):
            if step<n_samples-30000:
                X[i,:,:]=data[:,step:step+30000]
                if normalize:
                    X[i,:,:] = X[i,:,:] / torch.max(torch.abs(X[i,:,:]))     
            else:
                X[i,:,0:n_samples%step]=data[:,step:step+n_samples%step]
                if normalize:
                    X[i,:,:] = X[i,:,:] / torch.max(torch.abs(X[i,:,:]))
     

    predictions = model(X)
    print('predictions done') 
    if not stack:
        return predictions
    if stack:
        #predictions = predictions.detach().numpy()
        return stack_predictions(predictions,n_samples,overlap)

def stack_predictions(predictions,n_samples,overlap=0.2):
    """
    creates an array that does not need to use np.sum, asign predictions and normalize inside of a loop
    one prediction at a time
    """
    predictions = predictions.detach().numpy()
    step_size   = int(30000 - (30000*overlap))
    n_steps     = int(np.ceil(n_samples/step_size))
    length      = (n_steps)*step_size + 30000
    n_steps     = int(np.ceil(n_samples/step_size))
    out         = np.zeros((predictions.shape[1],length))
    print(out.shape)
    for i,step in enumerate(range(0,n_samples,step_size)):
        #print(step)
        out[:,step:step+30000]+=predictions[i,:,:] 
        # normalize over overlap indices
        overlap_start = int(step + overlap*30000)
        overlap_end   = int(step + 30000)
        #out[:,overlap_start:overlap_end]= out[:,overlap_start:overlap_end]/2
    return out[:,:n_samples]
    


    
def picks_to_df(tst,picks):
    """
    take the picks, create a dataframe and write it
    """
    picks    = picks[0]
    sta_name = tst[0].stats.station
    net_name = tst[0].stats.network
    channels = [tr.stats.channel for tr in tst]
    print(channels)
    reftime  = tst[0].stats.starttime
    p_times  = [reftime + val for val in np.asarray(picks[0][0])/100]
    s_times  = [reftime + val for val in np.asarray(picks[1][0])/100]
    p_scores = picks[0][1]
    s_scores = picks[1][1]
    p_df     = pd.DataFrame({'network':net_name,'station':sta_name,'time':p_times,'score':p_scores,'phase':'P'})#,'channels':channels})
    s_df     = pd.DataFrame({'network':net_name,'station':sta_name,'time':s_times,'score':s_scores,'phase':'S'})#,'channels':channels})
    
    df = pd.concat((p_df,s_df))
    return df



def sanity_check_stream(st,mode='fill'):
    """
    this is to make sure the number of sample points on each channel is the same before casting it into torch
    mode is either fill, or crop
    crop cuts the waveforms between the max starttime and the min endtime among the channels
    fill pads the wavefroms between the min starttime and the max endtime
    """
    n_req = max([tr.stats.npts for tr in st])
    min_starttime = min([tr.stats.starttime for tr in st])
    max_starttime = max([tr.stats.starttime for tr in st])
    min_endtime   = min([tr.stats.endtime for tr in st])
    max_endtime   = max([tr.stats.endtime for tr in st])
    
    if mode=='fill':
        st = st.trim(starttime=min_starttime,endtime=max_endtime,pad=True,fill_value=st[0].data[-1])
    if mode=='crop':
        st = st.trim(starttime=max_starttime,endtime=min_endtime)
    
    return st

def execute(st,model,outname='skynet_picks.csv',threshold=0.5,stack=True,return_preds=False,overlap=0.2):
    stations = list(set([tr.stats.station for tr in st]))
    dfs = []
    all_preds = []
    for station in stations:
        try:
         
            print('Working on ',station)
            tst = st.select(station=station)
            tst = sanity_check_stream(tst)
            tst = tst.sort()
            predictions = predict_from_stream(tst,model,overlap=overlap,
                                                     normalize=True,stack=stack)
            if return_preds:
                all_preds.append(predictions)

            #picks = extract_picks(predictions.detach().numpy(),threshold=threshold)
            picks = extract_picks(predictions,threshold=threshold)
            # this is not passing the pick scores, why?
            #print(picks)
            tdf = picks_to_df(tst,picks)
            print(tdf)
            dfs.append(tdf)
        except Exception as e:
            print(e)
    df = pd.concat(dfs)
    df.to_csv(outname,index=False)
    print('Saved results in ',outname)
    #if return_preds:
    #    return predictions


def extract_picks(predictions,threshold=0.5):
    """
    Takes the tensor from running the network predictions and extract the peaks, which will be the picks
    """
    picks = []
    
    if len(predictions.shape)==3:
        for i in range(predictions.shape[0]):

            preds    = predictions[i,:,:]
            #[i,:,:].detach().numpy()
            p_preds  = preds[0,:]
            s_preds  = preds[1,:]
            #p_picks,_  = find_peaks(p_preds, distance=500,height=threshold,width=100)
            #s_picks,_  = find_peaks(s_preds, distance=500,height=threshold,width=100)
            p_picks  = find_peaks(p_preds, distance=500,height=threshold,width=100)
            s_picks  = find_peaks(s_preds, distance=1200,height=threshold,width=100)
            p_ = [p_picks[0],p_picks[1]['peak_heights']]
            s_ = [s_picks[0],s_picks[1]['peak_heights']]
            picks.append((p_,s_))

            #picks.append((p_picks,s_picks))
        return picks
    
    if len(predictions.shape)==2:
        preds    = predictions#[i,:,:].detach().numpy()
        p_preds  = preds[0,:]
        s_preds  = preds[1,:]        
        p_picks  = find_peaks(p_preds, distance=500,height=threshold,width=100)
        s_picks  = find_peaks(s_preds, distance=500,height=threshold,width=100)
        # pass only the postions and heights of the picks
        p_ = [p_picks[0],p_picks[1]['peak_heights']]
        s_ = [s_picks[0],s_picks[1]['peak_heights']]
        
        picks.append((p_,s_))
        return picks



def picks_to_df(tst,picks):
    """
    take the picks, create a dataframe and write it
    """
    picks    = picks[0]
    sta_name = tst[0].stats.station
    net_name = tst[0].stats.network
    channels = [tr.stats.channel for tr in tst]
    print(channels)
    reftime  = tst[0].stats.starttime
    p_times  = [reftime + val for val in np.asarray(picks[0][0])/100]
    s_times  = [reftime + val for val in np.asarray(picks[1][0])/100]
    p_scores = picks[0][1]
    s_scores = picks[1][1]
    p_df     = pd.DataFrame({'network':net_name,'station':sta_name,'time':p_times,'score':p_scores,'phase':'P'})#,'channels':channels})
    s_df     = pd.DataFrame({'network':net_name,'station':sta_name,'time':s_times,'score':s_scores,'phase':'S'})#,'channels':channels})
    
    df = pd.concat((p_df,s_df))
    return df


def get_labels(names,data):
    temp_labels = []
    for name in names:
        example=data[name]
        p,s = get_first_arrivals(example)
        temp_labels.append((p,s))
    return temp_labels

def compare_preds_labels(preds,labels):
    residuals=[]
    
    n = len(labels)
    for i in range(0,n):
        t_preds  = preds[i]
        t_labels = labels[i]
        
        if len(t_preds)==2:
            p_pred = t_preds[0]
            s_pred = t_preds[1]
        
        p_residual = t_labels[0]-p_pred
        s_residual = t_labels[1]-s_pred
        
        residuals.append((p_residual,s_residual))

    return residuals


def scrub_residuals(residuals):
    clean_residuals = []
    for residual in residuals:
        if (len(residual[0])==1 and len(residual[1])==1):
            clean_residuals.append((residual[0][0],residual[1][0]) )
            
    return np.asarray(clean_residuals)

def load_multistation_example(origin_ID,data,k_neighbors=5,seismometer_only=False,filtered=False,seisbench=True):
    """
    load the waveforms and arrivals over the network and cast into a torch_geometric graph
    the first part is the same as plotting it
    """
    examples = []
    names    = list(data.keys())
    outnames = []

    if seismometer_only:
        for name in names:
            if origin_ID in name and (name.split('.')[-1][1]=='H'):            
                outnames.append(name)
    else:
        for name in names:
            if origin_ID in name:
                outnames.append(name)

    pick_flag=[]
    for name in outnames:
        examples.append(data[name])
        example=data[name]
        p,s = get_first_arrivals(example,seisbench=seisbench)
        if p=='NaN':
            pick_flag.append(0)
        else:
            pick_flag.append(1)
            if seisbench==True:
                source_origin = [example.attrs['source_latitude_deg'],example.attrs['source_longitude_deg'],
                             example.attrs['source_depth_km'],example.attrs['source_origin_time'],
                             example.attrs['source_magnitude']]
            if seisbench==False:
                source_origin = [example.attrs['event_origin_latitude'],example.attrs['event_origin_longitude'],
                                 example.attrs['event_origin_depth'],example.attrs['event_origin_time'],
                                 example.attrs['magnitude']]    

    n_stations = len(examples)
    print(n_stations,' stations available')
    wav_length = examples[0][()].shape[-1]

    x = torch.zeros((n_stations,3,wav_length))
    y = torch.zeros((n_stations,3,wav_length))
    for i,example in enumerate(examples):
        if filtered:
            bandpassed_x = bandpass_data(example[()])
            bandpassed_x = bandpassed_x / np.max(np.abs(bandpassed_x))
            #print(bandpassed_x.strides)
            x[i,:,:] = torch.from_numpy(bandpassed_x.copy())
        else:
            x[i,:,:] = torch.from_numpy(example[()])
        y[i,:,:] = generate_triangle_label(example,half_width=250,length=wav_length,seisbench=seisbench)


    # get the station locations, calculate distances and create a knn graph
    if seisbench==True:
        station_codes = [example.attrs['station_code'] for example in examples]
        station_lats  = np.asarray([example.attrs['station_latitude_deg'] for example in examples])
        station_lons  = np.asarray([example.attrs['station_longitude_deg'] for example in examples])

    if seisbench==False:
        station_codes = [example.attrs['station'] for example in examples]
        station_lats  = np.asarray([example.attrs['station_latitude'] for example in examples])
        station_lons  = np.asarray([example.attrs['station_longitude'] for example in examples])

    #station_lats  = np.asarray([example.attrs['station_latitude'] for example in examples])
    #station_lons  = np.asarray([example.attrs['station_longitude'] for example in examples])
    trace_start_times = [example.attrs['trace_start_time'] for example in examples]

    knn = []
    for i,station in enumerate(station_codes):
        # for now station elevation is ignored
        distances = calculate_distances(station_lats,station_lons,np.zeros((len(station_lons))),station_lats[i],station_lons[i],0)
        # sort the distances and get the nearest neighbor indices, first nearest neighbor would be itself
        sorted_args = np.argsort(distances)
        knn.append(sorted_args[:k_neighbors+1])
    # turn knn into a list of tuples for each edge in both directions
    edge_index  = [] 
    for neighborhood in knn:
        tuples   = [[neighborhood[0], neighbor] for neighbor in neighborhood[1:] ]
        n_tuples = [[temp_tuple[1],temp_tuple[0]] for temp_tuple in tuples]
        edge_index+=tuples   
        edge_index+=n_tuples
     
    # cast into a pytorch geometric data
    edge_index = torch.transpose(torch.tensor(edge_index,dtype=torch.long),1,0).contiguous()
    graph=Data(x=x,y=y,edge_index=edge_index,station_codes=station_codes,
               station_latitudes=station_lats,station_longitudes=station_lons,start_times=trace_start_times,
               pick_flag=pick_flag,source_origin=source_origin)
    
    return graph

def plot_station_graph(graph,station_names=True):
    """
    takes a graph example and plots the station locations and the edges to the k nearest neighbors
    """
    fig = plt.figure(figsize=(7,7))
    #plt.scatter(graph.station_longitudes,graph.station_latitudes,marker='v',zorder=100)
    for sta_lon,sta_lat,name,pick_flags in zip(graph.station_longitudes,graph.station_latitudes,graph.station_codes,graph.pick_flag):
        if pick_flags==1:
            plt.scatter(sta_lon,sta_lat,marker='v',color='r',zorder=100)
        else:
            plt.scatter(sta_lon,sta_lat,marker='v',color='b',zorder=100)
        if station_names:
            plt.text(sta_lon+0.1,sta_lat+0.1,name,fontsize=12)


    # plot the edges by looping over the first half of the edge list
    station_longitudes=graph.station_longitudes
    station_latitudes=graph.station_latitudes
    for i in range(int(graph.edge_index.shape[-1])):
        plt.plot([station_longitudes[graph.edge_index[0,i]], station_longitudes[graph.edge_index[1,i]]], [station_latitudes[graph.edge_index[0,i]], station_latitudes[graph.edge_index[1,i]]],c='k',alpha=0.1)

    #add the source location
    try:
        olat = float(graph.source_origin[0])
        olon = float(graph.source_origin[1])
        print(olon,olat)
        plt.plot([olon],[olat],marker='*',c='gold',markersize=20,markeredgecolor='k')

    except Exception as e:
        print(e,'no origin')

    try:
        t_lat = graph.source_origin[0]
        t_lon = graph.source_origin[1]
        t_dep = graph.source_origin[2]
        t_mag = graph.source_origin[4]
        t_oti = graph.source_origin[3]
    
        plt.suptitle(str(t_lon)+', '+str(t_lat)+', '+str(t_dep/1000)+', M'+str(t_mag)+', '+str(t_oti)[:19])
        #plt.text(0,0,str(t_lon)+', '+str(t_lat)+', '+str(t_dep/1000)+', M'+str(t_mag)+', '+str(t_oti)[:19])
    except Exception as e:
        print(e)

    return fig

def plot_wav_graph(graph,distance_sorting=True,labels=True,component='Z',edges=False,preds=False,traveltimes=False):
    """
    plot the waveforms, edges, labels and other things
    """

    threshold=0.5
    fig=plt.figure()
    if   component=='Z':
         comp_index=2
    elif component=='N':
         comp_index=1
    elif component=='E':
         comp_index=0

    n_chans= graph.x.shape[0]
    xlim = graph.x.shape[-1]
    sorted_distances=np.arange(0,n_chans)
    if distance_sorting:
        tdeps = np.zeros((len(graph.station_latitudes)))
        distances = calculate_distances(graph.station_latitudes,graph.station_longitudes,tdeps,
                                    graph.source_origin[0],graph.source_origin[1],0)
        sorted_distances = np.argsort(distances)
        #print(distances) 
    

    #print(sorted_distances)
    offset = 0
    for i in sorted_distances:
   
         plt.plot(graph.x[i,comp_index,:]-offset,'k',linewidth=0.5,zorder=100)
         plt.plot(graph.y[i,1,:]-offset,'b',linewidth=1,linestyle='--')
         plt.plot(graph.y[i,0,:]-offset,'r',linewidth=1,linestyle='--')
         plt.text(-1100,-offset,graph.station_codes[i],fontsize=12,horizontalalignment='right',
                   verticalalignment='center')
         
         if preds:
             plt.plot(graph.preds[i,1,:]-offset,'b',linewidth=1)
             plt.plot(graph.preds[i,0,:]-offset,'r',linewidth=1)

             p_picks,_ = find_peaks(graph.preds[i,0,:], distance=150,height=threshold,width=100) 
             s_picks,_ = find_peaks(graph.preds[i,1,:], distance=150,height=threshold,width=100)
                         
             pick_pad=0.5
             if len(p_picks)>0:
                 #print(p_picks)
                 for p_pick in p_picks:
                     #plt.scatter(p_pick,-offset,s=100,marker='|',c='r')
                     plt.vlines(p_pick,ymin=-offset-pick_pad,ymax=-offset+pick_pad,colors='r')
                     #plt.scatter(p_pick,-0.2-offset,marker='2',c='r',s=100)
             if len(s_picks)>0:
                 for s_pick in s_picks:
                     #plt.scatter(s_pick,-offset,s=100,marker='|',c='b')
                     plt.vlines(s_pick,ymin=-offset-pick_pad,ymax=-offset+pick_pad,colors='b')     
                     #plt.scatter(s_pick,-0.2-offset,marker='2',c='b',s=200)
         offset+=1

    if edges:
        #overwrite the edge index to only use the first half
        #new_edge_index   = graph.edge_index[:,int(graph.edge_index.shape[-1]/2)]
        #graph.edge_index = new_edge_index     

        # add the station edges
        for i,node in enumerate(sorted_distances):
            out_node  = [sorted_distances[i]]
            mask      = np.asarray(graph.edge_index[1,:]==out_node[0]) 
            in_nodes  = graph.edge_index[0,:][mask] # these need to be translated to the plot coordinates
            p_in_nodes = []
            for in_node in in_nodes:
                 #print(in_node,sorted_distances)
                 p_in_nodes.append(np.where(sorted_distances==in_node.item())[0][0])
            out_nodes = [i]*len(in_nodes) 
            #print(in_nodes,out_nodes)             
      
            for j in range(0,len(in_nodes)):
                plt.plot([-1000,0],[-p_in_nodes[j], -out_nodes[j]], 'k',linewidth=1,alpha=0.1 )

    # estimate traveltimes using obspy taup and plot them with tolerance
    if traveltimes:
         offset = 0
         model  = TauPyModel(model="ak135")
         ttimes = []
         for i in sorted_distances:
             station_lat = graph.station_latitudes[i]
             station_lon = graph.station_longitudes[i]
             event_lat   = graph.source_origin[0]
             event_lon   = graph.source_origin[1]
             event_dep   = graph.source_origin[2]/1000
             t_distance,az,baz = calc_vincenty_inverse(event_lat,event_lon,station_lat,station_lon) 
            
             arrivals = model.get_travel_times(source_depth_in_km=event_dep,
                                  distance_in_degree=t_distance/111000,phase_list=["P","S","p","s"])
             #print(graph.station_codes[i],arrivals)
             arrivals_offset = (UT(graph.source_origin[3])-UT(graph.start_times[i]))
             #for arrival in arrivals:
             arrival = arrivals[0]
             p_timestamp = (arrival.time + arrivals_offset)*100
             sarrivals = model.get_travel_times(source_depth_in_km=event_dep,
                                   distance_in_degree=t_distance/111000,phase_list=["S","s"])
             sarrival = sarrivals[0]
             s_timestamp = (sarrival.time + arrivals_offset)*100
             tolerance   = 3*100 # seconds
             thickness   = 10
             plt.hlines(-offset,xmin=p_timestamp-tolerance,xmax=p_timestamp+tolerance,colors='r',linewidth=thickness,alpha=0.2) 
             plt.hlines(-offset,xmin=s_timestamp-tolerance,xmax=s_timestamp+tolerance,colors='b',linewidth=thickness,alpha=0.2)
             offset +=1


    try:
        t_lat = graph.source_origin[0]
        t_lon = graph.source_origin[1]
        t_dep = graph.source_origin[2]
        t_mag = graph.source_origin[4]
        t_oti = graph.source_origin[3]
    
        plt.suptitle(str(t_lon)+', '+str(t_lat)+', '+str(t_dep/1000)+', M'+str(t_mag)+', '+str(t_oti)[:19])
        #plt.text(0,0,str(t_lon)+', '+str(t_lat)+', '+str(t_dep/1000)+', M'+str(t_mag)+', '+str(t_oti)[:19])
    except Exception as e:
        print(e)

    plt.yticks([])
    plt.xlim(-2000,xlim)

    return fig

def calculate_distances(lats,lons,deps,ref_lat,ref_lon,ref_dep):
    """
    Calculates the distances using first:
    a spherical to cartesian coordinate conversion
    and then the L2-norm
    """
    R    = 6371 # km
    ref_dep = R - ref_dep 
    
    x_ref = ref_dep*np.sin(np.radians(90-ref_lat))*np.cos(np.radians(ref_lon))
    y_ref = ref_dep*np.sin(np.radians(90-ref_lat))*np.sin(np.radians(ref_lon))
    z_ref = ref_dep*np.cos(np.radians(90-ref_lat))
    
    x_ = (R-deps)*np.sin(np.radians(90-lats))*np.cos(np.radians(lons))
    y_ = (R-deps)*np.sin(np.radians(90-lats))*np.sin(np.radians(lons))
    z_ = (R-deps)*np.cos(np.radians(90-lats))

    distances = np.sqrt((x_ - x_ref)**2 + (y_ - y_ref)**2 + (z_ - z_ref)**2)

    return distances



 
def plot_example_section(origin_ID,data,channel='Z',component=False,filtered=False):
    """
    get all examples available for a given id and plot a record section
    """
    channel_index = {'E':0,'N':1,'Z':2}
    channel_index = channel_index[channel]
    
    examples = []
    names = list(data.keys())
    outnames = []
    for name in names:
        if origin_ID in name:
            outnames.append(name)
    for name in outnames:
        examples.append(data[name])
    #print(outnames)
    station_codes = [outname.split('.')[1] for outname in outnames]
    y_dim = len(list(station_codes))
    n_examples=[]
    if component:
        for example in examples:
            t_component = example.name.split('.')[2].split('_')[0]
            #print(t_component)
            if component==t_component:
                n_examples.append(example)
        examples=n_examples
        y_dim=len(examples)
    
    distances   = [example.attrs['path_ep_distance_deg'] for example in examples]
    sorted_dist = np.argsort(distances)
    
    #examples = examples[sorted_dist]
    figure=plt.figure(figsize=(10,y_dim))
    
    for i in range(y_dim):
        #plt.subplot(y_dim,1,i+1)
        example = examples[sorted_dist[i]]
        
        label = example.name.split('/')[2].split('_')[0]+ channel
        if filtered:
            data=bandpass_data(example[()],freqmin=1,freqmax=20)
            plt.plot(data[channel_index,:]-i,'k',linewidth=1,alpha=0.9,label=label)
        else:
            plt.plot(example[()][channel_index,:]-i,'k',linewidth=1,alpha=0.9,label=label)


        #plt.legend(loc='upper right',handlelength=0)
        plt.text(0,-i,label,horizontalalignment='right')
        plt.xlim(0,12000)
        plt.yticks([])
        #plt.axis('off')
        
        if example.attrs['trace_p_arrival_sample'] != 0:
            pg_pos = example.attrs['trace_p_arrival_sample']
            plt.scatter(pg_pos,-i,s=4000,c='red',marker='|',label='P',linewidth=1)
            #if i==0:plt.text(pg_pos-400,-5.2,'P',c='red')
        if example.attrs['trace_s_arrival_sample'] != 0:
            sg_pos = example.attrs['trace_s_arrival_sample']
            plt.scatter(sg_pos,0-i,s=4000,c='b',marker='|',label='S')
            #if i==0:plt.text(sg_pos,-5.2,'S',c='b')
    
    try:
        t_lat = example.attrs['source_latitude_deg']
        t_lon = example.attrs['source_longitude_deg']
        t_dep = example.attrs['source_depth_km']
        t_mag = example.attrs['source_magnitude']
        t_oti = example.attrs['source_origin_time']
    
        plt.suptitle(str(t_lon)+', '+str(t_lat)+', '+str(t_dep/1000)+', M'+str(t_mag)+', '+str(t_oti)[:19])
        #plt.text(0,0,str(t_lon)+', '+str(t_lat)+', '+str(t_dep/1000)+', M'+str(t_mag)+', '+str(t_oti)[:19])
    except Exception as e:
        print(e)

    return figure,distances




################ DATASET MANAGMENT AND QUERYING/SUBSETTING ################

def select_data(data,network=None,station=None,min_distance=None,max_distance=None,
                min_magnitude=None,max_magnitude=None,min_latitude=None,max_latitude=None,
                starttime=None,endtime=None):
    """
    take a database and subset based on multiple criteria    
    """
    names = list(data.keys())
    out_examples=[]
    out_names=[]
    # keep track of names as filters are applied
    for name in names:
         example = data[name]
         if network is not None:
             if example.attrs['network']==network:
                 out_names.append(name)
         if network is not None and station is not None:
             if example.attrs['network']==network and example.attrs['station']==station:
                 out_names.append(name)

         if station is not None:
             if example.attrs['station']==station:
                 out_examples.append(example)

         if min_magnitude is not None:
             if example.attrs['magnitude']>min_magnitude:
                 out_examples.append(example)

    return out_examples

def select_metadata(metadata):
    """
    this is to subset metadata, which is a pandas dataframe, making it easier
    """



################# PLOTTING #################

def plot_picks(st,df,component='Z',sta_order=None,distances=None):
    """
    plots a component of the waveforms and the picks from a skynet file
    """
    st.detrend('demean')

    figure=plt.figure(figsize=(13,7))
    picks = df.copy(deep=True)
    sttime = str(st[0].stats.starttime)
    sta_codes = list(set([tr.stats.station for tr in st]))
    max_len = max([tr.stats.npts for tr in st])
    if sta_order:
        sta_codes=sta_order
    for i,sta_code in enumerate(sta_codes):
        tst = st.select(station=sta_code)

        subset_picks = picks[picks['station']==sta_code]
        reftime = tst[0].stats.starttime
        pick_positions = [UT(time)- reftime for time in  subset_picks['time'] ]
        subset_picks.insert(2,'positions',pick_positions,True)

        p_picks = subset_picks[subset_picks['phase']=='P']
        s_picks = subset_picks[subset_picks['phase']=='S']

        pps = p_picks['positions'].to_list()
        sss = s_picks['positions'].to_list()

        plt.plot(tst.select(component=component)[0].normalize().data - i,'k',linewidth=0.5,alpha=0.6)
        plt.vlines(np.asarray(pps)*100,ymin=-i-0.5,ymax=-i+0.5,colors='r')
        plt.vlines(np.asarray(sss)*100,ymin=-i-0.5,ymax=-i+0.5,colors='b')
        plt.text(0,-i,sta_code,horizontalalignment='right',bbox=dict(boxstyle="round",fc='white'))
        if distances:
            temp = str(distances[i])+' km'
            plt.text(500,-i,temp,fontsize=12)    	

    plt.xlim(0,max_len)
    # put markers every ten minutes if the length is appropriate
    if max_len>30000:
        xlabels  = np.asarray(np.arange(0,max_len/60000)*10,dtype=int)
        xmarkers = np.arange(0,max_len,60000)
        #print(max_len,xlabels,xmarkers)
        plt.xticks(xmarkers,xlabels)
        # title with the start time of the first trace
        #sttime = str(tst[0].stats.starttime)
        plt.xlabel('Minutes after '+sttime)
    else:
       xlabels  = np.asarray(np.arange(0,max_len/6000 +1),dtype=int)
       xmarkers = np.arange(0,max_len+1,6000)
       plt.xticks(xmarkers,xlabels)
       plt.xlabel('Minutes after '+sttime)

    plt.yticks([])
    ax=plt.gca()

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return figure



def shift_and_stack(st,picks,component='Z',mode='envelope',phase='P',sta_order=None,distances=None):
    """
    shift waveforms to align to associated arrivals, process the waveforms according to the mode
    and stack.
    This is intended as a sanity check for new detections
    pass the dataframe corresponding to a single event
    """
    st.detrend('demean')

    figure=plt.figure(figsize=(13,7))
    picks = picks.copy(deep=True)
    picks['station'] = picks['station'].apply(split_names)
    picks['time'] = picks['time'].apply(adjust_times)
    
    sttime = str(st[0].stats.starttime)
    sta_codes = list(set([tr.stats.station for tr in st]))
    max_len = max([tr.stats.npts for tr in st]) 
    if sta_order:
        sta_codes=sta_order
    # estimate the required shifts to the earliest pick


    reftime = min([tr.stats.starttime for tr in st])
    pick_positions = [UT(time)- reftime for time in  picks['time'] ]
    picks.insert(2,'positions',pick_positions,True)

    p_picks = picks[picks['phase']=='P']
    s_picks = picks[picks['phase']=='S']

    pps = p_picks['positions'].to_list()
    sss = s_picks['positions'].to_list()

    ref_pick = min(pps)
    delays   = np.asarray(pps) - ref_pick 
    print(delays)

    # keep only the station codes for which there are arrivals
    if phase=='P':
        sta_codes = p_picks['station'].to_list()
    if phase=='S':
        sta_codes = s_picks['station'].to_list()

    #print(sta_codes)
    duration = 25000
    stack = []

    for i,sta_code in enumerate(sta_codes):
        try:
            tst = st.select(station=sta_code)
            #print(sta_code,tst)

            subset_picks = picks[picks['station']==sta_code]

            delay = delays[i]
            #print(tst)
            #print(len(tst.select(component=component)[0].data))
            temp_x = np.arange(0,len(tst.select(component=component)[0].data))-delay*100
            plt.plot(temp_x,np.abs(tst.select(component=component)[0].normalize().data) - i,'k',linewidth=0.5,alpha=0.6)
            stack.append(np.abs(tst.select(component=component)[0].normalize().data[int(delay*100):int(delay*100)+duration]))
            #plt.vlines(np.asarray(pps)*100,ymin=-i-0.5,ymax=-i+0.5,colors='r')
            #plt.vlines(np.asarray(sss)*100,ymin=-i-0.5,ymax=-i+0.5,colors='b')
            plt.text(0,-i,sta_code,horizontalalignment='right',bbox=dict(boxstyle="round",fc='white'))
            if distances:	
                 temp = str(distances[i])+' km'
                 plt.text(500,-i,temp,fontsize=12)
        except Exception as e:
            print(e)
            print('no waveforms for station ',sta_code)           
    #ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.xlim(0,30000)
    return figure,stack

def split_names(name):
    return name.split('.')[1]

def adjust_times(time):
    return time - 8*60*60



################MODELS########################    
class Long_PhaseNet(nn.Module):
    def __init__(self,kernel_size=7,stride=4):
        super(Long_PhaseNet,self).__init__()
        pad_size = int(kernel_size/2)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.conv1  = nn.Conv1d(3,8,  kernel_size=kernel_size,stride=1,padding='same')
        #
        self.conv2  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.conv3  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.conv4  = nn.Conv1d(8,11, kernel_size=kernel_size,stride=1,padding='same')
        self.conv5  = nn.Conv1d(11,11,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.conv6  = nn.Conv1d(11,16,kernel_size=kernel_size,stride=1,padding='same')
        self.conv7  = nn.Conv1d(16,16,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.conv8  = nn.Conv1d(16,22,kernel_size=kernel_size,stride=1,padding='same')
        self.conv9  = nn.Conv1d(22,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.conv10 = nn.Conv1d(22,32,kernel_size=kernel_size,stride=1,padding='same')
        
        # extra from original UNet
        self.conv11 = nn.Conv1d(32,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.conv12 = nn.Conv1d(32,40,kernel_size=kernel_size,stride=1,padding='same')
        
        self.dconv0  = nn.ConvTranspose1d(40,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.dconv01 = nn.Conv1d(64,32,kernel_size=kernel_size,stride=1,padding='same')
        #
        
        self.dconv1  = nn.ConvTranspose1d(32,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.dconv2  = nn.Conv1d(44,22,kernel_size=kernel_size,stride=1,padding='same')
        self.dconv3  = nn.ConvTranspose1d(22,16,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.dconv4  = nn.Conv1d(32,16,kernel_size=kernel_size,stride=1,padding='same')
        self.dconv5  = nn.ConvTranspose1d(16,11,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.dconv6  = nn.Conv1d(22,11,kernel_size=kernel_size,stride=1,padding='same')
        self.dconv7  = nn.ConvTranspose1d(11,8,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.dconv8  = nn.Conv1d(16,8,kernel_size=kernel_size,stride=1,padding='same')
        self.dconv9  = nn.Conv1d(8,3,kernel_size=kernel_size,stride=1,padding='same')
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,X):
        
        X1 = torch.relu(self.conv1(X))
        X2 = torch.relu(self.conv2(X1))
        X3 = torch.relu(self.conv3(X2))
        X4 = torch.relu(self.conv4(X3))
        X5 = torch.relu(self.conv5(X4))
        X6 = torch.relu(self.conv6(X5))
        X7 = torch.relu(self.conv7(X6))
        X8 = torch.relu(self.conv8(X7))
        X9 = torch.relu(self.conv9(X8))
        X10 = torch.relu(self.conv10(X9))
        
        # extra from original UNet
        X10_a = torch.relu(self.conv11(X10))
        X10_b = torch.relu(self.conv12(X10_a))
        
        #print('deepest')
        #print(X10_a.shape,X10_b.shape)
        
        X10_c = torch.relu(self.dconv0(X10_b))
        X10_c = torch.cat((X10_c,torch.zeros((X10_c.shape[0],X10_c.shape[1],1))),dim=-1)
        X10_c = torch.cat((X10,X10_c),dim=1)
        X10_d = torch.relu(self.dconv01(X10_c))
        
        #

        X11 = torch.relu(self.dconv1(X10_d))
        X12 = torch.cat((X11,X8),dim=1)
        #print(X11.shape,X8.shape,X12.shape)
        X12 = torch.relu(self.dconv2(X12))
        X13 = torch.relu(self.dconv3(X12))
        #X13 = torch.cat((X13,torch.zeros((X13.shape[0],X13.shape[1],1))),dim=2)
        #print(X6.shape,X13.shape)
        X14 = torch.relu(self.dconv4(torch.cat((X13,X6),dim=1)))
        X15 = torch.relu(self.dconv5(X14))
        #X15 = X15[:,:,:7501]
        X15 = torch.cat((X15,torch.zeros((X15.shape[0],X15.shape[1],1))),dim=2)
        #X4 = torch.cat((X4,torch.zeros((X4.shape[0],X4.shape[1],2))),dim=2)
        #print(X4.shape,X15.shape)
        X16 = torch.relu(self.dconv6(torch.cat((X15,X4),dim=1)))
        X17 = torch.relu(self.dconv7(X16))
        #X17 = X17[:,:,:30001]
        X17 = torch.cat((X17,torch.zeros((X17.shape[0],X17.shape[1],1))),dim=2)
        #print(X17.shape,X2.shape)
        
        X18 = torch.relu(self.dconv8(torch.cat((X17,X2),dim=1)))
        X19 = self.dconv9(X18)
        
        # add the softmax !
        X20 = self.softmax(X19)
        
        return X20




class BN_PhaseNet(nn.Module):
    def __init__(self,kernel_size=7,stride=4):
        super(BN_PhaseNet,self).__init__()
        pad_size = int(kernel_size/2)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.conv1  = nn.Conv1d(3,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.bn1    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv2  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.bn2    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv3  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn3    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv4  = nn.Conv1d(8,11, kernel_size=kernel_size,stride=1,padding='same')
        self.bn4    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.conv5  = nn.Conv1d(11,11,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn5    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.conv6  = nn.Conv1d(11,16,kernel_size=kernel_size,stride=1,padding='same')
        self.bn6    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.conv7  = nn.Conv1d(16,16,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn7    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.conv8  = nn.Conv1d(16,22,kernel_size=kernel_size,stride=1,padding='same')
        self.bn8    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.conv9  = nn.Conv1d(22,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn9    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.conv10 = nn.Conv1d(22,32,kernel_size=kernel_size,stride=1,padding='same')
        self.bn10    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        
        # extra from original UNet
        self.conv11 = nn.Conv1d(32,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn11   = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.conv12 = nn.Conv1d(32,40,kernel_size=kernel_size,stride=1,padding='same')
        self.bn12   = nn.BatchNorm1d(num_features=40,eps=1e-3)
        
        self.dconv0  = nn.ConvTranspose1d(40,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bnd0    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.dconv01 = nn.Conv1d(64,32,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd01    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        #
        
        self.dconv1  = nn.ConvTranspose1d(32,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bnd1    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.dconv2  = nn.Conv1d(44,22,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd2    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.dconv3  = nn.ConvTranspose1d(22,16,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd3    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.dconv4  = nn.Conv1d(32,16,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd4    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.dconv5  = nn.ConvTranspose1d(16,11,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd5    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.dconv6  = nn.Conv1d(22,11,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd6    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.dconv7  = nn.ConvTranspose1d(11,8,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd7    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.dconv8  = nn.Conv1d(16,8,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd8    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.dconv9  = nn.Conv1d(8,3,kernel_size=kernel_size,stride=1,padding='same')
            
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,X):
        
        X1  = torch.relu(self.bn1(self.conv1(X)))
        X2  = torch.relu(self.bn2(self.conv2(X1)))
        X3  = torch.relu(self.bn3(self.conv3(X2)))
        X4  = torch.relu(self.bn4(self.conv4(X3)))
        X5  = torch.relu(self.bn5(self.conv5(X4)))
        X6  = torch.relu(self.bn6(self.conv6(X5)))
        X7  = torch.relu(self.bn7(self.conv7(X6)))
        X8  = torch.relu(self.bn8(self.conv8(X7)))
        X9  = torch.relu(self.bn9(self.conv9(X8)))
        X10 = torch.relu(self.bn10(self.conv10(X9)))
        
        # extra from original UNet
        X10_a = torch.relu(self.bn11(self.conv11(X10)))
        X10_b = torch.relu(self.bn12(self.conv12(X10_a)))
        
        #print('deepest')
        #print(X10_a.shape,X10_b.shape)
        
        X10_c = torch.relu(self.bnd0(self.dconv0(X10_b)))
        X10_c = torch.cat((X10_c,torch.zeros((X10_c.shape[0],X10_c.shape[1],1),device=self.device)),dim=-1)
        X10_c = torch.cat((X10,X10_c),dim=1)
        X10_d = torch.relu(self.bnd01(self.dconv01(X10_c)))
        
        #

        X11 = torch.relu(self.bnd1(self.dconv1(X10_d)))
        X12 = torch.cat((X11,X8),dim=1)
        #print(X11.shape,X8.shape,X12.shape)
        X12 = torch.relu(self.bnd2(self.dconv2(X12)))
        X13 = torch.relu(self.bnd3(self.dconv3(X12)))
        #X13 = torch.cat((X13,torch.zeros((X13.shape[0],X13.shape[1],1))),dim=2)
        #print(X6.shape,X13.shape)
        X14 = torch.relu(self.bnd4(self.dconv4(torch.cat((X13,X6),dim=1))))
        X15 = torch.relu(self.bnd5(self.dconv5(X14)))
        #X15 = X15[:,:,:7501]
        X15 = torch.cat((X15,torch.zeros((X15.shape[0],X15.shape[1],1),device=self.device)),dim=2)
        #X4 = torch.cat((X4,torch.zeros((X4.shape[0],X4.shape[1],2))),dim=2)
        #print(X4.shape,X15.shape)
        X16 = torch.relu(self.bnd6(self.dconv6(torch.cat((X15,X4),dim=1))))
        X17 = torch.relu(self.bnd7(self.dconv7(X16)))
        #X17 = X17[:,:,:30001]
        X17 = torch.cat((X17,torch.zeros((X17.shape[0],X17.shape[1],1),device=self.device)),dim=2)
        #print(X17.shape,X2.shape)
        
        X18 = torch.relu(self.bnd8(self.dconv8(torch.cat((X17,X2),dim=1))))
        X19 = self.dconv9(X18)
        
        # add the softmax !
        X20 = self.softmax(X19)
        
        return X20


class Regional_Picker(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,kernel_size=7,stride=4):
        super(Regional_Picker,self).__init__()
        pad_size = int(kernel_size/2)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.conv1  = nn.Conv1d(in_channels,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.bn1    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv2  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.bn2    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv3  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn3    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv4  = nn.Conv1d(8,11, kernel_size=kernel_size,stride=1,padding='same')
        self.bn4    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.conv5  = nn.Conv1d(11,11,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn5    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.conv6  = nn.Conv1d(11,16,kernel_size=kernel_size,stride=1,padding='same')
        self.bn6    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.conv7  = nn.Conv1d(16,16,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn7    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.conv8  = nn.Conv1d(16,22,kernel_size=kernel_size,stride=1,padding='same')
        self.bn8    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.conv9  = nn.Conv1d(22,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn9    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.conv10 = nn.Conv1d(22,32,kernel_size=kernel_size,stride=1,padding='same')
        self.bn10    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        
        # extra from original UNet
        self.conv11 = nn.Conv1d(32,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn11   = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.conv12 = nn.Conv1d(32,40,kernel_size=kernel_size,stride=1,padding='same')
        self.bn12   = nn.BatchNorm1d(num_features=40,eps=1e-3)
        
        self.dconv0  = nn.ConvTranspose1d(40,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bnd0    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.dconv01 = nn.Conv1d(64,32,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd01    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        #
        
        self.dconv1  = nn.ConvTranspose1d(32,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bnd1    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.dconv2  = nn.Conv1d(44,22,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd2    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.dconv3  = nn.ConvTranspose1d(22,16,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd3    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.dconv4  = nn.Conv1d(32,16,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd4    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.dconv5  = nn.ConvTranspose1d(16,11,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd5    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.dconv6  = nn.Conv1d(22,11,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd6    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.dconv7  = nn.ConvTranspose1d(11,8,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd7    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.dconv8  = nn.Conv1d(16,8,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd8    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.dconv9  = nn.Conv1d(8,out_channels,kernel_size=kernel_size,stride=1,padding='same')
            
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,X):
        
        X1  = torch.relu(self.bn1(self.conv1(X)))
        X2  = torch.relu(self.bn2(self.conv2(X1)))
        X3  = torch.relu(self.bn3(self.conv3(X2)))
        X4  = torch.relu(self.bn4(self.conv4(X3)))
        X5  = torch.relu(self.bn5(self.conv5(X4)))
        X6  = torch.relu(self.bn6(self.conv6(X5)))
        X7  = torch.relu(self.bn7(self.conv7(X6)))
        X8  = torch.relu(self.bn8(self.conv8(X7)))
        X9  = torch.relu(self.bn9(self.conv9(X8)))
        X10 = torch.relu(self.bn10(self.conv10(X9)))
        
        # extra from original UNet
        X10_a = torch.relu(self.bn11(self.conv11(X10)))
        X10_b = torch.relu(self.bn12(self.conv12(X10_a)))
        
        #print('deepest')
        #print(X10_a.shape,X10_b.shape)
        
        X10_c = torch.relu(self.bnd0(self.dconv0(X10_b)))
        X10_c = torch.cat((X10_c,torch.zeros((X10_c.shape[0],X10_c.shape[1],1),device=self.device)),dim=-1)
        X10_c = torch.cat((X10,X10_c),dim=1)
        X10_d = torch.relu(self.bnd01(self.dconv01(X10_c)))
        
        #

        X11 = torch.relu(self.bnd1(self.dconv1(X10_d)))
        X12 = torch.cat((X11,X8),dim=1)
        #print(X11.shape,X8.shape,X12.shape)
        X12 = torch.relu(self.bnd2(self.dconv2(X12)))
        X13 = torch.relu(self.bnd3(self.dconv3(X12)))
        #X13 = torch.cat((X13,torch.zeros((X13.shape[0],X13.shape[1],1))),dim=2)
        #print(X6.shape,X13.shape)
        X14 = torch.relu(self.bnd4(self.dconv4(torch.cat((X13,X6),dim=1))))
        X15 = torch.relu(self.bnd5(self.dconv5(X14)))
        #X15 = X15[:,:,:7501]
        X15 = torch.cat((X15,torch.zeros((X15.shape[0],X15.shape[1],1),device=self.device)),dim=2)
        #X4 = torch.cat((X4,torch.zeros((X4.shape[0],X4.shape[1],2))),dim=2)
        #print(X4.shape,X15.shape)
        X16 = torch.relu(self.bnd6(self.dconv6(torch.cat((X15,X4),dim=1))))
        X17 = torch.relu(self.bnd7(self.dconv7(X16)))
        #X17 = X17[:,:,:30001]
        X17 = torch.cat((X17,torch.zeros((X17.shape[0],X17.shape[1],1),device=self.device)),dim=2)
        #print(X17.shape,X2.shape)
        
        X18 = torch.relu(self.bnd8(self.dconv8(torch.cat((X17,X2),dim=1))))
        X19 = self.dconv9(X18)
        
        # add the softmax !
        X20 = self.softmax(X19)
        
        return X20


# picker made for two minute long waveforms in Colombia
class col_picker(nn.Module):
    def __init__(self,kernel_size=7,stride=4):
        super(col_picker,self).__init__()
        pad_size = int(kernel_size/2)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        
        self.conv1  = nn.Conv1d(3,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.bn1    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv2  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=1,padding='same')
        self.bn2    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv3  = nn.Conv1d(8,8,  kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn3    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.conv4  = nn.Conv1d(8,11, kernel_size=kernel_size,stride=1,padding='same')
        self.bn4    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.conv5  = nn.Conv1d(11,11,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn5    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.conv6  = nn.Conv1d(11,16,kernel_size=kernel_size,stride=1,padding='same')
        self.bn6    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.conv7  = nn.Conv1d(16,16,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn7    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.conv8  = nn.Conv1d(16,22,kernel_size=kernel_size,stride=1,padding='same')
        self.bn8    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.conv9  = nn.Conv1d(22,22,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn9    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.conv10 = nn.Conv1d(22,32,kernel_size=kernel_size,stride=1,padding='same')
        self.bn10   = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.conv11 = nn.Conv1d(32,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bn11   = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.conv12 = nn.Conv1d(32,48,kernel_size=kernel_size,stride=1,padding='same')
        self.bn12   = nn.BatchNorm1d(num_features=48,eps=1e-3)
        
        self.dconv1  = nn.ConvTranspose1d(48,32,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bnd1    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.dconv2  = nn.Conv1d(64,32,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd2    = nn.BatchNorm1d(num_features=32,eps=1e-3)
        self.dconv3  = nn.ConvTranspose1d(32,22,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd3    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.dconv4  = nn.Conv1d(44,22,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd4    = nn.BatchNorm1d(num_features=22,eps=1e-3)
        self.dconv5  = nn.ConvTranspose1d(22,16,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd5    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.dconv6  = nn.Conv1d(32,16,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd6    = nn.BatchNorm1d(num_features=16,eps=1e-3)
        self.dconv7  = nn.ConvTranspose1d(16,11,kernel_size=kernel_size,stride=stride,padding=pad_size-1)
        self.bnd7    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.dconv8  = nn.Conv1d(22,11,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd8    = nn.BatchNorm1d(num_features=11,eps=1e-3)
        self.dconv9  = nn.ConvTranspose1d(11,8,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.bnd9    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.dconv10  = nn.Conv1d(16,8,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd10    = nn.BatchNorm1d(num_features=8,eps=1e-3)
        self.dconv11  = nn.Conv1d(8,3,kernel_size=kernel_size,stride=1,padding='same')
        self.bnd11    = nn.BatchNorm1d(num_features=3,eps=1e-3)        
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,X):
        
        X1 = torch.relu(self.bn1(self.conv1(X)));#print('X1',X1.shape)
        X2 = torch.relu(self.bn2(self.conv2(X1)));#print('X2',X2.shape)
        X3 = torch.relu(self.bn3(self.conv3(X2)));#print('X3',X3.shape)
        X4 = torch.relu(self.bn4(self.conv4(X3)));#print('X4',X4.shape)
        X5 = torch.relu(self.bn5(self.conv5(X4)));#print('X5',X5.shape)
        X6 = torch.relu(self.bn6(self.conv6(X5)));#print('X6',X6.shape)
        X7 = torch.relu(self.bn7(self.conv7(X6)));#print('X7',X7.shape)
        X8 = torch.relu(self.bn8(self.conv8(X7)));#print('X8',X8.shape)
        X9 = torch.relu(self.bn9(self.conv9(X8)));#print('X9',X9.shape)
        X10 = torch.relu(self.bn10(self.conv10(X9)));#print('X10',X10.shape)
        X11 = torch.relu(self.bn11(self.conv11(X10)));#print('X11',X11.shape)
        X12 = torch.relu(self.bn12(self.conv12(X11)));#print('X12',X12.shape,' deepest')
        ##print(X12.shape, ' deepest ')
        # the up branch
        X13 = torch.relu(self.bnd1(self.dconv1(X12)));#print('X13',X13.shape)
        X13 = torch.cat((torch.zeros((X13.shape[0],X13.shape[1],1),device=self.device),X13,torch.zeros((X13.shape[0],X13.shape[1],1),device=self.device)),dim=-1);#print('X13',X13.shape)
        X14 = torch.cat((X10,X13),dim=1);#print('X14',X14.shape)
        X14 = torch.relu(self.bnd2(self.dconv2(X14)));#print('X14',X14.shape)
        X15 = torch.relu(self.bnd3(self.dconv3(X14)));#print('X15',X15.shape)
        X15 = torch.cat((X15,torch.zeros((X15.shape[0],X15.shape[1],1),device=self.device)),dim=-1)
        X15 = torch.cat((X15,X8),dim=1);#print('X15',X15.shape)
        X16 = torch.relu(self.bnd4(self.dconv4(X15)));#print('X16',X16.shape)
        X17 = torch.relu(self.bnd5(self.dconv5(X16)));X17=X17[:,:,:-1];#print('X17',X17.shape)
        X17 = torch.cat((X6,X17),dim=1);#print('X17',X17.shape)
        X18 = torch.relu(self.bnd6(self.dconv6(X17)));#print('X18',X18.shape)
        X19 = torch.relu(self.bnd7(self.dconv7(X18)))
        X19 = torch.cat((X19,torch.zeros((X19.shape[0],X19.shape[1],1),device=self.device)),dim=-1)
        X19 = torch.cat((X4,X19),dim=1);#print('X19',X19.shape)
        X20 = torch.relu(self.bnd8(self.dconv8(X19)));#print('X20',X20.shape)
        X21 = torch.relu(self.bnd9(self.dconv9(X20)))
        X21 = torch.cat((torch.zeros((X21.shape[0],X21.shape[1],1),device=self.device),X21,torch.zeros((X21.shape[0],X21.shape[1],2),device=self.device)),dim=-1);#print('X21',X21.shape)
        X21 = torch.cat((X2,X21),dim=1);#print('X21',X21.shape)
        X22 = torch.relu(self.bnd10(self.dconv10(X21)));#print('X22',X22.shape)
        X23 = torch.relu(self.bnd11(self.dconv11(X22)));#print('X23',X23.shape)
        
        X24 = self.softmax(X23)
     
        return X24

