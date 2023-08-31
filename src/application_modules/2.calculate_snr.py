import os
import glob
import obspy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
a script to estimate the period-dependent SNR of the 
cross correlation functions

TODO: 
1. add the option to deal with ccfs with two lags
2. add the option to save the data matrix

by Chengxin Jiang @ANU (Jul/22)
'''

def make_freqvec_signal_window(minP,maxP,nper):
    '''
    based on the provided period, provide a velocity range for
    specific period

    minP: minimum period
    maxP: maximum period
    nper: number of period
    wave_type: rayl. or love wave
    '''

    # define step in frequency
    fmin = 1/maxP 
    fmax = 1/minP 
    fstep = (np.log(fmax)-np.log(fmin))/(nper-1)
    freq_vec = np.zeros(nper,dtype=np.float32)
    per_vec = np.zeros(nper,dtype=np.float32)

    # define velocity array to window signals
    vel_array = np.zeros(shape=(2,nper-2),dtype=np.float32)

    # make the frequency vec
    for ii in range(nper):
        freq_vec[ii] = np.exp(np.log(fmin)+fstep*ii)
    per_vec  = 1/freq_vec

    # provide a range of period-dependent velocity
    for ii in range(1,nper-1):
        if per_vec[ii] < 1:
            vmin = 0.5
            vmax = 2.0
        if per_vec[ii] < 10:
            vmin = 1.0
            vmax = 3.0 
        elif per_vec[ii] < 30:
            vmin = 2.0
            vmax = 4.0 
        elif per_vec[ii] < 100:
            vmin = 2.5
            vmax = 4.5
        vel_array[0,ii-1] = vmin
        vel_array[1,ii-1] = vmax

    return freq_vec,per_vec,vel_array


def fold_ccfs(st):
    '''
    check whether the provided CCFs are folded or not

    tr: obspy stream
    '''
    npts = st[0].stats.npts
    sps  = int(st[0].stats.sampling_rate)
    maxlag = int((npts-1)/sps)
    tvec = np.linspace(-maxlag,maxlag,npts)
    data = st[0].data

    # find positive and negative lag
    p_indx = np.where(tvec>=0)[0]
    n_indx = np.where(tvec<=0)[0]
    if len(p_indx) != len(n_indx):
        raise ValueError('Abort! positive and negative lags do not have same length')
    tdata = 0.5*(data[p_indx]+np.flip(data[n_indx]))
    st[0].data = tdata 

    return st


def estimate_snr(st,vel_array,freq_vec,plot_flag,tfile):
    '''
    estimate the signal-to-noise ratio based on provide velocity range

    st: obspy stream object
    vel_array: minimum and maximum velocity for signal
    plot_flag: plot the filtered waveform or not
    '''
    nper = vel_array.shape[1]+2

    # load some basic parameters
    sps  = int(st[0].stats.sampling_rate)
    npts = st[0].stats.npts
    st[0].taper(max_percentage=0.05,max_length=20)
    dist,tmp,tmp = obspy.geodetics.base.gps2dist_azimuth(st[0].stats['sac']['stla'],
                                                 st[0].stats['sac']['stlo'],
                                                 st[0].stats['sac']['evla'],
                                                 st[0].stats['sac']['evlo'])
    dist /= 1000

    # loop through each period range
    data = st[0].data 
    tdata = np.zeros(shape=(nper-2,npts),dtype=np.float32)  # filtered data 
    snr_vec = np.zeros(nper-2,dtype=np.float32)             # estimated snr
    snr_indx = np.zeros(shape=(nper-2,4),dtype=np.float32)  # signal window t1-t2; noise window t1-t2

    for ii in range(1,nper-1):
        fmin = freq_vec[ii-1]
        fmax = freq_vec[ii+1]
        tdata[ii-1] = np.float32(bandpass(data,fmin,fmax,df=sps,corners=4,zerophase=True))
        snr_indx[ii-1],snr_vec[ii-1] = get_snr_onefreq(tdata[ii-1],vel_array[:,ii-1],1/sps,dist)

    # plot the waveforms
    if plot_flag:
        # make time vector
        tvec = np.linspace(0,(npts-1)/sps,npts)
        plot_waveform(tdata,tvec,freq_vec,snr_vec,snr_indx,tfile,dist)

    return snr_vec


def get_snr_onefreq(data,vel_array,dt,dist):
    '''
    get the signal-to-noise ratio (snr) for the ccfs at one frequency range

    data: filtered waveform data
    vel_array: 2D array with minimum and maximum velocity for the SW
    dt: time interval
    dist: distance between source and receiver stations
    '''

    # make a time vector
    vmin = vel_array[0]
    vmax = vel_array[1]
    npts = len(data)
    tvec = np.linspace(0,(npts-1)*dt,npts)

    # define the signal window
    t1 = dist/vmax-maxP
    t2 = dist/vmin+maxP
    if t1<0:
        t1 = 0
    if t2>tvec[-1]:
        t2 = tvec[-1]
    s_indx = np.where((tvec>=t1)&(tvec<=t2))[0]
    if not len(s_indx):
        raise ValueError('the ccfs is too short to find the signal window')

    # define the noise window
    t11 = t2+5
    t22 = t11+30
    if t11>tvec[-1]:
        raise ValueError('the ccfs is too short to find the specified noise window')
    if t22>tvec[-1]:
        t11 = t11-10
        t22 = tvec[-1]
    n_indx = np.where((tvec>=t11)&(tvec<=t22))[0]

    # estimate signal-to-noise ratio
    snr = np.max(np.abs(data[s_indx]))/np.sqrt(np.sum(data[n_indx]**2/len(n_indx)))
    twin_array = [t1,t2,t11,t22]
    return twin_array,snr


def plot_waveform(tdata,tvec,freq_vec,snr_vec,snr_indx,tfile,dist):
    '''
    plot the filtered waveforms and defined signal and noise window
    '''
    plt.figure(figsize=(6,10))
    nper,npts = tdata.shape

    for ii in range(nper):
        plt.plot(tvec,tdata[ii]/np.max(tdata[ii])+ii*2,'k-',lw=2)
        plt.plot([snr_indx[0],snr_indx[0]],[ii*2,ii*2+1],'r-',lw=1)
        plt.plot([snr_indx[1],snr_indx[1]],[ii*2,ii*2+1],'r-',lw=1)
        plt.plot([snr_indx[2],snr_indx[2]],[ii*2,ii*2+1],'b-',lw=1)
        plt.plot([snr_indx[3],snr_indx[3]],[ii*2,ii*2+1],'b-',lw=1)
        plt.title("%s %4.1f km"%(tfile.split('/')[-1],dist))
        plt.text(tvec[5],ii*2+1,"%4.1f-%4.1f s]"%(1/freq_vec[ii+2],1/freq_vec[ii]))
        plt.text(tvec[-20],ii*2+1,"%4.1f"%snr_vec[ii])
    plt.savefig(tfile+'_wf.pdf',format='pdf',dpi=300)
    plt.close()


#######################
#### MAIN FUNCTION ####
#######################

# define rootpath 
rootpath = '/noise2/chengxin/nodes/Lake_George/STACK_DH/STACK_SAC/ZZ'
ALLFILES = glob.glob(os.path.join(rootpath,'*/*.SAC'))

# basic parameters controlling output numbers of SNR values
maxP = 1
minP = 0.1
nper = 10
unfolded = True                     # whether the provided Allfiles are folded ccfs
plot_flag = True                    # show the period-dependent ccfs
wavetype = 'Rayleigh'               # wavetype of provided ccfs

# estimate the frequency vector and define the time window for "signal"
freq_vec,per_vec,vel_array = make_freqvec_signal_window(minP,maxP,nper)

# loop through each file
for ifile in ALLFILES:
    st = obspy.read(ifile)

    # folder the ccfs if needed
    if unfolded:
        nst = fold_ccfs(st)
    else:
        nst = st

    # estimate the snr
    snr_vec = estimate_snr(nst,vel_array,freq_vec,plot_flag,ifile)
    df = pd.DataFrame({'period':np.flip(per_vec[1:-1]),
                      'snr':np.flip(snr_vec)})

    # write into file
    outname = ifile+'_snr.txt'
    df.to_csv(outname,index=False,
                      header=True,
                      float_format='%.2f',
                      sep=' ')
