import os
import glob
import obspy
from scipy import interpolate
import numpy as np
import pandas as pd 

'''
this script provides the functions to 1) collect the phase velocity
measurements from FTAN; 2) interpolate the snr estimation and 3) filter
out these data of bad quality for tomography

by Chengxin Jiang @ANU (Jul/22)
'''

def get_dist_array(allfiles):
    '''
    return an array containing station coordinates and distance of each station pair

    allfiles: array of all ccfs files
    '''
    nfiles = len(allfiles)
    sta_array = np.zeros(shape=(nfiles,5),dtype=np.float32)

    for ii,ifile in enumerate(allfiles):
        st = obspy.read(ifile)
        stla,stlo = st[0].stats['sac']['stla'],st[0].stats['sac']['stlo']
        evla,evlo = st[0].stats['sac']['evla'],st[0].stats['sac']['evlo']
        dist,tmp,tmp = obspy.geodetics.base.gps2dist_azimuth(stla,stlo,evla,evlo)
        sta_array[ii] = [evlo,evla,stlo,stla,dist/1000]

    return sta_array


# define rootpath 
rootpath = '/noise2/chengxin/OK/NOISE/STACK_1200s/STACK_SAC/ZZ'
ALLFILES = sorted(glob.glob(os.path.join(rootpath,'*/*.SAC')))
nfiles = len(ALLFILES)

# target period range and selection criteria
per  = [1,2,3,4,5,6,8,10,12,14,16,18,20,25,30]
snr_cri = 5
wlen_cri= 1.5
nper = len(per)

# design 2d array for snr and disp values
snr_array  = np.zeros(shape=(nfiles,nper),dtype=np.float32)
disp_array = np.zeros(shape=(nfiles,nper),dtype=np.float32)

# get station distance
sta_array = get_dist_array(ALLFILES)

# loop through each station-pair
for ii,ifile in enumerate(ALLFILES):
    #disp_file = ifile+'_2_DISP.1'
    snr_file  = ifile+'_snr.txt'

    #if not os.path.isfile(disp_file) or not os.path.isfile(snr_file):
    #    print('cannot file disp or snr files for %s'%ifile.split('/')[-1])

    # load the dispersion file
    #df1 = pd.read_csv(disp_file,
    #                  sep='\s+',
    #                  names=["indx","per","cper","group","phase","pow1","pow2"])
    # do the interpolation and do interpolation
    #fc_inc = interpolate.interp1d(df1["cper"].values,
    #                              df1["phase"].values,
    #                              bounds_error=False,
    #                              fill_value=0)
    #disp_array[ii] = fc_inc(per)
    disp_array[ii] = np.ones(nper)

    # load the snr files and do interpolation
    df2 = pd.read_csv(snr_file,sep='\s+')
    fc_inc = interpolate.interp1d(df2["period"].values,
                                  df2["snr"].values,
                                  bounds_error=False,
                                  fill_value=0)
    snr_array[ii] = fc_inc(per)

# loop through each period
for ii in range(nper):
    flag = np.zeros(nfiles,dtype=int)

    indx1 = np.where(disp_array[:,ii]>0)[0]
    indx2 = np.where(snr_array[:,ii]>0)[0]
    indx = np.intersect1d(indx1,indx2)
    if not len(indx):
        print('contine! not enough data for period %ss'%per[ii])

    # do data qc 
    ave_vel = np.mean(disp_array[indx,ii])
    wlen = wlen_cri*per[ii]*ave_vel 
    indx3 = np.where(sta_array[:,4]>wlen)[0]
    indx4 = np.where(snr_array[:,ii]>=snr_cri)[0]
    indx5 = np.intersect1d(indx3,indx4)
    indx0 = np.intersect1d(indx,indx5)
    flag[indx0] = 1

    # save into file
    outfn = os.path.join(rootpath,"disp_snr_{0:02d}s.txt".format(per[ii]))
    df = pd.DataFrame({"slon":sta_array[indx,0],
                       "slat":sta_array[indx,1],
                       "rlon":sta_array[indx,2],
                       "rlat":sta_array[indx,3],
                       "phase":disp_array[indx,ii],
                       "snr":snr_array[indx,ii],
                       "dist":sta_array[indx,4],
                       "qc_flag":flag[indx]})
    df.to_csv(outfn,sep=" ",index=True,header=False,float_format="%.3f")
