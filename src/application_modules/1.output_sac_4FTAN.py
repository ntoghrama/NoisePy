import os
import glob
import pathlib
import pyasdf
import numpy as np
from obspy.io.sac.sactrace import SACTrace

'''
this script outputs the stacked cross-correlation functions into either SAC traces
or binary txt files [for Image Transformation Analysis of Yao et al., 2005].

add new functionality to output the positive, negative and symetric component of 
the cross-correlation function

by Chengxin Jiang @ANU (Jun/22)
'''

def make_output_names(ccf_name,comp,out_SAC):
    '''
    load station name information from the ccfs files
    and use them to name the output SAC files

    fname: full path of a cross-correlation function
    comp: component of the data
    out_SAC: whether output as SAC format
    '''

    # get station network and name 
    fname = ccf_name.split('/')[-1].split('_')
    staS = fname[0].split('.')[1]
    netS = fname[0].split('.')[0]
    staR = fname[1].split('.')[1]
    netR = fname[1].split('.')[0]

    # check whether folder exists
    tdir = pathlib.Path(os.path.join(STACKDIR,'STACK_SAC',comp,netS+'.'+staS))
    if not out_SAC:
        tdir.replace('STACK_SAC','STACK_DAT')
    if out_SAC:
        temp = netS+'.'+staS+'_'+netR+'.'+staR+'.SAC'
    else:
        temp = netS+'.'+staS+'_'+netR+'.'+staR+'.dat'

    tdir.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(tdir):
        os.mkdir(tdir)

    fout = os.path.join(tdir,temp)
    return fout

def load_data_parameters(ds):
    '''
    load the ccfs data and save the associated parameters into a dic 

    ds: one activate asdf file containing cross correlation functions
    '''
    data_dic = {}

    # load the two keys for a data structure
    slist = ds.auxiliary_data.list()
    data_dic['slist'] = slist
    rlist = ds.auxiliary_data[slist[0]].list()
    data_dic['rlist'] = rlist

    # read the parameters
    data_dic['maxlag']= ds.auxiliary_data[slist[0]][rlist[0]].parameters['maxlag']
    data_dic['dt']    = ds.auxiliary_data[slist[0]][rlist[0]].parameters['dt']
    data_dic['npts']  = ds.auxiliary_data[slist[0]][rlist[0]].data.shape[0]
    data_dic['slat']  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['latS']
    data_dic['slon']  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lonS']
    data_dic['rlat']  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['latR']
    data_dic['rlon']  = ds.auxiliary_data[slist[0]][rlist[0]].parameters['lonR']

    # construct the time series
    tvec = np.linspace(-data_dic['maxlag'],data_dic['maxlag'],data_dic['npts'])
    data_dic['tvec'] = tvec

    return data_dic

def save_sac(filename,data_dic,three_lags,corr):
    '''
    save the ccfs as sac format

    filename: output name
    data_dic: parameters associated with the ccfs
    three_lags: whether to output the positive, negative and symetric lags
    '''
    tvec = data_dic['tvec']
    nlag_indx = np.where(tvec<=0)[0]
    plag_indx = np.where(tvec>=0)[0]

    # write into SAC format 
    sac = SACTrace(nzyear=2000,
                    nzjday=1,
                    nzhour=0,
                    nzmin=0,
                    nzsec=0,
                    nzmsec=0,
                    b=-data_dic['maxlag'],
                    delta=data_dic['dt'],
                    stla=data_dic['rlat'],
                    stlo=data_dic['rlon'],
                    evla=data_dic['slat'],
                    evlo=data_dic['slon'],
                    lcalda=1,
                    data=corr)
    sac.write(filename,byteorder='big')

    # whether output other components
    if three_lags:
        if len(nlag_indx) != len(plag_indx):
            raise ValueError('ccfs missing one point')
        pcorr = corr[plag_indx]
        ncorr = np.flip(corr[nlag_indx])
        scorr = pcorr*0.5+ncorr*0.5
        sac = SACTrace(nzyear=2000,
                        nzjday=1,
                        nzhour=0,
                        nzmin=0,
                        nzsec=0,
                        nzmsec=0,
                        b=0,
                        delta=data_dic['dt'],
                        stla=data_dic['rlat'],
                        stlo=data_dic['rlon'],
                        evla=data_dic['slat'],
                        evlo=data_dic['slon'],
                        lcalda=1,
                        data=scorr)
        sac.write(filename+'_s',byteorder='big')  
        sac = SACTrace(nzyear=2000,
                        nzjday=1,
                        nzhour=0,
                        nzmin=0,
                        nzsec=0,
                        nzmsec=0,
                        b=0,
                        delta=data_dic['dt'],
                        stla=data_dic['rlat'],
                        stlo=data_dic['rlon'],
                        evla=data_dic['slat'],
                        evlo=data_dic['slon'],
                        lcalda=1,
                        data=pcorr)
        sac.write(filename+'_p',byteorder='big')              
        sac = SACTrace(nzyear=2000,
                        nzjday=1,
                        nzhour=0,
                        nzmin=0,
                        nzsec=0,
                        nzmsec=0,
                        b=0,
                        delta=data_dic['dt'],
                        stla=data_dic['rlat'],
                        stlo=data_dic['rlon'],
                        evla=data_dic['slat'],
                        evlo=data_dic['slon'],
                        lcalda=1,
                        data=ncorr)
        sac.write(filename+'_n',byteorder='big')



# absolute path for data output
STACKDIR = "/noise2/chengxin/nodes/Lake_George/STACK_SH"
ALLFILES = glob.glob(os.path.join(STACKDIR,'*/*.h5'))
OUTDIR = os.path.join(STACKDIR,'STACK_SAC')
COMP_OUT = ['ZZ','RR','TT']
dtype    = 'Allstack_linear'

# output file format
out_SAC = True
out_TXT = False
three_lags = False

if (not out_SAC) and (not out_TXT):
    raise ValueError('out_SAC and out_TXT cannot be False at the same time')

nfiles = len(ALLFILES)
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# loop through station pairs
for ii in range(nfiles):

    with pyasdf.ASDFDataSet(ALLFILES[ii],mode='r') as ds:
        
        # load data parameters: npts, dt, maxlag etc
        data_dic = load_data_parameters(ds)

        # make sure data exists
        if dtype in data_dic['slist']:
            for comp in COMP_OUT:
                
                # make sure component exists
                if comp in data_dic['rlist']:

                    # get station info from file name
                    filename = make_output_names(ALLFILES[ii],comp,out_SAC)

                    # read the correlations
                    corr = ds.auxiliary_data[dtype][comp].data[:]
                    
                    if out_SAC:
                        save_sac(filename,data_dic,three_lags,corr)

                    if out_TXT:
                        #-------make an array for output-------
                        npts = len(corr)
                        indx = npts//2
                        data = np.zeros((3,indx+2),dtype=np.float32)
                        data[0,0] = data_dic['slon']; data[1,0] = data_dic['slat']; data[2,0] = 0
                        data[0,1] = data_dic['rlon']; data[1,1] = data_dic['rlat']; data[2,1] = 0
                        tt   = 0
                        for jj in range(indx):
                            data[0,2+jj] = tt
                            data[1,2+jj] = corr[indx+jj]
                            data[2,2+jj] = corr[indx-jj]
                            tt = tt+data_dic['dt']
                        
                        np.savetxt(filename,np.transpose(data))
