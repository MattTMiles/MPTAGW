'''
The purpose of this script is to estimate the clock corrections from pulsar residuals where the clock correction files are not accounted for
This will return the difference between the estimated clock corrections, the difference between the clock correction file and the estimations.
In addition this will return and save a publication-ready plot detailing the recovered signal.

The parsed arguments and descriptions detail different versions of the signal that can be returned, in short they are described here:
- data: Directory containing IPTA standard timing files and ephemerides of the chosen pulsars
- results: Directory where the results will be output to. The script will create it if it does not already exist.
- solo: Takes a single pulsar name. This optionally recreates the signal from data of a single pulsar.
- residuals: Uses a pre-created file containg information of the pulsar timing residuals. 
    Due to the different formats this file can come in this is NOT RECOMMENDED.
- extended: Activates the nested matrix model to recreate the final signal from a series of pulsars. 
    This method is ~10x slower but is considerably more accurate as it takes into account the true covariance of the clock signal.
- excepted: Takes a single pulsar name. This excepts the chosen pulsar from the recreation of the signal. 
    This is used to test the influence of a single pulsar to the signal.
- scaled: If numerical precision errors are suspected (potential due to large integrations and matrix inversions at the scale of 1e-30),
    this will scale integration inputs and inversion inputs by 1e30 and then return them as expected. Should only be used as a check.
- gw: Includes the expected covariance signal of a gravitaional wave signal to test for the origin of anomalies in the signal.

'''

# Pulsar Timing standard ephemeris information:
# Comparing to BIPM2019
# Using DE438

# Recommended run:
# python clock_recover_experimental.py -datadir <data directory> -results <results directory> -extended


import numpy as np 
import psrchive
import matplotlib.pyplot as plt
import bilby
import scipy.integrate as integrate
import sys
import os 
import glob
import argparse
import pandas as pd
from scipy import interpolate


parser = argparse.ArgumentParser(description="Clock comparison")
parser.add_argument("-data", dest="data", help="data directory to use", required = True)
parser.add_argument("-results", dest="results", help="result directory to use", required = True)
parser.add_argument("-solo", dest="solo", help="(Optional) Adding this flag will make this work on a singular pulsar", required=False)
parser.add_argument("-residuals", dest="residuals", help="(Optional) Adding this flag will make this work on a particular residual file (Not satisfactory)", required=False)
parser.add_argument("-extended",dest="extended", action="store_true", help="(Optional) Adding this flag result in a more advanced nested matrix method", required=False)
parser.add_argument("-excepted", dest="excepted", help="(Optional) Adding this flag will exclude a singular pulsar from the process program data.", required=False)
parser.add_argument("-scaled",dest="scaled", action="store_true", help="(Optional) Adding this flag results in everything being scaled by 1e30 if you suspect numerical precision errors.", required=False)
parser.add_argument("-gw",dest="gw", action="store_true", help="(Optional) Adding this flag adds in an additional covariance matrix with the grav wave amplitude and spectral index into it.", required=False)
args = parser.parse_args()

data_dir = str(args.data)
results_dir = str(args.results)
solo = str(args.solo)
excepted =str(args.excepted)
residuals = str(args.residuals)
ext = args.extended
scaled = args.scaled
gw = args.gw

def pwr_spec_clock(f,clk_amp,clk_gamma):
    #These are filled in based on noise results

    A = clk_amp 
    gamma = clk_gamma 
    f_c = 1 #years
    return ((10**A)**2/(12*(np.pi**2)))*((f/f_c)**(-gamma))

def pwr_spec_clock2(f,clk_gamma):
    #These are filled in based on noise results

    #A = clk_amp 
    gamma = clk_gamma 
    f_c = 1 #years
    return ((f/f_c)**(-gamma))

def pwr_spec_spin(f,spin_amp,spin_gamma):
    #These are filled in based on noise results

    A = spin_amp 
    gamma = spin_gamma 
    f_c = 1 #years
    return ((10**A)**2/(12*(np.pi**2)))*((f/f_c)**(-gamma))

def pwr_spec_dm(f,dm_amp,dm_gamma):
    #These are filled in based on noise results

    A = dm_amp 
    gamma = dm_gamma 
    f_c = 1 #years
    return ((10**A)**2)*((f/f_c)**(-gamma))

def pwr_spec_dm2(f,dm_gamma):
    #These are filled in based on noise results

    #A = dm_amp 
    gamma = dm_gamma 
    f_c = 1 #years
    return ((f/f_c)**(-gamma))

# Specify the paramaters of the simulated signal

npt = 500
mjdpt = np.zeros(npt)

#Use longest timespan pulsar as basis of signal recreation
resfile_1909 = "/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/best_10_data_updated_pars/J1909-3744_residuals.dat"
mjds_1909 = np.loadtxt(resfile_1909,usecols=0)
mk2utc = np.loadtxt('/fred/oz002/rshannon/tempo2/clock/mk2utc.clk',skiprows=7)
mk2utc_mjds = mk2utc[:,0]
global_mjds = mk2utc_mjds[:-1]


mjdmin = np.min(global_mjds)
mjdmax = np.max(global_mjds)


#Create artificial time range for reconstruction
for i in range(npt):
    mjdpt[i] = mjdmin + ((mjdmax-mjdmin)/npt)*(i+0.5)

#define observing span
Tdays = mjdmax - mjdmin

nday=int(Tdays+1)
# Convert days to years
T = Tdays/365.25

#Make directories if they don't exist 

if not os.path.exists(results_dir):
    try:
        os.makedirs(results_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

total_df = []


def master_function(pulsar,residuals=residuals):
    '''
    For each pulsar, this returns the clock signal and covariance that can be inferred from a single pulsar.
    '''

    # Make directories if not already there
    pulsar_dir = results_dir+"/"+pulsar

    if not os.path.exists(pulsar_dir):
        try:
            os.makedirs(pulsar_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    pulsar_cov = pulsar_dir+"/covariance/"

    if not os.path.exists(pulsar_cov):
        try:
            os.makedirs(pulsar_cov)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    os.system("cd "+pulsar_cov)

    #Collect ephemeris
    parfile = glob.glob(data_dir+"/*"+pulsar+"*.par")[0]
    #Collect pulsar timing residuals
    if residuals=="None":
        residuals = glob.glob(data_dir+"/*"+pulsar+"_residuals.dat")[0]


    #Load in data from residuals
    t_res = np.loadtxt(residuals,usecols=2)
    freqs = np.loadtxt(residuals,usecols=1)
    mjds = np.loadtxt(residuals,usecols=0)
    uncs = np.loadtxt(residuals,usecols=3)

    num_toas = len(mjds)

    mjd_1 = mjds[0]
    mjd_end = mjds[-1]
 
    #Inferred clock signal red noise gaussian process paramaters as obtained by Enterprise (noise modelling software)
    clk_amp = -13.35
    clk_gamma = 2.1

    mjd_dummy=np.zeros(nday)

    for i in range(len(mjd_dummy)):
        mjd_dummy[i] = i
    
    #Initialize the 1D covariance matrices
    
    cov_clock = np.zeros(nday)
    cov_spin = np.zeros(nday)
    cov_dm = np.zeros(nday)
    cov_ecorr = np.zeros(nday)
    cov_white = np.zeros(nday)

    cov_gw = np.zeros(nday)

    #Initialise the noise parameters before setting them
    dm_amp = 0
    dm_gamma = 0
    red_amp = 0
    red_gamma = 0
    ecorr = 0
    equad = 0
    efac = 0
    gw_amp = 0
    gw_gamma = 0

    #Search through ephemeris file for noise parameters
    openpar = open(parfile,"r")
    openpar.seek(0)
    
    for line in openpar.readlines():
        if "TNDMAmp" in line:
            dm_amp = float(line.split(" ")[-1])
        if "TNDMGam" in line: 
            dm_gamma = float(line.split(" ")[-1])
        if "#TNRedAmpSpin" in line:
            red_amp = float(line.split(" ")[-1])
            #red_amp = 0
        if "#TNRedGamSpin" in line:
            red_gamma = float(line.split(" ")[-1])
            #red_gamma = 0
        if "TNECORR" in line:
            ecorr = float(line.split(" ")[-1])
            ecorr = (1e-6*ecorr)/(365.25*86400)
        if "TNSECORR" in line:
            ecorr = float(line.split(" ")[-1])
            ecorr = ecorr/np.sqrt(256/3600)
            ecorr = (1e-6*ecorr)/(365.25*86400)
        if "TNGLobalEQ" in line or "TNGlobalEQ" in line:
            equad = float(line.split(" ")[-1])
        if "TNGlobalEF" in line:
            efac = float(line.split(" ")[-1])

    #Implant GW signal if selected
    if gw is not None:
        gw_amp = -14.5
        gw_gamma = 4.33

    
    #Arbitrary noise alterations below (convention)
    #equad = equad*2
    #ecorr = ecorr*2
    #efac = efac*2

    #Define the covariance functions for each noise process for the pulsar
    for i in range(len(mjd_dummy)):
        
        #DM noise constant as per Lentati et al. 2016
        K = 2.410*10**-16

        #time step in years
        time_step = (mjd_dummy[i])/365.25
        cov_dm[i] = ((1/(K**2))*((10**dm_amp)**2)*integrate.quad(lambda f: \
            (pwr_spec_dm2(f,dm_gamma)*np.cos(2*np.pi*f*time_step))\
                ,1/T,365.25/28)[0])

        cov_clock[i] = ((10**clk_amp)**2/(12*(np.pi**2)))*integrate.quad(lambda f: \
                (pwr_spec_clock2(f,clk_gamma)*np.cos(2*np.pi*f*time_step))\
                    ,1/T,365.25/28)[0]

        cov_spin[i] = ((10**red_amp)**2/(12*(np.pi**2)))*integrate.quad(lambda f: \
            (pwr_spec_clock2(f,red_gamma)*np.cos(2*np.pi*f*time_step))\
                ,1/T,365.25/28)[0]

        cov_gw[i] = ((10**gw_amp)**2/(12*(np.pi**2)))*integrate.quad(lambda f: \
            (pwr_spec_clock2(f,gw_gamma)*np.cos(2*np.pi*f*time_step))\
                ,1/T,365.25/28)[0]

    
    #Initialize the covariance matrices

    C_clock = np.zeros((num_toas,num_toas))
    C_spin = np.zeros((num_toas,num_toas))
    C_dm = np.zeros((num_toas,num_toas))
    C_ecorr = np.zeros((num_toas,num_toas))
    C_white = np.zeros((num_toas,num_toas))

    C_gw = np.zeros((num_toas,num_toas))
    
    #Populate the covariance matrices
    for i in range(num_toas):
        #print("{}".format(i))
        #equad = -8
        C_white[i,i] =(((efac)*((uncs[i]*10**-6)/(365.25*86400)))**2) + (((10**(equad))/(365.25*86400))**2)
        
        for j in range(num_toas):
            
            interp_i = abs(int(mjds[j]-mjds[i]))
            
            #C_spin[i,j] = 0
            if red_amp !=0:
                #C_spin[i,j] = integrate.quad(lambda f: \
                #    (pwr_spec_spin(f,red_amp,red_gamma)*np.cos(2*np.pi*f*time_step))\
                #        ,1/T,num_toas/T)[0]
                C_spin[i,j] = cov_spin[interp_i]

            if dm_amp != 0:
                C_dm[i,j] = cov_dm[interp_i]/((freqs[i]**2)*(freqs[j]**2))
            else:
                C_dm[i,j] = 0

            if gw is not None:
                C_gw[i,j] = cov_gw[interp_i]
            else:
                C_gw[i,j] = 0

            
            if abs(mjds[i]-mjds[j]) < 0.001:
                C_ecorr[i,j] = (ecorr)**2
                #C_ecorr[i,j] = 0
            else:
                C_ecorr[i,j] = 0


            C_clock[i,j] = cov_clock[interp_i]

    np.save(pulsar_cov+"/"+pulsar+"_C_clock",C_clock)
    np.save(pulsar_cov+"/"+pulsar+"_C_spin",C_spin)
    np.save(pulsar_cov+"/"+pulsar+"_C_dm",C_dm)
    np.save(pulsar_cov+"/"+pulsar+"_C_ecorr",C_ecorr)
    np.save(pulsar_cov+"/"+pulsar+"_C_white",C_white)

    #Mid-run check to see that the values are not out by a significant order
    print(pulsar)
    print("C_clock: {}; C_spin: {}; C_dm: {}; C_ecorr: {}; C_white: {}".format(C_clock[0][0],C_spin[0][0],C_dm[0][0],C_ecorr[0][0],C_white[0][0]))
    
    if gw is not None:
        Cov_total = C_clock + C_spin + C_dm + C_ecorr + C_white + C_gw
        #Cov_total = C_clock + C_spin + C_ecorr + C_white 
        Cov_total_no_clock = C_spin + C_ecorr + C_white + C_gw
        #Cov_total_no_clock = C_spin + C_dm + C_ecorr + C_white
    else:
        Cov_total = C_clock + C_spin + C_dm + C_ecorr + C_white
        #Cov_total = C_clock + C_spin + C_ecorr + C_white 
        #Cov_total_no_clock = C_spin + C_ecorr + C_white
        Cov_total_no_clock = C_spin + C_dm + C_ecorr + C_white


    np.save(pulsar_cov+"/"+pulsar+"_Cov_total",Cov_total)

    #Invert the combined covariance matrices
    Cov_inv = np.linalg.inv(Cov_total)

    ngrand = num_toas + npt
    Cov_simulated = np.zeros((ngrand,ngrand))

    #Populate the recreated "simulated" signal
    for i in range(num_toas):
        for j in range(num_toas):
            Cov_simulated[i][j] = Cov_total_no_clock[i][j]
    
    Cov_clock_simulated = np.zeros((ngrand,ngrand))

    #Populate the global clock signal
    for i in range(ngrand):
        for j in range(ngrand):

            if i < num_toas:
                mjd1 = mjds[i]
            else:
                mjd1 = mjdpt[i-num_toas]
            
            if j < num_toas:
                mjd2 = mjds[j]
            else:
                mjd2 = mjdpt[j-num_toas]
            
            imjd = abs(int(mjd2-mjd1))

            Cov_clock_simulated[i][j] = cov_clock[imjd]

    #Combine the covariance matrices of the clock signal and the pulsar intrinsic noise processes
    Cov_simulated = Cov_simulated + Cov_clock_simulated
    
    Cov_inv_simulated = np.linalg.inv(Cov_simulated)

    t_res_sim = np.zeros(ngrand)

    for i in range(num_toas):
        t_res_sim[i] = t_res[i]

    #Find the associaed uncertaities
    error_temp = np.matmul(Cov_inv_simulated,Cov_clock_simulated)
    sim_clock_matrix = Cov_clock_simulated - np.matmul(Cov_clock_simulated,error_temp)
    sim_clock_error = np.diagonal(sim_clock_matrix)
    sim_clock_seconds = np.sqrt(np.abs(sim_clock_error))*(365.25*86400)

    #Find the reconstructed clock waveform
    temp_waveform = np.matmul(Cov_inv,t_res)
    clk_waveform = np.matmul(C_clock,temp_waveform)

    np.save(pulsar_cov+"/"+pulsar+"_clk_waveform",clk_waveform)

        
    temp_sim = np.matmul(Cov_inv_simulated,t_res_sim)
    clk_sim = np.matmul(Cov_clock_simulated,temp_sim)

    clk_sim_tosave = clk_sim[num_toas:]
    print("length is: {}".format(len(clk_sim_tosave)))
    
    np.save(pulsar_cov+"/"+pulsar+"_clk_sim_waveform",clk_sim_tosave)

    clk_error_tosave = sim_clock_seconds[num_toas:]
    np.save(pulsar_cov+"/"+pulsar+"_clk_sim_error_array",clk_error_tosave)

    
    mk2utc = np.loadtxt('/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/mk2utc.clk',skiprows=7)

    plt.plot(mjds,clk_waveform,label='reconstructed_clock_{}'.format(pulsar))
    plt.plot(mjdpt,clk_sim[num_toas:],label='simulated_clock_{}'.format(pulsar),color='xkcd:green')
    plt.fill_between(mjdpt, clk_sim[num_toas:] - sim_clock_seconds[num_toas:], clk_sim[num_toas:] + sim_clock_seconds[num_toas:],alpha=0.2,color="xkcd:green")
    plt.plot(mk2utc[:,0],-mk2utc[:,1],label='mk2utc')
    plt.title("How {} sees the clock".format(pulsar))
    plt.legend()
    plt.savefig(pulsar_cov+"/"+pulsar+"_clock_vs_MKT")
    plt.close()

    return Cov_total_no_clock, cov_clock, t_res, mjds
    #return Cov_total, cov_clock, t_res, mjds

# Condition if no extended nested matrices algorithm to be used
if ext==False:
    if args.solo is None:
        os.chdir(data_dir)
        for par in glob.glob("J*par"):
            if excepted is not None:
                pulsar = par.split(".")[0]
                if not pulsar == excepted:
                    print(pulsar)
                    master_function(pulsar,residuals)
            else:
                pulsar = par.split(".")[0]
                print(pulsar)
                master_function(pulsar,residuals)

    if args.solo is not None:
        os.chdir(data_dir)
        pulsar = solo
            #pulsar = par.split(".")[0]
        print(pulsar)
        
        master_function(pulsar,residuals)

# Condition where the extended nested matrix algorithm
if ext==True:
    if args.solo is not None:
        print("Extended is only for where there are multiple pulsars. It won't do anything if there's only one.")

    if args.solo is None and args.scaled is None:
        os.chdir(data_dir)
        if args.excepted is not None:
            num_pulsars = len(set(glob.glob("J*par"))-set(glob.glob("*"+excepted+"*")))
        else:
            num_pulsars = len(glob.glob("J*par"))

        # Create large scale nested covariance matrix 
        master_cov = np.zeros((num_pulsars,num_pulsars),dtype=object)
        master_cov_clock = np.zeros((num_pulsars,num_pulsars),dtype=object)
        master_res = np.zeros(num_pulsars,dtype=object)
        master_mjds = np.zeros(num_pulsars,dtype=object)

        
        total_length = []
        #Condition where a single pulsar is removed from the reconstruction
        if args.excepted is not None:
            for i, par in enumerate(set(glob.glob("J*par"))-set(glob.glob("*"+excepted+"*"))):

                pulsar = par.split(".")[0]
                if not pulsar==excepted:
                    print(pulsar)
                    #Calls master function and collects each pulsar's information
                    Cov_total_noClock, cov_clock, t_res, mjds = master_function(pulsar,residuals)
                    #Populate the nested covariance matrix for each pulsar
                    master_cov[i][i] = Cov_total_noClock
                    master_cov_clock = cov_clock
                    master_res[i] = t_res
                    length = len(t_res)
                    total_length.append(length)
                    master_mjds[i] = mjds
            
        else:
            for i, par in enumerate(glob.glob("J*par")):
                pulsar = par.split(".")[0]
                print(pulsar)
            
                Cov_total_noClock, cov_clock, t_res, mjds = master_function(pulsar,residuals)
                #plt.show()
                
                #Cov_total_in = np.zeros((2837, 2837),dtype=np.float64)
                #Cov_total_in[:Cov_total.shape[0],:Cov_total.shape[1]] = Cov_total
                master_cov[i][i] = Cov_total_noClock
                master_cov_clock = cov_clock
                master_res[i] = t_res
                length = len(t_res)
                total_length.append(length)
                master_mjds[i] = mjds

        
        total_length = np.sum(total_length)
        total_length_npt = total_length+npt
        
        
        #Initialise the covariance matrices
        master_cov_inv_sim = np.zeros((total_length_npt,total_length_npt),dtype=np.float64)
        master_cov_sim = np.zeros((total_length_npt,total_length_npt),dtype=np.float64)
        master_cov_clock_sim = np.zeros((total_length_npt,total_length_npt),dtype=np.float64)
        eval_master_cov = np.zeros((total_length,total_length),dtype=np.float64)

        ntoa_master = 0
        
        for i in range(len(master_res)):
            ntoa_int = len(master_res[i])
            
            print("Pulsar {}".format(i+1))
            for j in range(ntoa_int):
                for k in range(ntoa_int):
                    #To account for the 0's not being of matrix form do this try/except loop
                    try:
                        eval_master_cov[j+ntoa_master][k+ntoa_master] = master_cov[i][i][j][k]
                    except TypeError:
                        eval_master_cov[j+ntoa_master][k+ntoa_master] = 0
                    
            ntoa_master = ntoa_master + ntoa_int

        #initialise the size of the total reconstructed covariance matrix
        Cov_clock = np.zeros((total_length,total_length))

        master_mjds_active = np.hstack(master_mjds)
        master_res_active = np.hstack(master_res)

        #Create a large covariance matrix for the clock signal of the same size as the nested matrix
        for i in range(ntoa_master):
            for j in range(ntoa_master):

                interp_i = abs(int(master_mjds_active[j]-master_mjds_active[i]))

                Cov_clock[i,j] = master_cov_clock[interp_i]
        

        print("Creating master simulated clock")
        for i in range(total_length_npt):
            for j in range(total_length_npt):

                if i < len(master_res_active):
                    mjd1 = master_mjds_active[i]
                else:
                    mjd1 = mjdpt[i-len(master_res_active)]
                
                if j < len(master_res_active):
                    mjd2 = master_mjds_active[j]
                else:
                    mjd2 = mjdpt[j-len(master_res_active)]
                
                imjd = abs(int(mjd2-mjd1))
                master_cov_clock_sim[i][j] = master_cov_clock[imjd]
        
        #Combine the nested matrix and the clock signal matrix
        cov_total = eval_master_cov + Cov_clock
        #Invert
        master_cov_inv = np.linalg.inv(cov_total)

        #Create the simulated matrix mirroring the recreation over an arbitrary date range
        for i in range(total_length):
            for j in range(total_length):
                master_cov_inv_sim[i][j] = master_cov_inv[i][j]

        
        master_t_res_sim = np.zeros(total_length_npt,dtype=np.float64)

        # Where data exists, collect the timing residuals
        for i in range(ntoa_master):
            master_t_res_sim[i] = master_res_active[i]

        #Creates the recreated clock signal as per the nested matrix algorithm
        print("Creating master waveform")
        master_temp_waveform_sim = np.matmul(master_cov_inv_sim,master_t_res_sim)
        master_clk_sim = np.matmul(master_cov_clock_sim,master_temp_waveform_sim)

        master_clk_to_save = master_clk_sim[ntoa_master:]
        np.save(results_dir+"/clk_sim_waveform",master_clk_to_save)
        
        #Create the associated uncertainties
        print("Creating error array")
        master_error_temp = np.matmul(master_cov_inv_sim,master_cov_clock_sim)
        master_sim_clock_matrix = master_cov_clock_sim - np.matmul(master_cov_clock_sim,master_error_temp)
        master_sim_clock_error = np.diagonal(master_sim_clock_matrix)
        master_sim_clock_seconds = np.sqrt(np.abs(master_sim_clock_error))*(365.25*86400)

        master_error_to_save = master_sim_clock_seconds[ntoa_master:]
        np.save("/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/clk_sim_error",master_error_to_save)
        
        #Isolate the reported clock signal for comparison
        mk2utc = np.loadtxt('/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/mk2utc.clk',skiprows=7)
        mk2utc_error = 4.925*1e-9

        #Plot the outcome
        font = 20
        fig, axs = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
        fig.subplots_adjust(hspace=0)
        axs[0].plot(mjdpt,master_clk_to_save-mean_to_take, label='Recovered Clock Signal',color='xkcd:green')
        axs[0].fill_between(mjdpt, (master_clk_to_save - master_error_to_save) - mean_to_take, (master_clk_to_save + master_error_to_save) - mean_to_take,alpha=0.2,color="xkcd:green")
        axs[0].plot(mk2utc[:,0],(-mk2utc[:,1]) - np.mean(-mk2utc[:,1]),label='mk2utc', color='tab:blue')
        axs[0].fill_between(mk2utc[:,0], (-mk2utc[:,1] - mk2utc_error) - np.mean(-mk2utc[:,1]), (-mk2utc[:,1] + mk2utc_error) - np.mean(-mk2utc[:,1]), alpha=0.2, color='tab:blue')
        axs[1].scatter(master_mjds_active, master_res_active, marker='x', color='black', alpha=0.2)

        axs[0].set_title("Recovered clock signal",fontsize=font)
        axs[1].ticklabel_format(axis="y",style="sci",scilimits=(-6,-6))
        axs[1].yaxis.offsetText.set_fontsize(16)
        axs[0].yaxis.offsetText.set_fontsize(16)
        axs[1].tick_params(axis='both', which='major', labelsize=font)
        axs[0].ticklabel_format(axis="y",style="sci",scilimits=(-6,-6))
        axs[0].tick_params(axis='both', which='major', labelsize=font)
        axs[1].set_xlabel("MJD",fontsize=font)
        axs[1].set_ylabel("Residuals (s)",fontsize=font)
        axs[0].set_ylabel("Clock signals (s)",fontsize=font)
        fig.legend(fontsize=font)
        fig.savefig(results_dir+"/Clock_sim_MJDS")
        fig.show()


    #Description of the below is equivalent to above.
    if args.solo is None and args.scaled is not None:
        os.chdir(data_dir)
        if args.excepted is not None:
            num_pulsars = len(set(glob.glob("J*par"))-set(glob.glob("*"+excepted+"*")))
        else:
            num_pulsars = len(glob.glob("J*par"))

        
        master_cov = np.zeros((num_pulsars,num_pulsars),dtype=object)
        master_cov_clock = np.zeros((num_pulsars,num_pulsars),dtype=object)
        master_res = np.zeros(num_pulsars,dtype=object)
        master_mjds = np.zeros(num_pulsars,dtype=object)


        total_length = []
        if args.excepted is not None:
            for i, par in enumerate(set(glob.glob("J*par"))-set(glob.glob("*"+excepted+"*"))):

                pulsar = par.split(".")[0]
                if not pulsar==excepted:
                    print(pulsar)
                
                    Cov_total_noClock, cov_clock, t_res, mjds = master_function(pulsar,residuals)

                    master_cov[i][i] = 1e30*Cov_total_noClock
                    master_cov_clock = cov_clock
                    master_res[i] = t_res
                    length = len(t_res)
                    total_length.append(length)
                    master_mjds[i] = mjds
            
        else:
            for i, par in enumerate(glob.glob("J*par")):
                pulsar = par.split(".")[0]
                print(pulsar)
            
                Cov_total_noClock, cov_clock, t_res, mjds = master_function(pulsar,residuals)

                master_cov[i][i] = 1e30*Cov_total_noClock
                master_cov_clock = cov_clock
                master_res[i] = t_res
                length = len(t_res)
                total_length.append(length)
                master_mjds[i] = mjds

        
        total_length = np.sum(total_length)
        total_length_npt = total_length+npt
        

        master_cov_inv_sim = np.zeros((total_length_npt,total_length_npt),dtype=np.float64)
        master_cov_sim = np.zeros((total_length_npt,total_length_npt),dtype=np.float64)
        master_cov_clock_sim = np.zeros((total_length_npt,total_length_npt),dtype=np.float64)
        eval_master_cov = np.zeros((total_length,total_length),dtype=np.float64)

        ntoa_master = 0
        
        for i in range(len(master_res)):
            ntoa_int = len(master_res[i])
            
            print("Pulsar {}".format(i+1))
            for j in range(ntoa_int):
                for k in range(ntoa_int):

                    try:
                        eval_master_cov[j+ntoa_master][k+ntoa_master] = master_cov[i][i][j][k]
                    except TypeError:
                        eval_master_cov[j+ntoa_master][k+ntoa_master] = 0
                    
            ntoa_master = ntoa_master + ntoa_int

        Cov_clock = np.zeros((total_length,total_length))
    

        master_mjds_active = np.hstack(master_mjds)
        master_res_active = np.hstack(master_res)

        for i in range(ntoa_master):
            for j in range(ntoa_master):

                interp_i = abs(int(master_mjds_active[j]-master_mjds_active[i]))

                Cov_clock[i,j] = 1e30*master_cov_clock[interp_i]
        

        print("Creating master simulated clock")
        for i in range(total_length_npt):
            for j in range(total_length_npt):

                if i < len(master_res_active):
                    mjd1 = master_mjds_active[i]
                else:
                    mjd1 = mjdpt[i-len(master_res_active)]
                
                if j < len(master_res_active):
                    mjd2 = master_mjds_active[j]
                else:
                    mjd2 = mjdpt[j-len(master_res_active)]
                
                imjd = abs(int(mjd2-mjd1))
                master_cov_clock_sim[i][j] = 1e30*master_cov_clock[imjd]
        

        cov_total = eval_master_cov + Cov_clock
        
        master_cov_inv = np.linalg.inv(cov_total)

        for i in range(total_length):
            for j in range(total_length):
                master_cov_inv_sim[i][j] = master_cov_inv[i][j]

        
        master_t_res_sim = np.zeros(total_length_npt,dtype=np.float64)


        for i in range(ntoa_master):
            master_t_res_sim[i] = master_res_active[i]

        print("Creating master waveform")
        master_temp_waveform_sim = np.matmul(master_cov_inv_sim,master_t_res_sim)
        master_clk_sim = np.matmul(master_cov_clock_sim,master_temp_waveform_sim)

        master_clk_to_save = master_clk_sim[ntoa_master:]
        np.save(results_dir+"/clk_sim_waveform",master_clk_to_save)
        np.save(results_dir+"/master_mjds_active",master_mjds_active)
        np.save(results_dir+"/master_res_active",master_res_active)
        
        print("Creating error array")
        master_error_temp = np.matmul(master_cov_inv_sim,master_cov_clock_sim)
        master_sim_clock_matrix = master_cov_clock_sim - np.matmul(master_cov_clock_sim,master_error_temp)
        master_sim_clock_error = np.diagonal(master_sim_clock_matrix)
        master_sim_clock_seconds = 1e-15*np.sqrt(np.abs(master_sim_clock_error))*(365.25*86400)

        master_error_to_save = master_sim_clock_seconds[ntoa_master:]
        np.save("/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/clk_sim_error",master_error_to_save)

        mk2utc = np.loadtxt('/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/mk2utc.clk',skiprows=7)
        mk2utc_error = 4.925*1e-9

        arg_ind = np.argwhere(mjdpt>mk2utc[-1,0])[0]
        subset_clk = master_clk_to_save[:arg_ind[0]]
        mean_to_take = np.mean(subset_clk)

        mjd_sub = mjdpt[mjdpt<mk2utc[-1,0]]
        clk_sub = master_clk_to_save[mjdpt<mk2utc[-1,0]]
        error_sub = master_error_to_save[mjdpt<mk2utc[-1,0]]

        np.save("/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/1909_sim/recovered_clock/clk_recover",clk_sub)
        np.save("/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/1909_sim/recovered_clock/clk_recover_error",error_sub)
        np.save("/fred/oz002/users/mmiles/MSP_DR/clock_correction_work/enterprise_version/nanograv_clones/clocks/1909_sim/recovered_clock/clk_MJD",mjd_sub)


        new_x = np.linspace(58485, 59612, num=438)
        rec_step = interpolate.splrep(mjd_sub,(clk_sub-mean_to_take)*1e6,s=0)
        rec_interp = interpolate.splev(new_x,rec_step,der=0)
        

        mk_step = interpolate.splrep(mk2utc[:,0],((-mk2utc[:,1]) - np.mean(-mk2utc[:,1]))*1e6,s=1)
        mk_interp = interpolate.splev(new_x,mk_step,der=0)

        res = mk_interp-rec_interp
        err = error_sub*1e6
        quad_err = np.sqrt(err**2 + (mk2utc_error*1e6)**2)

        font = 20
        fig, axs = plt.subplots(3, 1, sharex=True,gridspec_kw={'height_ratios': [3, 3, 1]},figsize=(15,10))
        fig.subplots_adjust(hspace=0)
        axs[0].plot(mjd_sub,(clk_sub-mean_to_take)*1e6, label='Recovered Clock',color='xkcd:green')
        axs[0].fill_between(mjd_sub, ((clk_sub - error_sub) - mean_to_take)*1e6, ((clk_sub + error_sub) - mean_to_take)*1e6,alpha=0.2,color="xkcd:green")
        axs[0].plot(mk2utc[:,0],((-mk2utc[:,1]) - np.mean(-mk2utc[:,1]))*1e6,label='KTT-UTC(GPS)', color='tab:blue')
        axs[0].fill_between(mk2utc[:,0], ((-mk2utc[:,1] - mk2utc_error) - np.mean(-mk2utc[:,1]))*1e6, ((-mk2utc[:,1] + mk2utc_error) - np.mean(-mk2utc[:,1]))*1e6, alpha=0.2, color='tab:blue')
        axs[2].scatter(master_mjds_active, [0]*len(master_mjds_active), marker='x', color='black', alpha=0.2)

        axs[1].plot(new_x,res,label="Residual",color="xkcd:teal")
        axs[1].fill_between(new_x,res-quad_err,res+quad_err,alpha=0.2,color="xkcd:teal")
        axs[1].axhline(0,linestyle="--",color="xkcd:grey")
        axs[1].fill_between(new_x,-0.05,0.05,alpha=0.15,color="xkcd:grey")

        axs[2].yaxis.offsetText.set_fontsize(16)
        axs[1].yaxis.offsetText.set_fontsize(16)
        axs[0].yaxis.offsetText.set_fontsize(16)
        axs[2].tick_params(axis='both', which='major', labelsize=font)
        axs[1].tick_params(axis='both', which='major', labelsize=font)
        #axs[0].ticklabel_format(axis="y",style="sci",scilimits=(-6,-6))
        axs[0].tick_params(axis='both', which='major', labelsize=font)
        axs[2].set_xlabel("MJD",fontsize=font)
        #axs[1].set_ylabel("Residuals ($\mu$s)",fontsize=font)
        axs[2].yaxis.set_visible(False)
        axs[0].set_ylabel("Clock signals ($\mu$s)",fontsize=font)
        axs[1].set_ylabel("Residual ($\mu$s)",fontsize=font)
        fig.legend(fontsize=font-2,bbox_to_anchor=(0.90, 1.01))
        #fig.tight_layout()
        fig.savefig('/fred/oz002/users/mmiles/MSP_DR/paper_plots/clock_recovery_residual.pdf',dpi=150)
        fig.show()





        
        

