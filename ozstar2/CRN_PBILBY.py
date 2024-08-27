# This script runs enterprise for individual pulsars, or for an entire gravitational wave source

from __future__ import division

import os, glob, json, pickle, sys
import matplotlib.pyplot as plt
import numpy as np, os, dill, shutil, timeit
import scipy.linalg as sl

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
import enterprise.constants as const
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
from enterprise_extensions.blocks import common_red_noise_block
import corner
import multiprocessing
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

sys.path.insert(0, '/home/mmiles/soft/enterprise_warp/')
#sys.path.insert(0, '/fred/oz002/rshannon/enterprise_warp/')
from enterprise_warp import bilby_warp

sys.path.insert(0, '/home/mmiles/soft/parallel_nested_sampling_pta/')
import schwimmbad_fast
import utils as pb_utils
#from pb_utils import signal_wrapper
from schwimmbad_fast import MPIPoolFast as MPIPool

import bilby
import argparse
import time
import mpi4py
import dynesty
import datetime
import pandas as pd
from pandas import DataFrame
import signal




## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Single pulsar enterprise noise run.")
#parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
parser.add_argument("-pta", dest="pta", help="Enterprise PTA object to load in for the PTMCMC run",required = True)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)

args = parser.parse_args()

ptafile = str(args.pta)
custom_results = str(args.alt_dir)
results_dir = str(args.results)
#while True:
#    try:
pta = dill.load(open(ptafile,"rb"))
#    except SystemError:
#        pass
#    break


# class really_over_sampled:
#     def __init__(
#             self
#     ):
#         #self.pool = None
    
#     #@property
#     #def signal_catcher(self):



def prior_transform_function(theta):
  cube = np.zeros(len(theta))
  for i in range(len(pta.params)):
    cube[i] = ( pta.params[i].prior.func_kwargs['pmax'] - pta.params[i].prior.func_kwargs['pmin'])*theta[i] + pta.params[i].prior.func_kwargs['pmin']
  return list(cube)

def log_likelihood_function(cube):
  x0 = np.hstack(cube)
  return pta.get_lnlikelihood(x0)

def log_prior_function(x):
  return pta.get_lnprior(x)

def try_mkdir(outdir, par_nms):
  if not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
        np.savetxt(outdir+'/pars.txt', par_nms, fmt='%s') ## works as effectively `outdir` is globally declared
    except:
        FileExistsError
  return None


def safe_file_dump(data, filename, module):
    """Safely dump data to a .pickle file

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    os.rename(temp_filename, filename)

def write_current_state(sampler, resume_file, sampling_time, rotate=False):
    """Writes a checkpoint file
    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    rotate: bool
        If resume_file already exists, first make a backup file (ending in '.bk').
    """
    print("")
    print("Start checkpoint writing")
    if rotate and os.path.isfile(resume_file):
        resume_file_bk = resume_file + ".bk"
        print("Backing up existing checkpoint file to {}".format(resume_file_bk))
        shutil.copyfile(resume_file, resume_file_bk)
    sampler.kwargs["sampling_time"] = sampling_time
    if dill.pickles(sampler):
        safe_file_dump(sampler, resume_file, dill)
        print("Written checkpoint file {}".format(resume_file))
    else:
        print("Cannot write pickle resume file!")

def write_current_state_on_kill(signum=None, frame=None):
    """
    Make sure that if a pool of jobs is running only the parent tries to
    checkpoint and exit. Only the parent has a 'pool' attribute.

    For samplers that must hard exit (typically due to non-Python process)
    use :code:`os._exit` that cannot be excepted. Other samplers exiting
    can be caught as a :code:`SystemExit`.
    """
    #if self.pool.rank == 0:
    print("Killed, writing and exiting.")
    #if pool.is_master():
    #rank = mpipool.comm.Get_rank()
    #if mpipool.is_master():
    write_current_state(sampler, resume_file, sampling_time, rotate_checkpoints)
    #pb_utils.write_current_state(pbsampler, resume_file, sampling_time, rotate_checkpoints)
    pb_utils.write_sample_dump(pbsampler, samples_file, sampling_keys)
    pb_utils.plot_current_state(sampler, outdir, label, sampling_keys)
    os._exit(130)


label = results_dir
if custom_results is not None and custom_results != "":
    header_dir = custom_results
else:
    header_dir = "out_pbilby"

outDir=header_dir+'/{0}/'.format(results_dir)
print(outDir)

mpi4py.rc.threads=False
mpi4py.rc.recv_mprobe=False

start_time = time.perf_counter()
path0 = os.getcwd()

par_nms = pta.param_names
Npar = len(par_nms)
ndim = Npar
outdir = outDir

## you might want to change any of the following depending what you are running
nlive=200
tol=0.1
dynesty_sample='rwalk'
dynesty_bound='multi'
walks=100
maxmcmc=5000
nact=10
facc=0.5
min_eff=10.
vol_dec=0.5
vol_check=8.
enlarge=1.5
is_nestcheck=False
n_check_point=5
do_not_save_bounds_in_resume=False
check_point_deltaT=600
n_effective=np.inf
max_its=1e10
max_run_time=1.0e10
rotate_checkpoints=False
rotate_checkpoints = rotate_checkpoints
rstate = np.random

fast_mpi=False
mpi_timing=False
mpi_timing_interval=0
nestcheck_flag=False

try_mkdir(outdir, par_nms)

try:
    with open(outDir+"/run_summary.txt","w") as f:
        print(pta.summary(), file=f)
except:
    pass

# getting the sampling keys
sampling_keys = pta.param_names

t0 = datetime.datetime.now()
sampling_time=0

# signal.signal(signal.SIGTERM, pb_utils.write_current_state_on_kill(pool, pbsampler, resume_file, sampling_time, rotate_checkpoints))
# signal.signal(signal.SIGINT, pb_utils.write_current_state_on_kill(pool, pbsampler, resume_file, sampling_time, rotate_checkpoints))
# signal.signal(signal.SIGALRM, pb_utils.write_current_state_on_kill(pool, pbsampler, resume_file, sampling_time, rotate_checkpoints))

with MPIPool(parallel_comms=fast_mpi,
            time_mpi=mpi_timing,
            timing_interval=mpi_timing_interval,) as mpipool:
    #self.pool = pool
    
    if mpipool.is_master():

        signal.signal(signal.SIGTERM, write_current_state_on_kill)
        signal.signal(signal.SIGINT, write_current_state_on_kill)
        signal.signal(signal.SIGALRM, write_current_state_on_kill)

        POOL_SIZE = mpipool.size
        np.random.seed(1234)
        filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
        resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)
        resume_file = resume_file
        samples_file = "{}/{}_samples.dat".format(outdir, label)
        nestcheck_flag = is_nestcheck
        init_sampler_kwargs = dict(
            nlive=nlive,
            sample=dynesty_sample,
            bound=dynesty_bound,
            walks=walks,
            #rstate=rstate,
            #maxmcmc=maxmcmc,
            #nact=nact,
            facc=facc,
            first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
            #vol_dec=vol_dec,
            #vol_check=vol_check,
            enlarge=enlarge,
            #save_bounds=False,
        )


        pbsampler, sampling_time = pb_utils.read_saved_state(resume_file)
        #sampler = False
        if pbsampler is False:
            live_points = pb_utils.get_initial_points_from_prior(
                ndim,
                nlive,
                prior_transform_function,
                log_prior_function,
                log_likelihood_function,
                mpipool,
            )

            pbsampler = dynesty.NestedSampler(
                log_likelihood_function,
                prior_transform_function,
                ndim,
                pool=mpipool,
                queue_size=POOL_SIZE,
                #print_func=dynesty.utils.print_fn_fallback,
                live_points=live_points,
                #live_points=None,
                use_pool=dict(
                    update_bound=True,
                    propose_point=True,
                    prior_transform=True,
                    loglikelihood=True,
                ),
                **init_sampler_kwargs,
            )
        else:
            pbsampler.pool = mpipool
            pbsampler.M = mpipool.map
            pbsampler.queue_size = POOL_SIZE
            #pbsampler.rstate = np.random
            #pbsampler.nlive=nlive
        sampler_kwargs = dict(
            n_effective=n_effective,
            dlogz=tol,
            save_bounds=not do_not_save_bounds_in_resume,

        )
        
        if dynesty_sample == "rwalk":
            bilby.core.utils.logger.info(
                "Using the bilby-implemented rwalk sample method with ACT estimated walks. "
                f"An average of {2 * nact} steps will be accepted up to chain length "
                f"{maxmcmc}."
            )
            from bilby.core.sampler.dynesty_utils import AcceptanceTrackingRWalk
            from bilby.core.sampler.dynesty import DynestySetupError
            if walks > maxmcmc:
                raise DynestySetupError("You have maxmcmc < walks (minimum mcmc)")
            if nact < 1:
                raise DynestySetupError("Unable to run with nact < 1")
            AcceptanceTrackingRWalk.old_act = None
            dynesty.nestedsamplers._SAMPLING["rwalk"] = AcceptanceTrackingRWalk()
        
        run_time = 0
        print(pbsampler)
        #spbsampler.kwargs["live_points"] = live_points
        pbsampler.kwargs["nlive"] = nlive
        sampler = pbsampler
        
        for it, res in enumerate(pbsampler.sample(**sampler_kwargs)):
            (
                worst,
                ustar,
                vstar,
                loglstar,
                logvol,
                logwt,
                logz,
                logzvar,
                h,
                nc,
                worst_it,
                boundidx,
                bounditer,
                eff,
                delta_logz,
                blob
            ) = res
            i = it - 1
            dynesty.utils.print_fn_fallback(
                res, i, pbsampler.ncall, dlogz=tol
            )
            if (
                it == 0 or it % n_check_point != 0
            ) and it != max_its:
                continue
            iteration_time = (datetime.datetime.now() - t0).total_seconds()
            t0 = datetime.datetime.now()
            sampling_time += iteration_time
            sampling_time = sampling_time
            run_time += iteration_time
            if os.path.isfile(resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
            else:
                last_checkpoint_s = np.inf
            if (
                last_checkpoint_s > check_point_deltaT
                or it == max_its
                or run_time > max_run_time
                or delta_logz < tol
            ):
                pb_utils.write_current_state(pbsampler, resume_file, sampling_time, rotate_checkpoints)
                pb_utils.write_sample_dump(pbsampler, samples_file, sampling_keys)
                pb_utils.plot_current_state(sampler, outdir, label, sampling_keys)
                if it == max_its:
                    print("Max iterations %d reached; stopping sampling."%max_its)
                    #sys.exit(0)
                    break
                if run_time > max_run_time:
                    print("Max run time %e reached; stopping sampling."%max_run_time)
                    #sys.exit(0)
                    break
                if delta_logz < tol:
                    print("Tolerance %e reached; stopping sampling."%tol)
                    #sys.exit(0)
                    break
        # Adding the final set of live points.
        for it_final, res in enumerate(pbsampler.add_live_points()):
            pass
        # Create a final checkpoint and set of plots
        pb_utils.write_current_state(pbsampler, resume_file, sampling_time, rotate_checkpoints)
        pb_utils.write_sample_dump(pbsampler, samples_file, sampling_keys)
        pb_utils.plot_current_state(pbsampler, outdir, label, sampling_keys)

        sampling_time += (datetime.datetime.now() - t0).total_seconds()
        sampling_time = sampling_time
        out = pbsampler.results
        if nestcheck_flag is True:
            ns_run = nestcheck.data_processing.process_dynesty_run(out)
            nestcheck_path = os.path.join(outdir, "Nestcheck")
            try_mkdir(nestcheck_path)
            nestcheck_result = "{}/{}_nestcheck.pickle".format(nestcheck_path, label)
            with open(nestcheck_result, "wb") as file_nest:
                pickle.dump(ns_run, file_nest)
        weights = np.exp(out["logwt"] - out["logz"][-1])
        nested_samples = DataFrame(out.samples, columns=sampling_keys)
        nested_samples["weights"] = weights
        nested_samples["log_likelihood"] = out.logl

        samples = dynesty.utils.resample_equal(out.samples, weights)

        result_log_likelihood_evaluations = pb_utils.reorder_loglikelihoods(unsorted_loglikelihoods=out.logl,unsorted_samples=out.samples,sorted_samples=samples,)

        log_evidence = out.logz[-1]
        log_evidence_err = out.logzerr[-1]
        final_sampling_time = sampling_time

        posterior = pd.DataFrame(samples, columns=sampling_keys)
        nsamples = len(posterior)

        print("Sampling time = {}s".format(datetime.timedelta(seconds=sampling_time)))
        print('log evidence is %f'%log_evidence)
        print('error in log evidence is %f'%log_evidence_err)
        pos = posterior.to_json(orient="columns")

        with open(outdir+"/"+label+"_final_res.json", "w") as final:
            json.dump(pos, final)
        np.savetxt(outdir+"/"+"evidence.txt", np.c_[log_evidence, log_evidence_err, float(datetime.timedelta(seconds=sampling_time).total_seconds())], header="logZ \t logZ_err \t sampling_time_s")
        

end_time = time.perf_counter()
time_taken = end_time  - start_time
print("This took a total time of %f seconds."%time_taken)





