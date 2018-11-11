# -*- coding: utf-8 -*-

"""Console script for pyepi."""
import click
import time
import sys
from .interfaces import freesurfer
from .interfaces import fsl
from .tools import paths
from . import pipelines
import tempfile
import os
import ast
import subprocess
import shutil
import platform

RAW_DATA, RAW_DATA_NATIVE, SUBJECTS_DIR, SUBJECTS_DIR_NATIVE = paths.set_paths(platform=platform.node())


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.argument('job')
@click.option('--subject', default=None, help='subjects ID')
@click.option('--subjects_dir', default=None, help="Freesurfer's SUBJECTS_DIR (needed for [cvs-mni2subj])")
@click.option('--t1_file', default=None, help='T1 input volume (can be a dicom or some other volume)[recon]')
@click.option('--t2_file', default=None, help='T2 input volume (can be a dicom or some other volume)[recon]')
@click.option('--openmp', default=1, help='number of parallel threads for OpenMP [recon, cvs, tracula]')
@click.option('--dcm_file', default=None, help='DICOM file to be converted to Nifti [dcm2nii]')
@click.option('--output_nii_filename', default=None, help='Name of Nifti output volume [dcm2nii]')
@click.option('--output_nii_folder', default=None, help='Path to save Nifti output volume to [dcm2nii]')
@click.option('--trac_prep', default=None, help='Diffusion preprocessing [tracula]')
@click.option('--trac_bedp', default=None, help="FSL's BedpostX [tracula]")
@click.option('--trac_path', default=None, help='Pathways reconstruction [tracula]')
@click.option('--dwi_file', default=None,
              help='Path to diffusion DICOM or nifti file output of dcm2niix (preferred) [tracula]')
@click.option('--bvecfile', default=None, help='Path to .bvec file (use with nifti input from dcm2niix) [tracula]')
@click.option('--bvalfile', default=None, help='Path to .bval file (use with nifti input from dcm2niix) [tracula]')
# @click.option('--trac-root', default=None, help="Output folder for Tracula's results [tracula]")
@click.option('--doeddy', default=1,
              help='Perform registration-based eddy-current compensation? (1 or 0) [tracula]')
@click.option('--dorotbvecs', default=1,
              help='Rotate diffusion gradient vectors to match eddy-current compensation? (1 or 0)  [tracula]')
@click.option('--doregbbr', default=1,
              help='Perform diffusion-to-T1 registration by bbregister? (1 or 0) [tracula]')
@click.option('--doregmni', default=1, help='Perform registration of T1 to MNI template? (1 or 0) [tracula]')
@click.option('--doregcvs', default=0, help='Perform registration of T1 to CVS template? (1 or 0) [tracula]')
@click.option('--nstick', default=2,
              help='Number of "sticks" (anisotropic diffusion compartments) in the bedpostx [tracula]')
@click.option('--nburnin', default=200, help='Number of MCMC burn-in iterations [tracula]')
@click.option('--niters', default=7500, help='Number of MCMC iterations [tracula]')
@click.option('--nkeep', default=5,
              help='Frequency with which MCMC path samples are retained for path distribution [tracula]')
@click.option('--subjects_dir', default=None, help='Use a different SUBJECTS_DIR for processing this subject [tracula]')
@click.option('--verbose', default=0, help='If =1 will print the output of all functions.')
# probtrac
def preproc(job, **kwargs):
    """Script for running various pre-processing pipelines.

    \b
    Assumes the following are *always* defined as variables in the shell:
        FREESURFER_HOME -- freesurfer's home folder, usually '/usr/local/freesurfer'
        SUBJECTS_DIR    -- freesurfer's subjects folder, usually '/usr/local/freesurfer/subjects'
        FSLDIR          -- FSL's dir, usually 'usr/local/fsl'


    \b
    JOB can be:
     'recon'             -- Freesurfer's recon
     'cvs-subj2mni'      -- CVS registration of subject's recon to cvs_avg35_inMNI152 template
     'cvs-mni2subj'      -- CVS registration of cvs_avg35_inMNI152 template to subject's recon
     'tracula'           -- Freesurfer's tracula (Diffusion preprocessing, Bedpost and Pathways reconstruction)
     'dcm2nii'           -- Convert DICOM series to Nifti using dcm2niix (executable needs to be in PATH variable,
                            download it from https://github.com/rordenlab/dcm2niix/releases)

    \b
    Use underscore to combine multiple jobs : 'recon_cvs2mni_tracula' to run all three. If multiple jobs are lister,
    they will always run in the following order:
        RECON --> CVS --> DCM2NIIX --> TRACULA
    Default values specified in Freesurfer's and FSL's documentation are always used if no alternatives are provided.
    """

    jobs = job.split('_')

    if 'recon' in jobs:
        params_of_interest = ['subject', 't1_file', 't2_file', 'openmp', 'verbose']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print('\n * Running RECON with the following parameters:')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        tstart = time.time()
        print(' + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        freesurfer.recon(subj=kwargs['subject'], t1_file=kwargs['t1_file'], t2_file=kwargs['t2_file'],
                         openmp=kwargs['openmp'], verbose=kwargs['verbose'])
        print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')

    if 'cvs-subj2mni' in jobs:
        params_of_interest = ['subject', 'openmp', 'verbose']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print('\n * Running CVS subject to cvs_avg35_inMNI135 template with the following parameters:')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        tstart = time.time()
        print(' + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        freesurfer.cvs_subj2mni(subj=kwargs['subject'], openmp=kwargs['openmp'], verbose=kwargs['verbose'])
        print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')

    if 'cvs-mni2subj' in jobs:
        params_of_interest = ['subject', 'openmp', 'verbose', 'subjects_dir']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print('\n * Running CVS cvs_avg35_inMNI135 template to subject with the following parameters:')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        tstart = time.time()
        print(' + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        freesurfer.cvs_mni2subj(subj=kwargs['subject'], openmp=kwargs['openmp'], verbose=kwargs['verbose'],
                                subjects_dir=kwargs['subjects_dir'])
        print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')

    if 'dcm2nii' in jobs:
        params_of_interest = ['dcm_file', 'output_nii_filename', 'output_nii_folder', 'verbose']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print('\n * Running DICOM to Nifti conversion with the following parameters:')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        tstart = time.time()
        print(' + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        freesurfer.dcm2niix(dcm_file=kwargs['dcm_file'], output_filename=kwargs['output_nii_filename'],
                            output_folder=kwargs['output_nii_folder'], verbose=kwargs['verbose'])
        print(' + Finished in ' + str((time.time() - tstart)) + ' seconds.')

    if 'tracula' in jobs:
        tmpdir = tempfile.TemporaryDirectory()
        # tracula config file first
        params_of_interest = ['subject', 'dwi_file', 'bvecfile', 'bvalfile', 'doeddy', 'dorotbvecs', 'subjects_dir',
                              'doregbbr', 'doregmni', 'doregcvs', 'nstick', 'nburnin', 'niters', 'nkeep']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print(
                '''\n * Creating Tracula's configuration file with the following parameters 
                (other defaults may be automatically added to the configuration file, check Tracula's documentation:''')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        cfg = freesurfer.tracula_config(subj=kwargs['subject'], dicom=kwargs['dwi_file'], config_folder=None,
                                        # tmpdir.name,
                                        bvecfile=kwargs['bvecfile'], bvalfile=kwargs['bvalfile'],
                                        doeddy=kwargs['doeddy'],
                                        dorotbvecs=kwargs['dorotbvecs'], doregbbr=kwargs['doregbbr'],
                                        doregmni=kwargs['doregmni'], doregcvs=kwargs['doregcvs'],
                                        nstick=kwargs['nstick'],
                                        nburnin=kwargs['nburnin'], nsample=kwargs['niters'], nkeep=kwargs['nkeep'],
                                        subjects_dir=kwargs['subjects_dir'])

        # if running on Windows Linux Subsystem the path to cfg needs to be changed to point to /mnt/c/Users/...
        if sys.platform == 'win32':
            breakdown = cfg.replace(':', '').split(sep='\\')
            breakdown[0] = breakdown[0].lower()
            breakdown = ['mnt'] + breakdown
            cfg_linux = '/' + '/'.join(breakdown)
        else:
            cfg_linux = cfg

        # run tracula
        params_of_interest = ['subject', 'trac_prep', 'trac_bedp', 'trac_path', 'verbose']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print('\n * Running Tracula with the following parameters:')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        if kwargs['trac_prep'] == '1':
            tstart = time.time()
            print(' + Starting Tracula pre-processing step at : ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                  time.localtime()))
            freesurfer.tracula_run(subj=kwargs['subject'], prep=True, bedp=False, path=False, cfg_file=cfg_linux,
                                   verbose=kwargs['verbose'])
            print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')
        if kwargs['trac_bedp'] == '1':
            tstart = time.time()
            print(' + Starting Tracula bedpostX step at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            freesurfer.tracula_run(subj=kwargs['subject'], prep=False, bedp=True, path=False, cfg_file=cfg_linux,
                                   verbose=kwargs['verbose'])
            print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')
        if kwargs['trac_path'] == '1':
            tstart = time.time()
            print(' + Starting Tracula pathways reconstruction step at : ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                           time.localtime()))
            freesurfer.tracula_run(subj=kwargs['subject'], prep=False, bedp=False, path=True, cfg_file=cfg_linux,
                                   verbose=kwargs['verbose'])
            print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')
        paths.silentremove(cfg)

    return 0


@click.command()
@click.argument('job')
@click.option('--subject', default=None, help='subjects ID')
@click.option('--bedpostx_folder', default=None, help="Path to subject's bedpostX folder")
@click.option('--seedmask', default=None, help="Nifti seed volume")
@click.option('--seed2diff_xfm', default=None, help="Linear transformation matrix from seed space to diffusion space")
@click.option('--avoidmask', default=None, help="ROI to be avoided when performing tractography")
@click.option('--terminationmask', default=None, help="ROI where fibers are terminated")
@click.option('--probtrack_dir', default=None, help="Path to folder where output will be generated")
@click.option('--waypoints', default=None,
              help="Path to Nifti volume which will be used as ROI. Multiple volumes can be used.")
@click.option('--waycond', default='AND', help="Waypoint condition, can be 'AND' , 'OR'")
@click.option('--loopcheck', default=True,
              help="Perform loopchecks on paths - slower, but allows lower curvature threshold")
@click.option('--onewaycondition', default=True, help="Apply waypoint conditions to each half tract separately")
@click.option('--curvature_thr', default=0.2, help="Curvature threshold - default=0.2")
@click.option('--nsamples', default=5000, help="Number of samples - default=5000")
@click.option('--nsteps', default=2000, help="Number of steps per sample - default=2000")
@click.option('--step_length', default=0.5, help="Steplength in mm - default=0.5")
@click.option('--fib_thr', default=0.01,
              help="Volume fraction before subsidary fibre orientations are considered - default=0.01")
@click.option('--dist_thr', default=0, help="Discards samples shorter than this threshold (in mm - default=0)")
@click.option('--sampling_radius', default=0,
              help="Sample random points within a sphere with radius x mm from the center of the seed voxels (e.g. --sampvox=0.5, 0.5 mm radius sphere). Default=0")
@click.option('--verbose', default=0, help='If =1 will print the output of all functions.')
def trac(job, **kwargs):
    """ Tractography tools

    \b
    Assumes the following are *always* defined as variables in the shell:
        FREESURFER_HOME -- freesurfer's home folder, usually '/usr/local/freesurfer'
        SUBJECTS_DIR    -- freesurfer's subjects folder, usually '/usr/local/freesurfer/subjects'
        FSLDIR          -- FSL's dir, usually '/usr/local/fsl'


    \b
    JOB can be:
         'probtrack-seedmask'  -- probabilistic tractography using FSL's probtrackx2 and a Nifti seedmask

    """
    jobs = job.split('_')

    if 'probtrack-seedmask' in jobs:
        params_of_interest = ['subject', 'bedpostx_folder', 'seedmask', 'seed2diff_xfm', 'avoidmask', 'terminationmask',
                              'probtrack_dir', 'waypoints', 'waycond', 'loopcheck', 'onewaycondition', 'curvature_thr',
                              'nsamples', 'nsteps', 'step_length', 'fib_thr', 'dist_thr', 'sampling_radius', 'verbose']
        params = []
        [params.append(k + ' : ' + str(kwargs[k])) for k in params_of_interest if kwargs[k] is not None]
        if len(params) > 0:
            print('\n * Running probabilistic tractography with the following parameters:')
            [print('\t' + p) for p in params]
        else:
            print(' ! Bad function call, run "pyepi --help" first.')

        tstart = time.time()
        print(' + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        fsl.probtrack_with_seedmask(subj=kwargs['subject'],
                                    bedpostx_folder=kwargs['bedpostx_folder'],
                                    seedmask=kwargs['seedmask'],
                                    seed2diff_xfm=kwargs['seed2diff_xfm'],
                                    avoidmask=kwargs['avoidmask'],
                                    terminationmask=kwargs['terminationmask'],
                                    output_dir=kwargs['probtrack_dir'],
                                    waypoints=kwargs['waypoints'],
                                    waycond=kwargs['waycond'],
                                    loopcheck=kwargs['loopcheck'],
                                    onewaycondition=kwargs['onewaycondition'],
                                    curvature_thr=kwargs['curvature_thr'],
                                    nsamples=kwargs['nsamples'],
                                    nsteps=kwargs['nsteps'],
                                    step_length=kwargs['step_length'],
                                    fib_thr=kwargs['fib_thr'],
                                    dist_thr=kwargs['dist_thr'],
                                    sampling_radius=kwargs['sampling_radius'],
                                    verbose=kwargs['verbose'])
        print(' + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.')

    return 0


@click.command()
@click.argument('pipe')
@click.argument('subject')
@click.option('--recon/--no-recon', default=True, help='run recon')
# @click.option('--norecon', default=None, help='skip recon')
@click.option('--tracula/--no-tracula', default=True, help='run tracula')
# @click.option('--notracula', default=None, help='skip tracula')
@click.option('--cvs_subj2mni/--no-cvs_subj2mni', default=True, help='run subj to MNI CVS registration')
# @click.option('--nocvs_subj2mni', default=None, help='skip subj to MNI CVS registration')
# @click.option('--cvs_mni2subj', default=None, help='run MNI to subj CVS registration')
# @click.option('--nocvs_mni2subj', default=None, help='skip MNI to subj CVS registration')
# @click.option('--probtrack', default=None, help='run probabilistic tractography with contacts as seeds')
# @click.option('--noprobtrack', default=None, help='skip probabilistic tractography with contacts as seeds')
# @click.option('--tessprobtrack', default=None, help='tesselate probabilistic tractography')
# @click.option('--notessprobtrack', default=None, help="don't tesselate probabilistic tractography")
# @click.option('--morphprobtrack', default=None, help='morph probabilistic tractography to MNI')
# @click.option('--nomorphprobtrack', default=None, help="don't morph probabilistic tractography to MNI")
def pipeline(pipe, subject, **kwargs):
    """ UNIBUC pipelines

    \b
    Assumes the following files are *always* updated with correct paths:
        pyepi.pipelines.preprocess_new_patient.py
        pyepi.pipelines.generate_report.py


    \b
    PIPE can be:
         'newpatient' -- fully preprocess new patient with recon, tracula, tractography, CVS, etc
         'report'     -- generate Jupyter Notebook and HTML report with implantation scheme and SPES connectivity


    """
    tstart = time.time()
    print(' + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    pipelines_dir = os.path.dirname(pipelines.__file__)

    if pipe == 'newpatient':
        params = []
        [params.append(k) for k in kwargs.keys() if kwargs[k]]
        [params.append('no' + k) for k in kwargs.keys() if not kwargs[k]]
        run_list = ['python', os.path.join(pipelines_dir, 'preprocess_new_patient.py'), subject] + params
        p = subprocess.Popen(run_list, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in iter(p.stdout.readline, b''):
            print(line.decode(encoding='utf-8').replace('\r', '').replace('\n', ''))
        p.stdout.close()
        p.wait()

    if pipe == 'report':
        if not os.path.isfile(os.path.join(RAW_DATA_NATIVE, subject, 'SPES.xls')):
            print("WARNING: SPES.xls file is missing from the raw data folder. Connectivity will not show in the report.")
        else:
            overwrite = False
            if os.path.isfile(os.path.join(SUBJECTS_DIR_NATIVE, subject, 'SPES', 'SPES.xls')):
                choice = input(
                    '    -> SPES.xls already exists in the reports folder. Do you want to overwrite it with the '
                    'version from raw data folder? [y/n] (default: y) : ')
                if (choice == '') or (choice.lower() == 'y'):
                    overwrite = True
            if overwrite:
                os.makedirs(os.path.join(SUBJECTS_DIR_NATIVE, subject, 'SPES'), exist_ok=True)
                shutil.copyfile(os.path.join(RAW_DATA_NATIVE, subject, 'SPES.xls'),
                                os.path.join(SUBJECTS_DIR_NATIVE, subject, 'SPES', 'SPES.xls'))
        run_list = ['python', os.path.join(pipelines_dir, 'generate_report.py'), subject]
        p = subprocess.Popen(run_list, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in iter(p.stdout.readline, b''):
            print(line.decode(encoding='utf-8').replace('\r', '').replace('\n', ''))
        p.stdout.close()
        p.wait()

    print(' + Finished in ' + str((time.time() - tstart) / 60) + ' minutes.')

    return 0


if __name__ == "__main__":
    sys.exit(preproc())  # pragma: no cover
