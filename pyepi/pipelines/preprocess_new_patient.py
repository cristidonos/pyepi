"""This script will pre-process a new patient, by completing the following steps:

1. Freesurfer's recon
2. Coordinate conversion to Freesurfer's space using .ppr and "Patient Data.xlsx"
3. Freesurfer's Tracula
4  FSL's probabilistic tractography with SEEG contacts as seeds.


The RAW_DATA folder needs to be defined and have the following structure:

RAW_DATA
├── subject (i.e. SEEG72)
│   ├── DTI (diffusion dicoms)
│   ├── DWI (this one gets created during preprocessing, contains diffusion dicoms converted to Nifti and bval/bvec files)
│   ├── T1  (T1 dicoms)
│   └── T2  (T2 dicoms, optional)

The output will be generated in Freesurfer's $SUBJECTS_DIR folder.

To be called with 2 positional arguments:
    - SUBJECT_ID
    - VERBOSE (optional, if second argument exist verbose=True)
ex: python3 preprocess_new_patient SEEG77 verbose

Email notifications may be sent when the script finishes

"""

from pyepi.interfaces import freesurfer, fsl
import time
from pyepi.tools import paths, notifications
import os
import sys

if sys.platform == 'win32':
    # Cristi's WSL setup
    RAW_DATA = '/mnt/d/CloudSynology/rawdata/'
    SUBJECTS_DIR = '/mnt/d/CloudSynology/subjects/'
if sys.platform == 'linux':
    # Cristi's Virtual Box setup (fedora64_osboxes)
    RAW_DATA = '/home/osboxes/host/CloudSynology/rawdata/'
    SUBJECTS_DIR = '/home/osboxes/subjects/'

# PARAMETERS (using paths in WSL format, ie. /mnt/d/....)
cvs_subj2mni = False
cvs_mni2subj = False
send_notification_when_done = True
send_notifications_to_email = 'cristidonos@yahoo.com'

openmp = 4
doeddy = 1
dorotbvecs = 1
doregbbr = 1
doregmni = 1
doregcvs = 0
nstick = 2
nburnin = 200
niters = 7500
nkeep = 5

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Subject ID needs to be specified as the first argument:\n    preprocess_new_patient TEST_SUBJECT01 ')
        sys.exit()

    subj = sys.argv[1]
    verbose = False
    if len(sys.argv) > 2:
        verbose = True

    try:
        t1dir = RAW_DATA + os.sep + subj + os.sep + 'T1' + os.sep
        if sys.platform == 'win32':
            t1dir = paths.wsl2win(t1dir)
        t1dcm = os.listdir(t1dir)[0]
        t1file = t1dir + t1dcm
        if not os.path.isfile(t1file):
            raise OSError
        else:
            if sys.platform == 'win32':
                t1file = paths.win2wsl(t1file)
    except OSError:
        print("ERROR: Subject's T1 folder is empty or does not exist.")
        print('EXECUTION STOPPED.')
        sys.exit()

    try:
        t2dir = RAW_DATA + os.sep + subj + os.sep + 'T2' + os.sep
        if sys.platform == 'win32':
            t2dir = paths.wsl2win(t2dir)
        t2dcm = os.listdir(t2dir)[0]
        t2file = t2dir + t2dcm
        if not os.path.isfile(t2file):
            raise OSError
        else:
            if sys.platform == 'win32':
                t2file = paths.win2wsl(t2file)
    except OSError:
        print("WARNING: Subject's T1 folder is empty or does not exist.")
        choice = input('    -> Do you want to proceed without a T2 scan? [y/n] (default: y) : ')
        if (choice == '') or (choice.lower() == 'y'):
            t2file = None
        else:
            print('EXECUTION STOPPED.')
            sys.exit()

    try:
        dtidir = RAW_DATA + os.sep + subj + os.sep + 'DTI' + os.sep
        if sys.platform == 'win32':
            dtidir = paths.wsl2win(dtidir)
        dtidcm = os.listdir(dtidir)[0]
        dtifile = dtidir + dtidcm
        if not os.path.isfile(dtifile):
            raise OSError
        else:
            if sys.platform == 'win32':
                dtifile = paths.win2wsl(dtifile)
    except OSError:
        print("WARNING: Subject's DTI folder is empty or does not exist.")
        choice = input('    -> Do you want to proceed without a DTI scan? [y/n] (default: y) : ')
        if (choice == '') or (choice.lower() == 'y'):
            dtifile = None
        else:
            print('EXECUTION STOPPED.')
            sys.exit()

    email_body = [subj]

    # RECON
    tstart = time.time()
    log = "\n* Running Freesurfer's recon-all."
    print(log)
    email_body.append(log)
    log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(log)
    email_body.append(log)
    freesurfer.recon(subj=subj, t1_file=t1file, t2_file=t2file, openmp=openmp, verbose=verbose)
    log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
    print(log)
    email_body.append(log)

    # CVS
    if cvs_subj2mni:
        tstart = time.time()
        log = "\n* Running subject to template CVS registration."
        print(log)
        email_body.append(log)
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        freesurfer.cvs_subj2mni(subj=subj, openmp=openmp, verbose=verbose)
        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)

    if cvs_mni2subj:
        tstart = time.time()
        log = "\n* Running template to subject CVS registration."
        print(log)
        email_body.append(log)
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        freesurfer.cvs_mni2subj(subj=subj, openmp=openmp, subjects_dir=SUBJECTS_DIR, verbose=verbose)
        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)

    # DTI
    if dtifile is not None:
        log = '\n* Converting DTI dicoms to Nifti.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        dwidir = RAW_DATA + '//' + subj + '//DWI'
        freesurfer.dcm2niix(dcm_file=dtifile,
                            output_filename='dwi',
                            output_folder=dwidir)
        log = '    + Finished in ' + str((time.time() - tstart)) + ' seconds.'
        print(log)
        email_body.append(log)

        # tracula config file first
        cfg = freesurfer.tracula_config(subj=subj, dicom=dwidir + '//dwi.nii', config_folder=None,
                                        doeddy=doeddy, dorotbvecs=dorotbvecs, doregbbr=doregbbr, doregmni=doregmni,
                                        doregcvs=doregcvs, nstick=nstick, nburnin=nburnin, nsample=niters, nkeep=nkeep,
                                        subjects_dir=SUBJECTS_DIR)
        # if running on Windows Linux Subsystem the path to cfg needs to be changed to point to /mnt/c/Users/...
        if sys.platform == 'win32':
            breakdown = cfg.replace(':', '').split(sep='\\')
            breakdown[0] = breakdown[0].lower()
            breakdown = ['mnt'] + breakdown
            cfg_linux = '/' + '/'.join(breakdown)
        else:
            cfg_linux = cfg

        # run tracula
        log = '\n* Running Tracula.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting Tracula at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        freesurfer.tracula_run(subj=subj, prep=True, bedp=True, path=True, cfg_file=cfg_linux, verbose=verbose)

        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)

        paths.silentremove(cfg)

        if send_notification_when_done:
            try:
                notifications.sendmail(to_address=send_notifications_to_email, subject='New patient job done.',
                                       message='\n'.join(email_body))
            except:
                pass
