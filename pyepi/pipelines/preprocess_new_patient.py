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
│   ├── T2  (T2 dicoms, optional)
│   └── Patient Data.xlsx  (format detailed in io.load_contacts , optional)
│   └── subject.ppr  (ppr file containing the final trajectories, optional)

The output will be generated in Freesurfer's $SUBJECTS_DIR folder.

To be called with 1 positional arguments:
    - SUBJECT_ID
And optional additional arguments to enable/disable various preprocessing steps:
    - recon / norecon (optional, if second argument exist verbose=True)
    - tracula / notracula
    - cvs_subj2mni
    - cvs_mni2subj
    - verbose (print output)

ex: python3 preprocess_new_patient SEEG77 verbose notracula

Email notifications may be sent when the script finishes

"""

from pyepi.interfaces import freesurfer, fsl
import time
from pyepi.tools import paths, notifications
from pyepi.tools import io
import os
import sys
import numpy as np
import pandas as pd
import subprocess

if sys.platform == 'win32':
    # Cristi's WSL setup
    RAW_DATA = r'/mnt/d/CloudSynology/rawdata/'
    SUBJECTS_DIR = r'/mnt/d/CloudSynology/subjects/'  # as seen in WSL
    SUBJECTS_DIR_NATIVE = r'D:\\CloudSynology\\subjects\\'  # in native OS
if sys.platform == 'linux':
    # Cristi's Virtual Box setup (fedora64_osboxes)
    RAW_DATA = r'/home/osboxes/host/CloudSynology/rawdata/'
    SUBJECTS_DIR = r'/home/osboxes/subjects/'
    SUBJECTS_DIR_NATIVE = r'/home/osboxes/subjects/'  # in native OS

# PARAMETERS (using paths in WSL format, ie. /mnt/d/....)
recon = True
tracula = True
cvs_subj2mni = False
cvs_mni2subj = False
probtrack = True
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

    # override default values for quick tests
    if 'verbose' in sys.argv:
        verbose = True
    if 'norecon' in sys.argv:
        recon = False
    if 'recon' in sys.argv:
        recon = True
    if 'tracula' in sys.argv:
        tracula = True
    if 'notracula' in sys.argv:
        tracula = False
    if 'cvs_subj2mni' in sys.argv:
        cvs_subj2mni = True
    if 'cvs_mni2subj' in sys.argv:
        cvs_mni2subj = True
    if 'probtrack' in sys.argv:
        probtrack = True
    if 'noprobtrack' in sys.argv:
        probtrack = False

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
            dtifile = False
        else:
            print('EXECUTION STOPPED.')
            sys.exit()

    try:
        pprfile = RAW_DATA + os.sep + subj + os.sep + subj + '.ppr'
        patientdatafile = RAW_DATA + os.sep + subj + os.sep + 'Patient Data.xlsx'
        if sys.platform == 'win32':
            pprfile = paths.wsl2win(pprfile)
            patientdatafile = paths.wsl2win(patientdatafile)
        if (not os.path.isfile(pprfile)) or (not os.path.isfile(patientdatafile)):
            raise OSError
    except OSError:
        print("WARNING: Subject's .ppr and/or Patient Data.xlsx are missing.")
        choice = input(
            '    -> Do you want to proceed without probabilistic tractography with contact seeds? [y/n] (default: y) : ')
        if (choice == '') or (choice.lower() == 'y'):
            patientdatafile = None
            pprfile = None
            probtrack = False
        else:
            print('EXECUTION STOPPED.')
            sys.exit()

    email_body = [subj]

    # RECON
    if recon:
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
    if (dtifile is not None) and tracula:
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
    else:
        # no dtifile --> no tracula
        tracula = False

    if tracula:
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

    if probtrack:
        log = '\n* Running probabilistic tractography with contacts as seeds.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)

        ppr_scans, ppr_anat, ppr_trajectories = io.read_ppr(pprfile)
        coords, landmarks, mri_uid = io.load_contacts(patientdatafile)
        coords['dummy'] = np.ones_like(coords.loc[:, 'x'])
        landmarks['dummy'] = np.ones_like(landmarks.loc[:, 'x'])
        # mri index in ppr
        scan = [k for k in ppr_scans.values() if k['uid'] == mri_uid][0]
        xfm_ppr = scan['xfm']
        fscoords = np.array(coords.loc[:, ['x', 'y', 'z', 'dummy']]).dot(scan['wt'].T).dot(np.diag([-1, -1, 1, 1]))
        all_coords = pd.concat([coords,
                                pd.DataFrame(fscoords, columns=['xmri', 'ymri', 'zmri', 'dummymri']),
                                ], axis=1)
        all_coords = all_coords.drop(columns=all_coords.columns[['dummy' in c for c in all_coords.columns]])
        all_coords.to_excel(SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')
        r = subprocess.run(
            ['bash', '-i', '-c', 'mri_info --ras2vox-tkr ' + SUBJECTS_DIR + subj + '/dmri/brain_anat.nii.gz'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        ras2vox = r.stdout.decode('utf-8').split('\n')[-5:-1]
        ras2vox = np.array([line.split() for line in ras2vox], dtype=np.float)
        contacts = [c for c in all_coords['name'].values]
        for c in contacts:
            native_file, wsl_file = paths.wsl_tempfile('seedmask.txt')
            coords = all_coords[all_coords['name'] == c][['xmri', 'ymri', 'zmri']].values
            coords_str = ' '.join([str(s) for s in ras2vox.dot(np.append(coords, 1))[0:3]])
            # save seedfile for FSL's probtrackx2
            with open(native_file, 'w', newline='\n') as f:
                f.write(coords_str)

            fsl.probtrack_with_seedcoords(subj=subj,
                                          bedpostx_folder=SUBJECTS_DIR + subj + '/dmri.bedpostX/',
                                          seedcoords=wsl_file,
                                          seed2diff_xfm=SUBJECTS_DIR + subj + '/dmri/xfms/anatorig2diff.bbr.mat',
                                          seedref=SUBJECTS_DIR + subj + '/dmri/brain_anat_orig.nii.gz',
                                          avoidmask=None,
                                          terminationmask=None,
                                          output_dir=SUBJECTS_DIR + subj + '/probtrac_contacts/',
                                          output_file=c.replace("'", "+"),
                                          waypoints=None,
                                          waycond='AND',
                                          loopcheck=True,
                                          onewaycondition=True,
                                          curvature_thr=0.2,
                                          nsamples=5000,
                                          nsteps=2000,
                                          step_length=0.5,
                                          fib_thr=0.01,
                                          dist_thr=0.0,
                                          sampling_radius=2)
            paths.silentremove(native_file)
        log = '    + Finished in ' + str((time.time() - tstart) / 60) + ' minutes.'
        print(log)
        email_body.append(log)

    if send_notification_when_done:
        try:
            notifications.sendmail(to_address=send_notifications_to_email, subject='New patient job done.',
                                   message='\n'.join(email_body))
        except:
            pass
