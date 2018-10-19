"""This script will pre-process a new patient, by completing the following steps:

1. Freesurfer's recon
2. Coordinate conversion to Freesurfer's space using .ppr and "Patient Data.xlsx"
3. Freesurfer's Tracula
4. FSL's probabilistic tractography with SEEG contacts as seeds.
5. Tesselation of probabilistic distribution of fibers.

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
    - probtrack / noprobtrack
    - tessprobtrack / notessprobtrack
    - cvs_subj2mni
    - cvs_mni2subj
    - verbose (print output)

ex: python3 preprocess_new_patient SEEG77 verbose notracula

Email notifications may be sent when the script finishes

"""

from pyepi.interfaces import freesurfer, fsl
import time
from pyepi.tools import paths, notifications, volumes, io
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
import glob
import psutil
import tqdm
import deco
import time
import random

if sys.platform == 'win32':
    # Cristi's WSL setup
    RAW_DATA = r'/mnt/d/CloudSynology/rawdata/'
    SUBJECTS_DIR = r'/mnt/d/CloudSynology/subjects/'  # as seen in WSL
    SUBJECTS_DIR_NATIVE = r'd:\\CloudSynology\\subjects\\'  # in native OS
if sys.platform == 'linux':
    # Cristi's Virtual Box setup (fedora64_osboxes)
    RAW_DATA = r'/home/osboxes/host/CloudSynology/rawdata/'
    SUBJECTS_DIR = r'/home/osboxes/subjects/'
    SUBJECTS_DIR_NATIVE = r'/home/osboxes/subjects/'  # in native OS

MAX_RAM_SIZE = psutil.virtual_memory()[0] / 2. ** 30  # in GB
CPU_COUNT = psutil.cpu_count()

# PARAMETERS (using paths in WSL format, ie. /mnt/d/....)
recon = True
tracula = True
cvs_subj2mni = True
cvs_mni2subj = True
save_contact_coordinates = True
probtrack = True
tessprobtrack = True
morphprobtrack = True
send_notification_when_done = True
send_notifications_to_email = 'cristidonos@yahoo.com'

# tracula
openmp = max(1, CPU_COUNT - 2)  # leave two cores free for other processes
doeddy = 1
dorotbvecs = 1
doregbbr = 1
doregmni = 1
doregcvs = 0
nstick = 2
nburnin = 200
niters = 7500
nkeep = 5

# probtrack with seed coords
curvature_thr = 0.2
nsamples = 5000
nsteps = 2000
step_length = 0.5
fib_thr = 0.01
dist_thr = 0.0
sampling_radius = 2

# tesselation
tess_probtrack_threshold = 0.05  # minimum probtrack probability for tesselation
smooth_surface_iterations = 5


# HELP FUNCTIONS FOR PARALLEL PROCESSING
@deco.concurrent(processes=int(min(MAX_RAM_SIZE // 8, CPU_COUNT)))
def par_cvs_apply_morph(subj=None, volume=None, output_dir=None,
                        verbose=None):
    current_file = output_dir + os.path.basename(volume).replace('nii.gz', 'mgz')
    if sys.platform == 'win32':
        keep_going = not os.path.isfile(paths.wsl2win(current_file))
        # bash processes may end up "suspended" on WSL. Check and kill such processes
        for pid in psutil.process_iter():
            if (pid.name() == 'bash') and (pid.status() == 'stopped'):
                pid.kill()
    else:
        keep_going = not os.path.isfile(current_file)
    if keep_going:
        time.sleep(1 + random.random() * 5)
        freesurfer.cvs_apply_morph(subj=subj, subjects_dir=SUBJECTS_DIR,
                                   volume=volume,
                                   output_volume=os.path.basename(volume).replace('nii.gz', 'mgz'),
                                   output_dir=output_dir,
                                   interpolation='linear',
                                   verbose=verbose)
        time.sleep(1 + random.random() * 5)


@deco.synchronized
def run_cvs_apply_morph(probtrac_files, probtrac_cvs_dir):
    for volume in probtrac_files:
        par_cvs_apply_morph(subj=subj,
                            volume=volume,
                            output_dir=probtrac_cvs_dir,
                            verbose=verbose)


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
    if 'nocvs_subj2mni' in sys.argv:
        cvs_subj2mni = False
    if 'nocvs_mni2subj' in sys.argv:
        cvs_mni2subj = False
    if 'probtrack' in sys.argv:
        probtrack = True
    if 'noprobtrack' in sys.argv:
        probtrack = False
    if 'tessprobtrack' in sys.argv:
        tessprobtrack = True
    if 'notessprobtrack' in sys.argv:
        tessprobtrack = False

    print('\nNumber of CPUs: ' + str(CPU_COUNT) + '.')
    print('RAM: ' + str(MAX_RAM_SIZE) + ' Gb.\n')

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
            save_contact_coordinates = False
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

    # save contact coordinates in Freesurfer's space
    if save_contact_coordinates:
        tstart = time.time()
        log = "\n* Processing contacts coordinates."
        print(log)
        email_body.append(log)
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

        mri_norm = nib.load(SUBJECTS_DIR_NATIVE + subj + os.sep + 'mri' + os.sep + 'norm.mgz')
        vox_fscoords = np.dot(np.linalg.inv(mri_norm.get_header().get_vox2ras_tkr()), fscoords.T).T

        all_coords = pd.concat([coords,
                                pd.DataFrame(fscoords, columns=['xmri', 'ymri', 'zmri', 'dummymri']),
                                pd.DataFrame(vox_fscoords, columns=['xmrivox', 'ymrivox', 'zmrivox', 'dummymrivox']),
                                ], axis=1)
        all_coords = all_coords.drop(columns=all_coords.columns[['dummy' in c for c in all_coords.columns]])
        all_coords = volumes.identify_voxel_location(all_coords,
                                                     SUBJECTS_DIR_NATIVE + subj + os.sep + 'mri' + os.sep + 'aparc+aseg.mgz',
                                                     os.path.dirname(
                                                         freesurfer.__file__) + os.sep + 'FreesurferLUT.xlsx')
        all_coords.to_excel(SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')

        average_struct_coords = volumes.average_structure_coordinates(
            SUBJECTS_DIR_NATIVE + subj + os.sep + 'mri' + os.sep + 'aparc+aseg.mgz', os.path.dirname(
                freesurfer.__file__) + os.sep + 'FreesurferLUT.xlsx')
        average_struct_coords.to_excel(SUBJECTS_DIR_NATIVE + subj + os.sep + 'Average_structure_coordinates.xlsx')
        log = '    + Finished in ' + str((time.time() - tstart) / 60) + ' minutes.'
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
        print('           ', end=' ')
        email_body.append(log)
        all_coords = pd.read_excel(SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')
        r = subprocess.run(
            ['bash', '-i', '-c', 'mri_info --ras2vox-tkr ' + SUBJECTS_DIR + subj + '/dmri/brain_anat.nii.gz'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        ras2vox = r.stdout.decode('utf-8').split('\n')[-5:-1]
        ras2vox = np.array([line.split() for line in ras2vox], dtype=np.float)
        contacts = [c for c in all_coords['name'].values]
        probtrac_files = []
        for c in tqdm.tqdm(contacts, ncols=60):
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
                                          curvature_thr=curvature_thr,
                                          nsamples=nsamples,
                                          nsteps=nsteps,
                                          step_length=step_length,
                                          fib_thr=fib_thr,
                                          dist_thr=dist_thr,
                                          sampling_radius=sampling_radius)
            paths.silentremove(native_file)
            # get the probabilistic tractography filename
            input_volume = glob.glob(
                SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts' + os.sep + c.replace("'",
                                                                                               "+") + '*.nii.gz')
            if len(input_volume) != 1:
                print('ERROR: are there multiple probtrack files with the same contact name?!')
                print('EXECUTION STOPPED.')
                sys.exit()
            else:
                input_volume = input_volume[0]
                probtrac_files.append(input_volume)
            if tessprobtrack:
                output_surface = input_volume.replace('.nii.gz', '.surf')
                freesurfer.tesselate(input_volume, tess_probtrack_threshold, output_volume=None, normalize=True,
                                     normalize_by=nsamples, output_surface=output_surface,
                                     smooth_surface_iterations=smooth_surface_iterations)
        log = '    + Finished in ' + str((time.time() - tstart) / 60) + ' minutes.'
        print(log)
        email_body.append(log)

    if morphprobtrack:
        log = '\n* Morphing probabilistic tractography to CVS template.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)

        email_body.append(log)
        probtrac_cvs_dir_native = SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts_cvs_avg35' + os.sep
        probtrac_cvs_dir = SUBJECTS_DIR + subj + '/probtrac_contacts_cvs_avg35/'
        os.makedirs(probtrac_cvs_dir_native, exist_ok=True)
        #        Get file names from folder
        probtrac_files = [f for f in glob.glob(
            SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts' + os.sep + '*.nii.gz')]

        if sys.platform == 'win32':
            probtrac_files = [paths.win2wsl(pf) for pf in probtrac_files]

        # don't know why, but sometimes bash process appears suspended in Windows when running multiple instances
        # rerun cvs_apply_morph until all files have been processed
        need_to_rerun = True
        while need_to_rerun:
            probtrac_files_in_cvs = [f for f in glob.glob(
                SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts_cvs_avg35' + os.sep + '*.mgz')]
            files_left = [x for x in [os.path.basename(p) for p in [pf.replace('.nii.gz', '') for pf in probtrac_files]]
                          if
                          x not in [os.path.basename(pc) for pc in
                                    [pfc.replace('.mgz', '') for pfc in probtrac_files_in_cvs]]]
            new_probtrac_files = []
            for f in files_left:
                new_probtrac_files.extend([p for p in probtrac_files if f in p])
            probtrac_files = new_probtrac_files
            if len(probtrac_files) > 0:
                run_cvs_apply_morph(probtrac_files, probtrac_cvs_dir)
            else:
                need_to_rerun = False

        # print('\n           ', end=' ')
        # for pf in tqdm.tqdm(probtrac_files, ncols=60):
        #     if sys.platform == 'win32':
        #         volume = paths.win2wsl(pf)
        #     else:
        #         volume = pf
        #     freesurfer.cvs_apply_morph(subj=subj, subjects_dir=SUBJECTS_DIR,
        #                                volume=volume,
        #                                output_volume=os.path.basename(volume).replace('nii.gz', 'mgz'),
        #                                output_dir=probtrac_cvs_dir,
        #                                interpolation='linear',
        #                                verbose=verbose)

        log = '    + Finished in ' + str((time.time() - tstart) / 60) + ' minutes.'
        print(log)
        email_body.append(log)

    if send_notification_when_done:
        try:
            notifications.sendmail(to_address=send_notifications_to_email, subject='New patient job done.',
                                   message='\n'.join(email_body))
        except:
            pass
