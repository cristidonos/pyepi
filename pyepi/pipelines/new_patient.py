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

import glob
import os
import random
import subprocess
import sys
import time

import deco
import nibabel as nib
import numpy as np
import pandas as pd
import psutil

from pyepi.interfaces import freesurfer, fsl
from pyepi.tools import paths, notifications, volumes, inout


class newpatient:
    RAW_DATA, RAW_DATA_NATIVE, SUBJECTS_DIR, SUBJECTS_DIR_NATIVE = paths.set_paths(hostname=paths.HOSTNAME)

    MAX_RAM_SIZE = psutil.virtual_memory()[0] / 2. ** 30  # in GB
    CPU_COUNT = psutil.cpu_count()

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
    # @deco.concurrent(processes=int(min(MAX_RAM_SIZE // 8, CPU_COUNT)))
    def par_cvs_apply_morph(self, subj=None, volume=None, output_dir=None,
                            verbose=None):
        current_file = output_dir + os.path.basename(volume).replace('nii.gz', 'mgz')
        if sys.platform == 'win32':
            keep_going = not os.path.isfile(paths.wsl2win(current_file))
            # bash processes may end up "suspended" on WSL. Check and kill such processes
            for pid in psutil.process_iter():
                if (pid.name() == 'bash') and (pid.status() == 'stopped'):
                    print('killing pid' + pid.name())
                    pid.kill()
        else:
            keep_going = not os.path.isfile(current_file)
        if keep_going:
            time.sleep(1 + random.random() * 5)
            freesurfer.cvs_apply_morph(subj=subj, subjects_dir=self.SUBJECTS_DIR,
                                       volume=volume,
                                       output_volume=os.path.basename(volume).replace('nii.gz', 'mgz'),
                                       output_dir=output_dir,
                                       interpolation='linear',
                                       verbose=verbose)
            time.sleep(1 + random.random() * 5)

    # @deco.synchronized
    def run_cvs_apply_morph(self, probtrac_files, probtrac_cvs_dir):
        for volume in probtrac_files:
            self.par_cvs_apply_morph(subj=self.subj,
                                volume=volume,
                                output_dir=probtrac_cvs_dir,
                                verbose=self.verbose)

    def run_data_consistency_checks(self, subj):
        """ Checks if all files are in the default folder structure and which jobs can be run given the availabla data

        Parameters
        ----------
        subj: string
            Subject ID
        Returns
        -------
        data_dir: dictionary
            Dictionary with paths to files of interest
        allowed_jobs: dictionary
            Dictionary of jobs that can run given the available data
        """
        allowed_jobs = {}
        data_dir = {}
        allowed_jobs['verbose'] = True
        allowed_jobs['send_notification_when_done'] = True
        allowed_jobs['save_contact_coordinates'] = True
        try:
            t1dir = self.RAW_DATA + os.sep + subj + os.sep + 'T1' + os.sep
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
        data_dir['t1file'] = t1file
        allowed_jobs['recon'] = True
        allowed_jobs['cvs_subj2mni'] = True
        allowed_jobs['cvs_mni2subj'] = True

        try:
            t2dir = self.RAW_DATA + os.sep + subj + os.sep + 'T2' + os.sep
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
        data_dir['t2file'] = t2file

        try:
            dtidir = self.RAW_DATA + os.sep + subj + os.sep + 'DTI' + os.sep
            if sys.platform == 'win32':
                dtidir = paths.wsl2win(dtidir)
            dtidcm = os.listdir(dtidir)[0]
            dtifile = dtidir + dtidcm
            if not os.path.isfile(dtifile):
                raise OSError
            else:
                if sys.platform == 'win32':
                    dtifile = paths.win2wsl(dtifile)
                allowed_jobs['tracula'] = True
        except OSError:
            print("WARNING: Subject's DTI folder is empty or does not exist.")
            choice = input('    -> Do you want to proceed without a DTI scan? [y/n] (default: y) : ')
            if (choice == '') or (choice.lower() == 'y'):
                dtifile = None
                allowed_jobs['tracula'] = False
            else:
                print('EXECUTION STOPPED.')
                sys.exit()
        data_dir['dtifile'] = dtifile

        try:
            pprfile = self.RAW_DATA + os.sep + subj + os.sep + subj + '.ppr'
            patientdatafile = self.RAW_DATA + os.sep + subj + os.sep + 'Patient Data.xlsx'
            if sys.platform == 'win32':
                pprfile = paths.wsl2win(pprfile)
                patientdatafile = paths.wsl2win(patientdatafile)
            allowed_jobs['save_contact_coordinates'] = True
            allowed_jobs['probtrack'] = True
            allowed_jobs['morphcontacts'] = True
            allowed_jobs['tessprobtrack'] = True
            allowed_jobs['morphprobtrack'] = True
            if (not os.path.isfile(pprfile)) or (not os.path.isfile(patientdatafile)):
                raise OSError
        except OSError:
            print("WARNING: Subject's .ppr and/or Patient Data.xlsx are missing.")
            choice = input(
                '    -> Do you want to proceed without probabilistic tractography with contact seeds? [y/n] (default: y) : ')
            if (choice == '') or (choice.lower() == 'y'):
                patientdatafile = None
                pprfile = None
                allowed_jobs['save_contact_coordinates'] = False
                allowed_jobs['probtrack'] = False
                allowed_jobs['morphcontacts'] = False
                allowed_jobs['tessprobtrack'] = False
                allowed_jobs['morphprobtrack'] = False
            else:
                print('EXECUTION STOPPED.')
                sys.exit()
        data_dir['pprfile'] = pprfile
        data_dir['patientdatafile'] = patientdatafile
        return data_dir, allowed_jobs

    def setup_jobs(self, args, allowed_jobs, jobs):
        for k in jobs.keys():
            if 'no' + k in args:
                jobs[k] = False
            else:
                jobs[k] = jobs[k] & allowed_jobs[k]

        return jobs

    def run_recon(self, subj, data_dir, jobs, email_body):
        tstart = time.time()
        log = "\n* Running Freesurfer's recon-all."
        print(log)
        email_body.append(log)
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        freesurfer.recon(subj=subj, t1_file=data_dir['t1file'], t2_file=data_dir['t2file'], openmp=self.openmp,
                         verbose=jobs['verbose'])
        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        return email_body

    def run_save_contact_coordinates(self, subj, data_dir, email_body):
        tstart = time.time()
        log = "\n* Processing contacts coordinates."
        print(log)
        email_body.append(log)
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        ppr_scans, ppr_anat, ppr_trajectories = inout.read_ppr(data_dir['pprfile'])
        coords, landmarks, mri_uid = inout.load_contacts(data_dir['patientdatafile'])
        coords['dummy'] = np.ones_like(coords.loc[:, 'x'])
        landmarks['dummy'] = np.ones_like(landmarks.loc[:, 'x'])
        # mri index in ppr
        scan = [k for k in ppr_scans.values() if k['uid'] == mri_uid][0]
        xfm_ppr = scan['xfm']
        fscoords = np.array(coords.loc[:, ['x', 'y', 'z', 'dummy']]).dot(scan['wt'].T).dot(np.diag([-1, -1, 1, 1]))

        mri_norm = nib.load(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'mri' + os.sep + 'norm.mgz')
        vox_fscoords = np.dot(np.linalg.inv(mri_norm.get_header().get_vox2ras_tkr()), fscoords.T).T

        all_coords = pd.concat([coords,
                                pd.DataFrame(fscoords, columns=['xmri', 'ymri', 'zmri', 'dummymri']),
                                pd.DataFrame(vox_fscoords, columns=['xmrivox', 'ymrivox', 'zmrivox', 'dummymrivox']),
                                ], axis=1)
        all_coords = all_coords.drop(columns=all_coords.columns[['dummy' in c for c in all_coords.columns]])
        all_coords = volumes.identify_voxel_location(all_coords,
                                                     self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'mri' + os.sep + 'aparc+aseg.mgz',
                                                     os.path.dirname(
                                                         freesurfer.__file__) + os.sep + 'FreesurferLUT.xlsx')
        all_coords.to_excel(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')

        average_struct_coords = volumes.average_structure_coordinates(
            self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'mri' + os.sep + 'aparc+aseg.mgz', os.path.dirname(
                freesurfer.__file__) + os.sep + 'FreesurferLUT.xlsx')
        average_struct_coords.to_excel(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'Average_structure_coordinates.xlsx')
        log = '    + Finished in ' + str((time.time() - tstart) / 60) + ' minutes.'
        print(log)
        return email_body

    def run_cvs_subj2mni(self, subj, jobs, email_body):
        tstart = time.time()
        log = "\n* Running subject to template CVS registration."
        print(log)
        email_body.append(log)
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        freesurfer.cvs_subj2mni(subj=subj, openmp=self.openmp, verbose=jobs['verbose'])
        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)
        return email_body

    def run_cvs_mni2subj(self, subj, jobs, email_body):
        tstart = time.time()
        log = "\n* Running template to subject CVS registration."
        print(log)
        email_body.append(log)
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        freesurfer.cvs_mni2subj(subj=subj, openmp=self.openmp, subjects_dir=self.SUBJECTS_DIR, verbose=jobs['verbose'])
        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)
        return email_body

    def run_morphcontacts(self, subj, email_body):
        log = '\n* Morphing contacts to CVS template.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)

        contacts_dir_native = self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'contacts' + os.sep
        contacts_dir = self.SUBJECTS_DIR + subj + '/contacts/'
        contacts_cvs_dir_native = self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'contacts_cvs_avg35' + os.sep
        contacts_cvs_dir = self.SUBJECTS_DIR + subj + '/contacts_cvs_avg35/'
        os.makedirs(contacts_dir_native, exist_ok=True)
        os.makedirs(contacts_cvs_dir_native, exist_ok=True)

        all_coords = pd.read_excel(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')
        contacts_files = [contacts_dir_native + ac.replace("'", '+') + '.mgz' for ac in all_coords.name.values]

        for i in np.arange(0, all_coords.shape[0]):
            filename = all_coords[['name']].loc[i].values[0].replace("'", "+")
            volumes.contact_to_volume(
                contact_coords=np.int16(np.round(all_coords[['xmrivox', 'ymrivox', 'zmrivox']].loc[i].values)),
                reference_volume=os.path.join(self.SUBJECTS_DIR_NATIVE, subj, 'mri', 'norm.mgz'),
                contact_volume=os.path.join(contacts_dir_native, filename + '.mgz'),
            )

        if sys.platform == 'win32':
            contacts_files = [paths.win2wsl(pf) for pf in contacts_files]

        # don't know why, but sometimes bash process appears suspended in Windows when running multiple instances
        # rerun cvs_apply_morph until all files have been processed
        need_to_rerun = True
        while need_to_rerun:
            contacts_files_in_cvs = [f for f in glob.glob(
                self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'contacts_cvs_avg35' + os.sep + '*.mgz')]
            files_left = [x for x in [os.path.basename(p) for p in [pf.replace('.mgz', '') for pf in contacts_files]]
                          if
                          x not in [os.path.basename(pc) for pc in
                                    [pfc.replace('.mgz', '') for pfc in contacts_files_in_cvs]]]
            new_contacts_files = []
            for f in files_left:
                new_contacts_files.extend([p for p in contacts_files if f in p])
            contacts_files = new_contacts_files
            if len(contacts_files) > 0:
                self.run_cvs_apply_morph(contacts_files, contacts_cvs_dir)
            else:
                need_to_rerun = False

                # update contact coordinates file with mni coordinates and Yeo atlas
                all_coords = pd.read_excel(os.path.join(self.SUBJECTS_DIR_NATIVE, subj, 'Contact_coordinates.xlsx'))

        mni_cols = ['name', 'mni_xmri', 'mni_ymri', 'mni_zmri', 'mni_xmrivox', 'mni_ymrivox', 'mni_zmrivox',
                    'mni_nvoxels',
                    'mni_mean_intensity', 'mni_std_intensity']
        mni_coords = pd.DataFrame(columns=mni_cols)

        for c in all_coords.name.values:
            ix = np.where(all_coords.name == c)[0]
            mri_coords, mri_vox, mri_stats = volumes.contact_volume_to_mni_coordinates(
                contact_volume=contacts_cvs_dir_native + c.replace("'", '+') + '.mgz')
            mni_xmri, mni_ymri, mni_zmri = mri_coords
            mni_xmrivox, mni_ymrivox, mni_zmrivox = mri_vox
            mni_coords = pd.concat([mni_coords,
                                    pd.DataFrame(
                                        [[c, mni_xmri, mni_ymri, mni_zmri, mni_xmrivox, mni_ymrivox, mni_zmrivox,
                                          mri_stats['nvoxels'], mri_stats['mean_voxel_intensity'],
                                          mri_stats['std_voxel_intensity']]], columns=mni_cols)
                                    ], sort=False).reset_index(drop=True)

        all_coords = all_coords.merge(mni_coords.reset_index(drop=True), sort=False)
        all_coords = volumes.identify_voxel_location(all_coords,
                                                     self.SUBJECTS_DIR_NATIVE + 'Yeo_JNeurophysiol11_MNI152' + os.sep + 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz',
                                                     os.path.dirname(freesurfer.__file__) + os.sep + 'YeoLUT.xlsx',
                                                     name_prefix='mni')
        all_coords.to_excel(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')

        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)
        return email_body

    def run_convert_dti_dicoms(self, subj, data_dir, email_body):
        log = '\n* Converting DTI dicoms to Nifti.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        data_dir['dwidir'] = self.RAW_DATA + '//' + subj + '//DWI'
        freesurfer.dcm2niix(dcm_file=data_dir['dtifile'],
                            output_filename='dwi',
                            output_folder=data_dir['dwidir'])
        log = '    + Finished in ' + str((time.time() - tstart)) + ' seconds.'
        print(log)
        email_body.append(log)
        return email_body

    def run_tracula(self, subj, data_dir, email_body):
        # tracula config file first
        cfg = freesurfer.tracula_config(subj=subj, dicom=data_dir['dwidir'] + '//dwi.nii', config_folder=None,
                                        doeddy=self.doeddy, dorotbvecs=self.dorotbvecs, doregbbr=self.doregbbr, doregmni=self.doregmni,
                                        doregcvs=self.doregcvs, nstick=self.nstick, nburnin=self.nburnin, nsample=self.niters, nkeep=self.nkeep,
                                        subjects_dir=self.SUBJECTS_DIR)
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
        freesurfer.tracula_run(subj=subj, prep=True, bedp=True, path=True, cfg_file=cfg_linux, verbose=self.verbose)

        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)
        paths.silentremove(cfg)
        return email_body

    def run_probtrack(self, subj, jobs, email_body):
        log = '\n* Running probabilistic tractography with contacts as seeds.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)
        email_body.append(log)
        all_coords = pd.read_excel(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'Contact_coordinates.xlsx')
        r = subprocess.run(
            ['bash', '-i', '-c', 'mri_info --ras2vox-tkr ' + self.SUBJECTS_DIR + subj + '/dmri/brain_anat.nii.gz'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        ras2vox = r.stdout.decode('utf-8').split('\n')[-5:-1]
        ras2vox = np.array([line.split() for line in ras2vox], dtype=np.float)
        contacts = [c for c in all_coords['name'].values]
        probtrac_files = []
        for c in contacts:
            native_file, wsl_file = paths.wsl_tempfile('seedmask.txt')
            coords = all_coords[all_coords['name'] == c][['xmri', 'ymri', 'zmri']].values
            coords_str = ' '.join([str(s) for s in ras2vox.dot(np.append(coords, 1))[0:3]])
            # save seedfile for FSL's probtrackx2
            with open(native_file, 'w', newline='\n') as f:
                f.write(coords_str)

            fsl.probtrack_with_seedcoords(subj=subj,
                                          bedpostx_folder=self.SUBJECTS_DIR + subj + '/dmri.bedpostX/',
                                          seedcoords=wsl_file,
                                          seed2diff_xfm=self.SUBJECTS_DIR + subj + '/dmri/xfms/anatorig2diff.bbr.mat',
                                          seedref=self.SUBJECTS_DIR + subj + '/dmri/brain_anat_orig.nii.gz',
                                          avoidmask=None,
                                          terminationmask=None,
                                          output_dir=self.SUBJECTS_DIR + subj + '/probtrac_contacts/',
                                          output_file=c.replace("'", "+"),
                                          waypoints=None,
                                          waycond='AND',
                                          loopcheck=True,
                                          onewaycondition=True,
                                          curvature_thr=self.curvature_thr,
                                          nsamples=self.nsamples,
                                          nsteps=self.nsteps,
                                          step_length=self.step_length,
                                          fib_thr=self.fib_thr,
                                          dist_thr=self.dist_thr,
                                          sampling_radius=self.sampling_radius)
            paths.silentremove(native_file)
            # get the probabilistic tractography filename
            input_volume = glob.glob(
                self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts' + os.sep + c.replace("'",
                                                                                               "+") + '*.nii.gz')
            if len(input_volume) != 1:
                print('ERROR: are there multiple probtrack files with the same contact name?!')
                print('EXECUTION STOPPED.')
                sys.exit()
            else:
                input_volume = input_volume[0]
                probtrac_files.append(input_volume)
            if jobs['tessprobtrack']:
                output_surface = input_volume.replace('.nii.gz', '.surf')
                freesurfer.tesselate(input_volume, self.tess_probtrack_threshold, output_volume=None, normalize=True,
                                     normalize_by=self.nsamples, output_surface=output_surface,
                                     smooth_surface_iterations=self.smooth_surface_iterations)
        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)
        return email_body

    def run_morphprobtrack(self, subj, email_body):
        log = '\n* Morphing probabilistic tractography to CVS template.'
        print(log)
        email_body.append(log)
        tstart = time.time()
        log = '    + Starting at : ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(log)

        email_body.append(log)
        probtrac_cvs_dir_native = self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts_cvs_avg35' + os.sep
        probtrac_cvs_dir = self.SUBJECTS_DIR + subj + '/probtrac_contacts_cvs_avg35/'
        os.makedirs(probtrac_cvs_dir_native, exist_ok=True)
        #        Get file names from folder
        probtrac_files = [f for f in glob.glob(
            self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts' + os.sep + '*.nii.gz')]

        if sys.platform == 'win32':
            probtrac_files = [paths.win2wsl(pf) for pf in probtrac_files]

        # don't know why, but sometimes bash process appears suspended in Windows when running multiple instances
        # rerun cvs_apply_morph until all files have been processed
        need_to_rerun = True
        while need_to_rerun:
            probtrac_files_in_cvs = [f for f in glob.glob(
                self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'probtrac_contacts_cvs_avg35' + os.sep + '*.mgz')]
            files_left = [x for x in [os.path.basename(p) for p in [pf.replace('.nii.gz', '') for pf in probtrac_files]]
                          if
                          x not in [os.path.basename(pc) for pc in
                                    [pfc.replace('.mgz', '') for pfc in probtrac_files_in_cvs]]]
            new_probtrac_files = []
            for f in files_left:
                new_probtrac_files.extend([p for p in probtrac_files if f in p])
            probtrac_files = new_probtrac_files
            if len(probtrac_files) > 0:
                self.run_cvs_apply_morph(probtrac_files, probtrac_cvs_dir)
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

        log = '    + Finished in ' + str((time.time() - tstart) / 3600) + ' hours.'
        print(log)
        email_body.append(log)
        return email_body

    def __init__(self, subj, args):
        self.subj = subj
        self.verbose = False

        # PARAMETERS (using paths in WSL format, ie. /mnt/d/....)
        job_list = ['verbose', 'recon', 'tracula', 'cvs_subj2mni', 'cvs_mni2subj', 'save_contact_coordinates',
                    'morphcontacts',
                    'probtrack', 'tessprobtrack', 'morphprobtrack', 'send_notification_when_done']
        jobs={}
        _ = [jobs.__setitem__(k, True) for k in job_list]

        data_dir, allowed_jobs = self.run_data_consistency_checks(subj)
        jobs = self.setup_jobs(args, allowed_jobs, jobs)
        print(args)
        print(jobs)
        print('\nNumber of CPUs: ' + str(self.CPU_COUNT) + '.')
        print('RAM: ' + str(self.MAX_RAM_SIZE) + ' Gb.\n')

        email_body = [subj]

        # RECON
        if jobs['recon']:
            email_body = self.run_recon(subj, data_dir, jobs, email_body)

        # save contact coordinates in Freesurfer's space
        if jobs['save_contact_coordinates']:
            email_body = self.run_save_contact_coordinates(subj, data_dir, email_body)

        # CVS
        if jobs['cvs_subj2mni']:
            email_body = self.run_cvs_subj2mni(subj, jobs, email_body)

        if jobs['cvs_mni2subj']:
            email_body = self.run_cvs_mni2subj(subj, jobs, email_body)

        # Morph contacts' coordinates
        if jobs['morphcontacts']:
            email_body = self.run_morphcontacts(subj, email_body)

        # DTI
        if (data_dir['dtifile'] is not None) and jobs['tracula']:
            email_body = self.run_convert_dti_dicoms(subj, data_dir, email_body)
        else:
            # no dtifile --> no tracula
            jobs['tracula'] = False

        if jobs['tracula']:
            email_body = self.run_tracula(subj, data_dir, email_body)

        if jobs['probtrack']:
            email_body = self.run_probtrack(subj, jobs, email_body)

        if jobs['morphprobtrack']:
            email_body = self.run_morphprobtrack(subj, email_body)

        if jobs['send_notification_when_done']:
            try:
                notifications.sendmail(to_address=self.send_notifications_to_email, subject='New patient job done.',
                                       message='\n'.join(email_body))
            except:
                pass

        # save log in subject's output folder
        with open(self.SUBJECTS_DIR_NATIVE + subj + os.sep + 'log.txt', 'wt') as file:
            file.write('\n'.join(email_body))
