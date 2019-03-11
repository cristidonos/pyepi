"""
Interface to Freesurfer
"""
import os
import sys
import subprocess

from pyepi.tools import paths
from pyepi.tools import volumes
import pathlib


def recon(subj=None, t1_file=None, t2_file=None, openmp=None, verbose=0):
    """Wrapper for Freesurfer's recon script.

    Parameters
    ----------
    subj: string
        Subject ID
    t1_file: string
        Path to T1 file
    t2_file: string
        Path to T2 file (optional)
    openmp: int
        Number of cpus to use for parallel processing ([0..15])
    verbose: int
        If =1 will print bash command output and errors even if command was successful.


    """

    if subj is not None:
        cmd = ['recon-all -s ' + subj,
               '-all']
    else:
        print('\n! Must specify a subject ID.\n')
        return

    cmd.append('-i ' + t1_file)

    if t2_file is not None:
        cmd.append('-T2 ' + t2_file)

    if openmp is not None:
        if 0 < openmp <= 16:
            cmd.append('-openmp ' + str(openmp))
        else:
            print('\n! Openmp parameter should be an integer in [1..15] range.\n')
            return
    cmd = ' '.join(cmd)
    result = subprocess.run(['bash',  '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))


def cvs_subj2mni(subj=None, openmp=None, verbose=0):
    """Wrapper for Freesurfer's CVS registration of subject to cvs_avg25_inMNI152 template.

    Parameters
    ----------
    subj: string
        Subject ID
    openmp: int
        Number of cpus to use for parallel processing ([0..15])
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """

    if subj is not None:
        cmd = ['mri_cvs_register --mov ' + subj,
               '--template cvs_avg35_inMNI152 --nocleanup ']
    else:
        print('\n! Must specify a subject ID.\n')
        return
    if openmp is not None:
        if 0 < openmp <= 16:
            cmd.append('--openmp ' + str(openmp))
        else:
            print('\n! Openmp parameter should be an integer in [1..15] range.\n')
            return
    cmd = ' '.join(cmd)

    setup_cmd = 'export PATH=$PATH:$FREESURFER_HOME/bin && source $FREESURFER_HOME/SetUpFreeSurfer.sh'
    # cmd = ' && '.join([setup_cmd, cmd])
    result = subprocess.run(['bash',  '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))


def cvs_mni2subj(subj=None, subjects_dir=None, openmp=None, verbose=0):
    """Wrapper for Freesurfer's CVS registration of cvs_avg25_inMNI152 template to subject's space.

    Parameters
    ----------
    subj: string
        Subject ID
    subjects_dir: string
        Path to Freesurfer's SUBJECTS_DIR
    openmp: int
        Number of cpus to use for parallel processing ([0..15])
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """

    if (subj is not None) and (subjects_dir is not None):
        cmd = ['mri_cvs_register --mov cvs_avg35_inMNI152 --template ' +
               subj + ' --outdir ' + subjects_dir + '//' + subj + '//' + 'cvs_avg35_inMNI135_to_subj --nocleanup ']
    else:
        print("\n! Must specify a subject ID and Freesurfer's SUBJECTS_DIR.\n")
        return
    if openmp is not None:
        if 0 < openmp <= 16:
            cmd.append('--openmp ' + str(openmp))
        else:
            print('\n! Openmp parameter should be an integer in [1..15] range.\n')
            return
    cmd = ' '.join(cmd)

    setup_cmd = 'export PATH=$PATH:$FREESURFER_HOME/bin && source $FREESURFER_HOME/SetUpFreeSurfer.sh'
    # cmd = ' && '.join([setup_cmd, cmd])
    result = subprocess.run(['bash',  '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))


def cvs_apply_morph(subj, subjects_dir, volume, output_volume, output_dir, morph_to_cvs=True, interpolation='linear',
                    verbose=0):
    """ Apply CVS morph to a given volume

    subj: string
        subject id
    subjects_dir: string
        Freesurfer's subjects dir
    volume: string
        Volume to be morphed
    output_volume: string
        Output volume name (ex: test.mgz)
    output_dir: string
        Path to output_volume
    morph_to_cvs: bool
        If True, will morph volume to CVS template, if False will morph volume from CVS template space to subj's space
    interpolation: string
        Morph interpolation type. Use 'linear' for continous intensity distributions and 'nearest' for atlas-like volumes (ex: aseg)
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """
    if morph_to_cvs:
        cmd = ('applyMorph --template ' +
               subjects_dir + 'cvs_avg35_inMNI152/mri/norm.mgz' +
               ' --transform ' + subjects_dir + subj +
               '/cvs/el_reg_tocvs_avg35_inMNI152.tm3d  vol ' +
               volume + ' ' +
               output_dir + output_volume + ' ' + interpolation)
    else:
        cmd = ('applyMorph --template ' +
               subjects_dir + subj + '/mri/norm.mgz' +
               ' --transform ' + subjects_dir + subj +
               '/cvs_avg35_inMNI135_to_subj/el_reg_to' + subj + '.tm3d  vol ' +
               volume + ' ' +
               output_dir + output_volume + ' ' + interpolation)
    # result = subprocess.run(['bash', '-i', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(['bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print('starting new morph')
        # for line in paths.execute(' '.join(['bash', '-c', '"', cmd ,'"'])):
        #     print(line)

    if (verbose == 1) or (len(result.stderr) > 0):
            print('\n\n' + result.stdout.decode('utf-8'))
            print('\n' + result.stderr.decode('utf-8'))


def tracula_config(subj, dicom, config_folder=None, subjects_dir=None, dtroot=None, bvecfile=None, bvalfile=None,
                   doeddy=True, dorotbvecs=True, doregbbr=True, doregmni=True, doregcvs=False, nstick=2, nburnin=200,
                   nsample=7500, nkeep=5):
    """Generate Freesurfer Tracula's configuration file (https://surfer.nmr.mgh.harvard.edu/fswiki/dmrirc)

    Parameters
    ----------
    subj: string
        subject id
    dicom: string
        path to diffusion dicom or nifti file output of dcm2niix
    config_foder: string
        path to folder where the config file is to be generated
    subjects_dir: string
        Freesurfer's subjects dir
    dtroot: string
        output folder for Tracula's results
    bvecfile: string
        path to .bvec file (use with nifti input from dcm2niix)
    bvalfile: string
        path to .bval file (use with nifti input from dcm2niix)
    doeddy: bool
        Perform registration-based eddy-current compensation? (True / False)
    dorotbvecs: bool
        Rotate diffusion gradient vectors to match eddy-current compensation?  (True / False)
    doregbbr: bool
        Perform diffusion-to-T1 registration by bbregister? (True / False)
    doregmni: bool
        Perform registration of T1 to MNI template?
    doregcvs: bool
        Perform registration of T1 to CVS template?
    nstick: int
        Number of "sticks" (anisotropic diffusion compartments) in the bedpostx
    nburnin: int
        Number of MCMC burn-in iterations
    nsample: int
        Number of MCMC iterations
    nkeep: int
        Frequency with which MCMC path samples are retained for path distribution

    """
    doeddy = 1 if doeddy else 0
    dorotbvecs = 1 if dorotbvecs else 0
    doregbbr = 1 if doregbbr else 0
    doregmni = 1 if doregmni else 0
    doregcvs = 1 if doregcvs else 0

    if config_folder is None:
        config_folder = str(pathlib.Path.home())

    lines = []

    if subjects_dir is not None:
        lines.append('setenv SUBJECTS_DIR ' + subjects_dir)

    lines.append('set subjlist = (' + subj + ')')
    lines.append('set dcmroot = ' + os.path.dirname(dicom))
    lines.append('set dcmlist = (' + os.path.basename(dicom) + ')')

    if bvecfile is None:
        lines.append(
            'set bvecfile = (' + os.path.dirname(dicom) + '/' + os.path.splitext(os.path.basename(dicom))[0] + '.bvec)')
    else:
        lines.append('set bvecfile = (' + bvecfile + ')')

    if bvalfile is None:
        lines.append(
            'set bvalfile = (' + os.path.dirname(dicom) + '/' + os.path.splitext(os.path.basename(dicom))[0] + '.bval)')
    else:
        lines.append('set bvalfile = (' + bvalfile + ')')

    lines.append('set doeddy = ' + str(doeddy))
    lines.append('set dorotbvecs = ' + str(dorotbvecs))
    lines.append('set doregbbr = ' + str(doregbbr))
    lines.append('set doregmni = ' + str(doregmni))
    lines.append('set doregcvs = ' + str(doregcvs))
    lines.append('set nstick = ' + str(nstick))
    lines.append('set nburnin = ' + str(nburnin))
    lines.append('set nsample = ' + str(nsample))
    lines.append('set nkeep = ' + str(nkeep))

    with open(config_folder + os.sep + subj + '_tracula_config', 'w', newline='\n') as f:
        for line in lines:
            f.write(line + '\n')

    return config_folder + os.sep + subj + '_tracula_config'


def tracula_run(subj, prep=True, bedp=True, path=True, cfg_file=None, verbose=0):
    """Wrapper for Tracula

    Parameters
    ----------
    subj: string
        Subject's ID
    prep: bool
        If True will run Tracula's pre-processing step
    bedp: bool
        If True will run Tracula's bedpostX step
    path: bool
        If True will run Tracula's pathways reconstruction step
    cfg_file: string
        Path to the configuration file
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """
    commands = []

    if prep:
        commands.append(
            'trac-all -prep -c ' + cfg_file)
    if bedp:
        commands.append(
            'trac-all -bedp -c ' + cfg_file)
    if path:
        commands.append(
            'trac-all -path -c ' + cfg_file)

    cmd = ' && '.join(commands)

    result = subprocess.run(['bash',  '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))


def tesselate(input_volume, threshold, output_surface=None, smooth_surface_iterations=None, normalize=False,
              normalize_by='max', output_volume=None,
              verbose=0):
    """Tesselate mgz/nifti volume to create surface, or output a thresholded volume

    Parameters
    ----------
    input_volume: string
        Path to input mgz/nifti volume (in system's native format)
    threshold: float
        Threshold for input volume. Only values higher than threshold will be tesselated / kept.
    output_surface: string
        Path to output surface (in system's native format)
    normalize: bool
        If True, the input volume will be normalized with the "normalize_by" value before thresholding.
    normalize_by: float
        Value to normalize input volume by.
    output_volume: string
        Path to output volume (in system's native format)
    verbose: int
        If =1 will print bash command output and errors even if command was successful.
    """
    commands = []
    if output_surface is None and output_volume is None:
        print("\n! Must specify at least an output_surface or an output_volume.\n")
        return
    if input_volume is not None:
        if normalize:
            normalized_volume_native, normalized_volume_wsl = paths.wsl_tempfile('normalized_volume.mgz')
            if sys.platform == 'win32':
                volumes.normalize(input_volume, normalized_volume_native, normalize_by=normalize_by)
                commands = ['mri_binarize --i ' + normalized_volume_wsl]
            else:
                volumes.normalize(input_volume, normalized_volume_native, normalize_by=normalize_by)
                commands = ['mri_binarize --i ' + normalized_volume_native]
        else:
            commands = ['mri_binarize --i ' + input_volume]
    else:
        print("\n! Must specify an input volume to tesselate.\n")
        return
    if threshold is not None:
        commands.append('--min ' + str(threshold))
    else:
        print("\n! Must specify a threshold to tesselate.\n")
        return
    if output_volume is not None:
        if sys.platform == 'win32':
            commands.append('--o ' + paths.win2wsl(output_volume))
        else:
            commands.append('--o ' + output_volume)
    if output_surface is not None:
        if sys.platform == 'win32':
            commands.append('--surf ' + paths.win2wsl(output_surface))
        else:
            commands.append('--surf ' + output_surface)
        if smooth_surface_iterations is not None:
            commands.append('--surf-smooth ' + str(smooth_surface_iterations))

    cmd = ' '.join(commands)
    result = subprocess.run(['bash',  '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    paths.silentremove(normalized_volume_native)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))


def dcm2niix(dcm_file, output_filename, output_folder, executable_path='', verbose=0):
    """Convert DICOM files to Nifti using dcm2niix. (dcm2niix needs to be in PATH and accessible in bash, download it
    from https://github.com/rordenlab/dcm2niix/releases

    Parameters
    ----------
    dcm_file: string
        Path to the first dicom in the series
    output_filename: string
        Name of the output Nifti volume
    output_folder: string
        Path to the folder where Nifti volume will be saved
    executable_path: string
        Path to where dcm2niix is located. Optional, use only is dcm2niix is not in PATH.
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """
    paths.silentremove(output_folder + output_filename + '.nii')
    paths.silentremove(output_folder + output_filename + '.bvec')
    paths.silentremove(output_folder + output_filename + '.bval')
    paths.silentremove(output_folder + output_filename + '.json')
    if (':' not in output_folder) and sys.platform == 'win32':
        os.makedirs(paths.wsl2win(output_folder), exist_ok=True)
    else:
        os.makedirs(output_folder, exist_ok=True)  # try, see if it works...id Windows Linux Subsytem is use it won't

    os.makedirs(output_folder, exist_ok=True)  # try, see if it works...id Windows Linux Subsytem is use it won't
    if executable_path is not None:
        cmd = executable_path + 'dcm2niix -f "' + output_filename + '" -d y -o "' + output_folder + '" "' + dcm_file + '"'
    else:
        cmd = 'dcm2niix -f "' + output_filename + '" -d y -o "' + output_folder + '" "' + dcm_file + '"'
    cmd = ' && '.join(['mkdir -p ' + output_folder, cmd])
    result = subprocess.run(['bash',  '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))
        print('\n' + result.args)



