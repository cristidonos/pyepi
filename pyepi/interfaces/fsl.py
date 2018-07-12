"""
Interface to FSL
"""
import os
import subprocess
import tempfile
from pyepi.tools import paths
import pathlib


def probtrack_with_seedmask(subj=None, bedpostx_folder=None, seedmask=None, seed2diff_xfm=None, avoidmask=None,
                            terminationmask=None, output_dir=None, waypoints=None, waycond='AND', loopcheck=True,
                            onewaycondition=True, curvature_thr=0.2, nsamples=5000, nsteps=2000, step_length=0.5,
                            fib_thr=0.01, dist_thr=0.0, sampling_radius=0, verbose=0):
    """Probabilistic tractography with seed mask, using FSL's probtrackx2

    Parameters
    ----------
    subj: string
        Subject id
    bedpostx_folder: string
        Path to subject's bedpostX folder
    seedmask: string
        Nifti seed volume
    seed2diff_xfm: string
        Linear transformation matrix from seed space to diffusion space
    avoidmask: string
        Reject pathways passing through locations given by this volume mask
    terminationmask: string
        Stop tracking at locations given by this mask file
    output_dir: string
        Path to folder where output will be generated
    waypoints: string
        Path to Nifti volume which will be used as ROI. Multiple volumes can be used.
    waycond: string
        Waypoint condition, can be "AND" , "OR"
    loopcheck: bool
        Perform loopchecks on paths - slower, but allows lower curvature threshold
    onewaycondition: bool
        Apply waypoint conditions to each half tract separately
    curvature_thr: float
        Curvature threshold - default=0.2
    nsamples: int
        Number of samples - default=5000
    nsteps: int
        Number of steps per sample - default=2000
    step_length: float
        Steplength in mm - default=0.5
    fib_thr: float
        Volume fraction before subsidary fibre orientations are considered - default=0.01
    dist_thr: int
        Discards samples shorter than this threshold (in mm - default=0)
    sampling_radius: float
        Sample random points within a sphere with radius x mm from the center of the seed voxels (e.g. --sampvox=0.5, 0.5 mm radius sphere). Default=0
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """
    cmd = ['probtrackx2 ',
           '-s ' + bedpostx_folder + os.sep + 'merged',
           '-m ' + bedpostx_folder + os.sep + 'nodif_brain_mask',
           '-x ' + seedmask,
           ]

    if seed2diff_xfm is not None:
        cmd.append('--xfm=' + seed2diff_xfm)
    if avoidmask is not None:
        cmd.append('--avoid=' + avoidmask)
    if terminationmask is not None:
        cmd.append('--stop=' + terminationmask)
    if output_dir is not None:
        cmd.append('--dir=' + output_dir)
        cmd.append('--forcedir')
    if waypoints is not None:
        tmpdir = tempfile.TemporaryDirectory()
        with open(tmpdir.name + os.sep + 'waypoints.txt', 'w') as wf:
            wpts = waypoints.split(' ')
            wf.write('\n'.join(wpts))
        cmd.append('--waypoints=' + tmpdir.name + os.sep + 'waypoints.txt')

    cmd.append('--waycond=' + waycond)
    if loopcheck:
        cmd.append('-l')
    if onewaycondition:
        cmd.append('--onewaycondition')

    cmd.append('-c ' + str(curvature_thr))
    cmd.append('-P ' + str(nsamples))
    cmd.append('-S ' + str(nsteps))
    cmd.append('--steplength=' + str(step_length))
    cmd.append('--fibthresh=' + str(fib_thr))
    cmd.append('--distthresh=' + str(dist_thr))
    cmd.append('--sampvox=' + str(sampling_radius))
    cmd.append('--opd')  # output pathways distribution anyway

    cmd = ' '.join(cmd)
    print(cmd)
    result = subprocess.run(['bash', '-i', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))


def probtrack_with_seedcoords(subj=None, bedpostx_folder=None, seedcoords=None, seed2diff_xfm=None, seedref=None,
                              avoidmask=None,
                              terminationmask=None, output_dir=None, output_file=None, waypoints=None, waycond='AND',
                              loopcheck=True,
                              onewaycondition=True, curvature_thr=0.2, nsamples=5000, nsteps=2000, step_length=0.5,
                              fib_thr=0.01, dist_thr=0.0, sampling_radius=0, verbose=0):
    """Probabilistic tractography with seed mask, using FSL's probtrackx2

    Parameters
    ----------
    subj: string
        Subject id
    bedpostx_folder: string
        Path to subject's bedpostX folder
    seedcoords: np.array
        Array of X, Y, Z seed coordinates in mm
    seed2diff_xfm: string
        Linear transformation matrix from seed space to diffusion space
    seedref: string
        Reference vol to define seed space - diffusion space assumed if absent
    avoidmask: string
        Reject pathways passing through locations given by this volume mask
    terminationmask: string
        Stop tracking at locations given by this mask file
    output_dir: string
        Path to folder where output will be generated
    output_file: string
        Output filename  (default='fdt_paths')
    waypoints: string
        Path to Nifti volume which will be used as ROI. Multiple volumes can be used.
    waycond: string
        Waypoint condition, can be "AND" , "OR"
    loopcheck: bool
        Perform loopchecks on paths - slower, but allows lower curvature threshold
    onewaycondition: bool
        Apply waypoint conditions to each half tract separately
    curvature_thr: float
        Curvature threshold - default=0.2
    nsamples: int
        Number of samples - default=5000
    nsteps: int
        Number of steps per sample - default=2000
    step_length: float
        Steplength in mm - default=0.5
    fib_thr: float
        Volume fraction before subsidary fibre orientations are considered - default=0.01
    dist_thr: int
        Discards samples shorter than this threshold (in mm - default=0)
    sampling_radius: float
        Sample random points within a sphere with radius x mm from the center of the seed voxels (e.g. --sampvox=0.5, 0.5 mm radius sphere). Default=0
    verbose: int
        If =1 will print bash command output and errors even if command was successful.

    """
    cmd = ['probtrackx2 ',
           '--simple',
           '-s ' + bedpostx_folder + os.sep + 'merged',
           '-m ' + bedpostx_folder + os.sep + 'nodif_brain_mask',
           '-x ' + seedcoords,
           ]

    if seed2diff_xfm is not None:
        cmd.append('--xfm=' + seed2diff_xfm)
    if seedref is not None:
        cmd.append('--seedref=' + seedref)
    if avoidmask is not None:
        cmd.append('--avoid=' + avoidmask)
    if terminationmask is not None:
        cmd.append('--stop=' + terminationmask)
    if output_dir is not None:
        cmd.append('--dir=' + output_dir)
        cmd.append('--forcedir')
    if output_file is not None:
        cmd.append('-o ' + output_file)
    if waypoints is not None:
        # TODO cross platform implementation of waypoints.txt file
        tmpdir = tempfile.TemporaryDirectory()
        with open(tmpdir.name + os.sep + 'waypoints.txt', 'w') as wf:
            wpts = waypoints.split(' ')
            wf.write('\n'.join(wpts))
        cmd.append('--waypoints=' + tmpdir.name + os.sep + 'waypoints.txt')

    cmd.append('--waycond=' + waycond)
    if loopcheck:
        cmd.append('-l')
    if onewaycondition:
        cmd.append('--onewaycondition')

    cmd.append('-c ' + str(curvature_thr))
    cmd.append('-P ' + str(nsamples))
    cmd.append('-S ' + str(nsteps))
    cmd.append('--steplength=' + str(step_length))
    cmd.append('--fibthresh=' + str(fib_thr))
    cmd.append('--distthresh=' + str(dist_thr))
    cmd.append('--sampvox=' + str(sampling_radius))
    cmd.append('--opd')  # output pathways distribution anyway

    cmd = ' '.join(cmd)
    result = subprocess.run(['bash', '-i', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (verbose == 1) or (len(result.stderr) > 0):
        print('\n\n' + result.stdout.decode('utf-8'))
        print('\n' + result.stderr.decode('utf-8'))
