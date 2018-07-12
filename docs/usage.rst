=====
Usage
=====

To use PyEpi in a project::

    import pyepi


Windows Subsystem for Linux  environment example::

    -------- freesurfer-Linux-centos7_x86_64-dev-20180227-5e5b9aa --------
    Setting up environment for FreeSurfer/FS-FAST (and FSL)
    FREESURFER_HOME   /usr/local/freesurfer
    FSFAST_HOME       /usr/local/freesurfer/fsfast
    FSF_OUTPUT_FORMAT nii.gz
    SUBJECTS_DIR      /mnt/d/CloudSynology/subjects
    MNI_DIR           /usr/local/freesurfer/mni


Various commands in WSL::

    epi-preproc recon --subject SEEG72 --t1_file /mnt/d/CloudSynology/rawdata/SEEG72/T1/IM0 --t2_file /mnt/d/CloudSynology/rawdata/SEEG72/T2/IM0 --openmp 4
    epi-preproc cvs-subj2mni --subject SEEG72
    epi-preproc cvs-mni2subj --subject SEEG72 --subjects_dir /mnt/d/CloudSynology/subjects/
    epi-preproc dcm2nii --dcm_file /mnt/d/CloudSynology/rawdata/SEEG72/DTI/IM0 --output_nii_filename dwi --output_nii_folder /mnt/d/CloudSynology/rawdata/SEEG72/DWI/
    epi-preproc tracula --subject SEEG72 --trac_prep 1 --trac_bedp 1 --trac_path 1 --openmp 4 --dwi_file /mnt/d/CloudSynology/rawdata/SEEG72/DWI/dwi.nii


All the above can be combined in one call as long as all necessary parameters are used (Windows example, using command line variables)::

    set subj=SEEG72
    set rawdir=/mnt/d/CloudSynology/rawdata
    epi-preproc recon_cvs-subj2mni_cvs-mni2subj_dcm2nii_tracula ^
        --subject %subj% ^
        --subjects_dir /mnt/d/CloudSynology/subjects/ ^
        --t1_file %rawdir%/%subj%/T1/IM0 ^
        --t2_file %rawdir%/%subj%/T2/IM0 ^
        --openmp 4 ^
        --dcm_file %rawdir%/%subj%/DTI/IM0 ^
        --output_nii_filename dwi ^
        --output_nii_folder %rawdir%/%subj%/DWI/ ^
        --trac_prep 1 ^
        --trac_bedp 1 ^
        --trac_path 1 ^
        --dwi_file %rawdir%/%subj%/DWI/dwi.nii


=====
Good to know.
=====
If running on Windows Subsytem for Linux, make sure to make link sh to bash, otherwise Tracula will crash during bedpostX step.::

    sudo rm /bin/sh
    sudo ln -s /bin/bash /bin/sh
