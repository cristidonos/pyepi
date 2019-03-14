=====
Usage
=====

| To install Pyepi, you need to build the Docker container first.
| The Docker container is based on Ubuntu16 and has installed Miniconda3 with Python3 and all necessary packages, Freesurfer 6, FSL 5.0.
| Clone Pyepi, change dir to Pyepi's root folder and run: ::

    docker build -t pyepi .

You can start the container in interactive mode using "pyepi" hostname  to run: ::


    docker run -it -P --hostname pyepi



=====
Custom Pipelines
=====

However, it is best to write pipelines to automate data processing. A sample pipeline used at the University of Bucharest is provided as an example. To run the pipeline, Pyepi assumes the following folder structure:

 | PARENT_FOLDER
 |    |-- rawdata
 |       |-- subject (i.e. SEEG72)
 |       |-- DTI (diffusion dicoms)
 |       |-- DWI (this one gets created during preprocessing, contains diffusion dicoms converted to Nifti and bval/bvec files)
 |       |-- T1  (T1 dicoms)
 |       |-- T2  (T2 dicoms, optional)
 |       |-- Patient Data.xlsx  (format detailed in io.load_contacts , optional)
 |       |-- subject.ppr  (ppr file containing the final trajectories, optional)
 |
 |    |-- subjects (where output will be saved. Subfolders with subject's id will be created automatically)
 |       |-- SUBJ001
 |       |-- SUBJ002 and so on.

To start the container with a mounted volume, use the command below, with the correct PARENT_FOLDER path: ::

    docker run -it -P --hostname pyepi -v /PARENT_FOLDER/:/home/host/ pyepi



Run the UNIBUC pipeline with: ::

    epi-pipe newpatient SUBJID

