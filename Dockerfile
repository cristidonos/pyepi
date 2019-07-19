FROM cristidonos/ubuconda3


RUN apt-get update && \
    apt-get install -y build-essential \
                       tcsh \
                       libtool-bin \
                       libtool \
                       automake \
                       gfortran \
                       libglu1-mesa-dev \
                       libfreetype6-dev \
                       uuid-dev \
                       libxmu-dev \
                       libxmu-headers \
                       libxi-dev \
                       libx11-dev \
                       libxml2-utils \
                       libxt-dev \
                       libjpeg62-dev \
                       libxaw7-dev \
                       liblapack-dev \
                       git \
                       gcc-4.8 \
                       g++-4.8 \
                       libgfortran-4.8-dev \
                       curl\
                       wget\
                       python-pip\
                       bc


RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

## Add sudo
#RUN apt-get -y install sudo
#
## Add user 'pyepi' with no password, add to sudo group
#RUN adduser --disabled-password --gecos '' pyepi
#RUN adduser pyepi sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER pyepi

ARG working_dir=/home/
WORKDIR $working_dir
RUN chmod a+rwx $working_dir

# Freesurfer
RUN wget -qO- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz | tar zxv --no-same-owner -C /$working_dir

# Configure environment
ENV FSLDIR=/usr/share/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH=/usr/lib/fsl:$PATH
ENV FSLMULTIFILEQUIT=TRUE
ENV POSSUMDIR=/usr/share/fsl
ENV LD_LIBRARY_PATH=/usr/lib/fsl:$LD_LIBRARY_PATH
ENV FSLTCLSH=/usr/bin/tclsh
ENV FSLWISH=/usr/bin/wish
ENV FSLOUTPUTTYPE=NIFTI_GZ

ENV OS Linux
ENV FS_OVERRIDE 0
ENV FIX_VERTEX_AREA=
#ENV SUBJECTS_DIR /$working_dir/freesurfer/subjects
ENV SUBJECTS_DIR /$working_dir/host/subjects
ENV FSF_OUTPUT_FORMAT nii.gz
ENV MNI_DIR /$working_dir/freesurfer/mni
ENV LOCAL_DIR /$working_dir/freesurfer/local
ENV FREESURFER_HOME /$working_dir/freesurfer
ENV FSFAST_HOME /$working_dir/freesurfer/fsfast
ENV MINC_BIN_DIR /$working_dir/freesurfer/mni/bin
ENV MINC_LIB_DIR /$working_dir/freesurfer/mni/lib
ENV MNI_DATAPATH /$working_dir/freesurfer/mni/data
ENV FMRI_ANALYSIS_DIR /$working_dir/freesurfer/fsfast
ENV PERL5LIB /$working_dir/freesurfer/mni/lib/perl5/5.8.5
ENV MNI_PERL5LIB /$working_dir/freesurfer/mni/lib/perl5/5.8.5
ENV PATH /$working_dir/freesurfer/bin:/$working_dir/freesurfer/fsfast/bin:/$working_dir/freesurfer/tktools:/$working_dir/freesurfer/mni/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=""

# Copy freesurfer license
COPY pyepi/extra/license.txt $working_dir/freesurfer/license.txt

# Copy maTT to home folder
RUN cp -R pyepi/extra/multiAtlasTT/ $working_dir

# Add Anaconda packages
ENV PATH /opt/conda/bin:$PATH
RUN conda install numpy pandas xlrd openpyxl tqdm matplotlib nibabel git mayavi psutil seaborn jupyter_contrib_nbextensions dcm2niix pip -c conda-forge

# add more packages with pip
RUN ["/bin/bash", "-c","\
    pip install git+https://github.com/alex-sherman/deco &&\
    pip install git+https://github.com/cristidonos/PySurfer &&\
    pip install git+https://github.com/cristidonos/pyepi"]

#install fsl
COPY pyepi/extra/fslinstaller.py $working_dir/fslinstaller.py
RUN /usr/bin/python2 $working_dir/fslinstaller.py -d /usr/share/fsl
#RUN apt-get install -y fsl
ENV FSLDIR=/usr/share/fsl
ENV PATH=/usr/bin:$PATH:$FSLDIR/bin
ENV LD_LIBRARY_PATH=/usr/lib/fsl:/usr/share/fsl/bin

RUN sed -i 's;#!/bin/sh;#!/bin/bash;g' /home/freesurfer/bin/bedpostx_mgh
RUN sed -i 's;#!/bin/sh;#!/bin/bash;g' /home/freesurfer/bin/fsl_sub_mgh

RUN alias python=/usr/bin/python2
RUN echo "PATH=/usr/bin:$PATH" > /root/.bashrc

# Configuring access to Jupyter
RUN mkdir /home/notebooks
RUN jupyter notebook --generate-config --allow-root
#RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /home/pyepi/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupytewr notebook as Docker main process
#CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/home/pyepi/notebooks", "--ip='*'", "--port=8888", "--no-browser"]
CMD ["/bin/bash"]
