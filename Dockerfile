FROM erdc/stack_base:python3

MAINTAINER Proteus Project <proteus@googlegroups.com>

USER root

RUN curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
RUN sudo apt-get install nodejs

USER jovyan

WORKDIR /home/$NB_USER

ENV CC mpicc
ENV CXX mpicxx
ENV F77 mpif77
ENV F90 mpif90

RUN cd proteus && git checkout master && git pull && make develop

ENV PATH /home/$NB_USER/proteus/linux/bin:$PATH
ENV LD_LIBRARY_PATH /home/$NB_USER/proteus/linux/lib:$LD_LIBRARY_PATH

RUN cd proteus && git pull && export PATH=${HOME}/bin:${PATH} && make lfs && git lfs fetch && git lfs checkout
RUN cd proteus && export PATH=${HOME}/bin:${PATH} && echo $PATH && ln -s /usr/bin/pkg-config ${HOME}/bin && hash -r && ./linux/bin/pip install matplotlib
RUN cd proteus && make jupyter
USER root

RUN ipython kernel install

USER $NB_USER

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"
