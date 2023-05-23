Installing bosk
===============

Please, read this guide carefully, because the package depends on the JAX,
which cannot be installed automatically.


Prerequisites
-------------


Environment
~~~~~~~~~~~

Make sure that the python >= 3.9 environment is ready.

For example, with `Anaconda <https://www.anaconda.com/distribution/>`_
the environment can be created and activated by running::

    conda create -n bosk_env python=3.10
    conda activate bosk_env

.. _install-jax:

JAX installation
~~~~~~~~~~~~~~~~

If it is needed to install bosk without JAX support, go to
:ref:`install-package`.

Bosk uses JAX for GPU computations, but JAX installation is not trivial.
Officially JAX is only distributed for Linux and Mac OS, so, unfortunatelly,
Windows users should use `WSL <https://docs.microsoft.com/en-us/windows/wsl/about>`_
to install JAX and use bosk with GPU support.

**CPU-only system**

If there is no GPU available, install the CPU JAX version::

    pip install --upgrade "jax[cpu]==0.4.10"

**GPU system**

Please, follow `The official JAX installation guide <https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier>`_
or run the following command to install CUDA12 version::
    
    pip install --upgrade "jax[cuda12_pip]==0.4.10" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

.. note::
    Notice that, according to the `NVIDIA guide <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_, Windows users
    must not install any NVIDIA GPU Linux driver within WSL 2. Those NTAILab participants, who uses Windows,
    installed CUDA Toolkit and cuDNN manually, so we only can recommend to use
    `part of the guide <https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder>`_
    where the preinstalled copy of CUDA is used to install JAX over it.

.. _install-package:

Package Installation
--------------------

To install the bosk package directly from GitHub run::

    pip install git+ssh://git@github.com:NTAILab/bosk.git

Alternatively the repo can be cloned and installed manually::

    git clone git@github.com:NTAILab/bosk.git
    cd bosk
    pip install -r requirements.txt
    python setup.py install

.. _dev_install:

Dev Installation
~~~~~~~~~~~~~~~~

For development purpose it is more convenient to install the package in develop mode,
to automatically update package on changes (actually it uses symlinks)::

    git clone git@github.com:NTAILab/bosk.git
    cd bosk
    pip install -r requirements.txt
    python setup.py develop
