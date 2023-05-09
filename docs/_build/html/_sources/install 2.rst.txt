Installing bosk
===============

Please, read this guide carefully, because the package depends on the JAX,
which cannot be installed automatically.

If it is needed to install bosk without JAX support, start with
:ref:`install-package`.


Prerequisites
-------------


Environment
~~~~~~~~~~~

Make sure that the python >= 3.9 envoronment is ready.

For example, with `Anaconda <https://www.anaconda.com/distribution/>`_
the environment can be created and activated by running::

    conda create -n bosk_env python=3.10
    conda activate bosk_env

.. _install-jax:

JAX installation
~~~~~~~~~~~~~~~~

Bosk uses JAX for GPU computations, but JAX installation is not trivial.

**CPU-only system**

If there is no GPU available, install the CPU JAX version::

    pip install --upgrade "jax[cpu]"

**GPU system**

Please, follow `The official JAX installation guide <https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier>`_
or run the following command to install CUDA12 version::
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Note that pip wheels are available only for linux, for other platforms either adapters
like Windows Subsystem for Linux can be used or JAX can be built manually.

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


Dev Installation
~~~~~~~~~~~~~~~~

For development purpose it is more convenient to install the package in develop mode,
to automatically update package on changes (actually it uses symlinks)::

    git clone git@github.com:NTAILab/bosk.git
    cd bosk
    pip install -r requirements.txt
    python setup.py develop
