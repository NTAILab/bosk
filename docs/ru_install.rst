Установка bosk
==============

Пожалуйста, внимательно прочитайте эту инструкцию, так как пакет bosk зависит от библиотеки JAX,
которая не может быть установлена в автоматическом режиме.

Перед установкой
----------------

Окружение
~~~~~~~~~

Убедитесь, что Вы создали изолированное окружение для python версии не ниже 3.9.

Например, с помощью `Anaconda <https://www.anaconda.com/distribution/>`_
Вы можете создать и активировать окружение, выполнив::

    conda create -n bosk_env python=3.10
    conda activate bosk_env

.. _install-jax:

Установка JAX
~~~~~~~~~~~~~

Если необходимо установить bosk без поддержки JAX, перейдите к
:ref:`установке пакета <install-package>`.

Bosk использует JAX для вычислений на графическом процессоре, но установка JAX не тривиальна.
Официально JAX распространяется только для Linux и Mac OS, поэтому, к сожалению,
пользователям Windows следует использовать `WSL <https://docs.microsoft.com/en-us/windows/wsl/about>`_ 
для установки JAX и использования bosk с поддержкой вычислений на ГПУ.

**Установка только для ЦПУ**

Если на Вашей системе нет графического процессора, Вы можете установить версию JAX для ЦПУ::

    pip install --upgrade "jax[cpu]==0.4.10"

**Установка с поддержкой вычислений на ГПУ**

Пожалуйста, обратитесь к `официальной инструкции по установке JAX <https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier>`_
или выполните следующую команду для установки версии с поддержкой CUDA12::

    pip install --upgrade "jax[cuda12_pip]==0.4.10" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

.. note::
    Обратите внимание, что, согласно `руководству NVIDIA <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_, 
    пользователи Windows не должны устанавливать какой-либо драйвер NVIDIA GPU Linux внутри WSL 2. 
    Участники NTAILab, использующие Windows, установливали CUDA Toolkit и cuDNN вручную,
    поэтому мы можем рекомендовать только `часть руководства <https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally -harder>`_, 
    где предустановленная копия CUDA используется для установки JAX поверх нее.

.. _install-package:

Установка пакета
----------------

Для установки пакета bosk напрямую из GitHub выполните::

    pip install git+ssh://git@github.com:NTAILab/bosk.git

Также Вы можете вручную скопировать репозиторий и установить bosk::

    git clone git@github.com:NTAILab/bosk.git
    cd bosk
    pip install -r requirements.txt
    python setup.py install

.. _dev_install:

Установка в режиме разработчика
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Если Вы планируете вносить изменения в bosk, будет удобнее установить пакет в режиме
разработчика, чтобы Python сам регистрировал вносимые изменения при запуске::

    git clone git@github.com:NTAILab/bosk.git
    cd bosk
    pip install -r requirements.txt
    python setup.py develop
