Contributing guidelines
=======================

You can take part in the bosk development and contribute to our project.
There are a lot of ways you can do it:

- Writing new code, e.g. implementations of new algorithms, or examples.
- Fixing bugs.
- Improving documentation.
- Reviewing open pull requests.

Bosk repository is located on the `GitHub`_ and we use the `Git`_ version control system.
The preferred way to contribute to bosk is to fork the main repository (the *main* branch),
make some changes in your repository and then submit a pull request.


Setting up a Development Environment
------------------------------------

We recommend to create individual virtual python environment using
python `venv`_ or `conda`_. After that, follow the :ref:`installation guide <dev_install>` and install
the bosk package in the develop mode.

Making Changes to the code
--------------------------
For a pull request to be accepted, your changes must meet the following requirements:

1. All changes related to **one feature** must belong to **one branch**.
   Each branch must be self-contained, with a single new feature or bug fix.
   `Create a new feature branch <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_
   by executing::

    git checkout -b new-feature-name

2. All code must follow the standard Python guidelines for code style,
   `PEP8 <https://peps.python.org/pep-0008/>`_. Additionally we try to
   fit `google code style <https://google.github.io/styleguide/pyguide.html>`_.
   To lint our project, we use the `pylint`_.

3. Each function, class, method, and attribute needs to be documented using doc strings.
   Bosk conforms to the
   `google docstring standard <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

4. Code submissions must always include unit tests.
   We are using `pytest <https://docs.pytest.org/>`_.
   All tests must be part of the ``tests`` directory.
   You can run the tests by executing::

    pytest

5. Remember that we use the MIT License and all bosk code must fit it.

Submitting a Pull Request
-------------------------

Make one or several commits in your branch. Each commit should correspond to some particular changes.
Commit messages should be short and informative. Once you have done with the code work,
`create a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.

Building the Documentation
--------------------------

The documentation is located in the ``docs`` folder and is written in
reStructuredText. HTML files of the documentation can be generated using `Sphinx`_.
Before the documentation building, make sure you have installed packages
listed in ``docs/requirements.txt``.

.. note::

    We use Jupyter notebooks as examples in the documentation. To render
    them properly you have to install a `Pandoc <https://pandoc.org/installing.html>`_
    utility manually.

The easiest way to build the documentation is to run::

    cd docs
    make html

Generated files will be located in ``docs/_build/html``.

.. _conda: https://conda.io/miniconda.html
.. _venv: https://docs.python.org/3/library/venv.html
.. _Git: https://git-scm.com/
.. _GitHub: https://github.com/NTAILab/bosk
.. _Sphinx: https://www.sphinx-doc.org/
.. _pylint: https://pylint.org/