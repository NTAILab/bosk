Getting started
===============

After :doc:`installation <install>` of the bosk framework, lets consider
the basic primitives.


Basic primitives
----------------

- :ref:`block` – basic computational node, which has multiple inputs and outputs (slots) \
  and can be trained with `fit`, and then can transform inputs with \
  `transform` to obtain outputs;
- :ref:`Pipeline <pipeline>` – Deep Forest itself, represented as a graph, defined on block slots. \
  It represents both `fit` and `transform` stage;
- :ref:`Executor <executor>` – subject, that executes the pipeline given some input data to obtain output.
  Can be used to train the forest, or to evaluate predictions.


.. _block:

Block
~~~~~

Any block can be fitted on some data and then used to transform the data.

Blocks may have different number of inputs and outputs.
Information about inputs, outputs and other block properties
is stored in :py:class:`bosk.block.meta.BlockMeta`.
The meta information may be shared between different instances of the same block class.
Each input is described by a :py:class:`bosk.block.meta.InputSlotMeta` and
each output is described by a :py:class:`bosk.block.meta.OutputSlotMeta`.

To connect blocks to each other, every block has `slots`:
:py:class:`BlockInputSlot` and :py:class:`BlockOutputSlot`,
which correspond to meta information. But the slots are unique for each block (i.e. block class instance).

Every block class should be derived from :py:class:`bosk.block.base.BaseBlock`,
define meta information and implement the :py:meth:`fit` and :py:meth:`transform` methods.

*Note*, that block prediction is always performed with the :py:meth:`transform` method, not with "predict"
or something else.

List of available blocks can be found in the :py:mod:`bosk.block.zoo`.

.. _pipeline:

Pipeline
~~~~~~~~

Pipelines are represented as sets of nodes (blocks) and connections between them.

There are tree ways to define a pipeline:

1. Manually, by defining a list of nodes and connections.
2. Using the :ref:`builder in functional style <functional-pipeline-builder>`.
3. :ref:`Automatically <automatic-pipeline>`, building Deep Forest layerwise with predefined layer structure.

Here we will consider only the second and the third options, since they are more convinient.


.. _functional-pipeline-builder:

Functional-style Pipeline Definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deep Forest can be built manually by using
`FunctionalPipelineBuilder`.
It allows to create arbitrary complex pipeline
by combining block placeholders (wrappers).

For example, to create Deep Forest with two layers, the following code can be used:

.. code-block:: python

   from bosk.executor import TopologicalExecutor
   from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
   from bosk.executor.sklearn_interface import BoskPipelineClassifier

   # make a pipeline
   b = FunctionalPipelineBuilder()
   # placeholders for input features `X` and target variable `y`
   x_ = b.Input('X')()
   y_ = b.TargetInput('y')()
   # make random forests for the first layer
   rf_ = b.RFC(random_state=123)(X=x_, y=y_)
   et_ = b.ETC(random_state=123)(X=x_, y=y_)
   # concatenate predictions of forests with `X`
   concat_ = b.Concat(['X', 'rf', 'et'], axis=1)(X=x_, rf=rf_, et=et_)
   # make the second layer
   rf2_ = b.RFC(random_state=456)(X=concat_, y=y_)
   et2_ = b.ETC(random_state=456)(X=concat_, y=y_)
   concat2_ = b.Concat(['X', 'rf2', 'et2'], axis=1)(X=x_, rf2=rf2_, et2=et2_)
   # make the final model
   proba_ = b.ETC(random_state=12345)(X=concat2_, y=y_)
   # use its predictions as a pipeline output
   b.Output('proba')(proba_)
   # build pipeline
   pipeline = b.build()
   # wrap pipeline into a scikit-learn model
   model = BoskPipelineClassifier(pipeline, executor_cls=RecursiveExecutor)
   # fit the model
   model.fit(X_train, y_train)
   # predict with the model
   test_preds = model.predict(X_test)



.. _automatic-pipeline:

Automatic layerwise Deep Forest construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bosk framework also allows to build Deep Forests fully automatically
given only data and set of optional parameters.

For example, the following code creates Deep Forest with at most three layers
with 2 block of each type in layer, using 2-fold cross-validation:

.. code-block:: python

   from bosk.auto.deep_forest import ClassicalDeepForestConstructor
   from bosk.executor import TopologicalExecutor
   from bosk.executor.sklearn_interface import BoskPipelineClassifier

    constructor = ClassicalDeepForestConstructor(
        TopologicalExecutor,
        rf_params=dict(),
        max_iter=3,
        layer_width=2,
        cv=2,
        random_state=12345,
    )
    # construct Deep Forest automatically based on data
    pipeline = constructor.construct(X_train, y_train)
    # make a scikit-learn model
    model = BoskPipelineClassifier(pipeline, executor_cls=TopologicalExecutor)
    model._classifier_init(y.data)
    test_preds = model.predict(X_test)

.. _executor:

Executor
~~~~~~~~

Executor is the subject that fits the pipelines and evaluate its outputs.

A pipeline executor acts like a function and can be applied
to a dictionary of input values.

The output of the executor is a special dictionary of output values,
which contain wrapped data (:py:class:`bosk.data.BaseData`).
In order to obtain NumPy arrays as output, the `.numpy()` method should be called
on the result.

Example of usage:

.. code-block:: python

    pipeline = make_pipeline()  # make a pipeline somehow
    fitter = TopologicalExecutor(pipeline, stage=Stage.FIT)
    fitter({'X': X_train, 'y': y_train})  # fit on dictionary of input numpy arrays
    predictor = TopologicalExecutor(pipeline, stage=Stage.TRANSFORM)
    predictions = predictor({'X': X_test}).numpy()  # result: dictionary of output numpy arrays

Executors and more detailed description are listed in :py:mod:`bosk.executor`.
