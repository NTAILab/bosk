Welcome to bosk's documentation!
================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation guide <install>
   getting_started
   Examples <user_guide>


Bosk is a framework for Deep Forest construction.

Following common principle of deep neural network frameworks, we consider models as
general computational graphs with some additional functionality,
in contrast to defining strictly layerwise structure.
In bosk a Deep Forest structure corresponds to two separate computational graphs:
one for fitting (training) and one for transforming (predicting).

This framework helps to construct new Deep Forests avoiding writing
error prone routine code, and provides tools for pipeline execution and debugging,
as well as a wide set of ready to use building blocks.
It supports both fully manual pipeline and automatical layerwise Deep Forest building.

Quick example
~~~~~~~~~~~~~

For example, to define Deep Forest with one layer, consisting of two forests
(Random Forest and Extremely Randomized Trees), which output probabilities
are concatenated with input feature vector and passed to the final
forest, the following code could be used:

.. code-block:: python

   # make a pipeline
   b = FunctionalPipelineBuilder()
   # placeholders for input features `x` and target variable `y`
   x = b.Input('x')()
   y = b.TargetInput('y')()
   # random forests
   random_forest = b.RFC(max_depth=5)
   extra_trees = b.ETC(n_estimators=200)
   # concatenation
   cat = b.Concat(['x', 'rf', 'et'])
   # layer that concatenates random forests outputs
   layer_1 = cat(x=x, rf=rf(X=x, y=y), et=extra_trees(X=x, y=y))
   # forest for the final prediction
   final_extra_trees = b.ETC()
   # pipeline output
   b.Output('proba')(final_extra_trees(X=layer_1, y=y))
   # build pipeline
   pipeline = b.build()

   # wrap pipeline into a scikit-learn model
   model = BoskPipelineClassifier(pipeline, executor_cls=RecursiveExecutor)
   # fit the model
   model.fit(X_train, y_train)
   # predict with the model
   test_preds = model.predict(X_test)

For more examples look at :doc:`getting_started`

