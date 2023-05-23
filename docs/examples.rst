Bosk usage examples
===================

On this page you can find various examples of the bosk usage with the comments and illustrations.
All listed below files and even more can be found in our `GitHub <https://github.com/NTAILab/bosk>`_
repository in the ``examples`` folder.

The quick self-sufficient example of the bosk usage is available below. You can copy it to the local
machine and run it after the bosk :doc:`installation <install>`.

.. code-block:: python
   from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
   from bosk.executor.topological import TopologicalExecutor
   from bosk.stages import Stage
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import roc_auc_score

   n_estimators = 20
   random_state = 42

   # firstly we must obtain the functional builder object
   b = FunctionalPipelineBuilder()
   # we get blocks wrappers and connect with each other
   X, y = b.Input()(), b.TargetInput()()
   rf_1 = b.RFC(n_estimators=n_estimators)(X=X, y=y)
   et_1 = b.ETC(n_estimators=n_estimators)(X=X, y=y)
   concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
   rf_2 = b.RFC(n_estimators=n_estimators)(X=concat_1, y=y)
   et_2 = b.ETC(n_estimators=n_estimators)(X=concat_1, y=y)
   stack = b.Stack(['rf_2', 'et_2'], axis=1)(rf_2=rf_2, et_2=et_2)
   average = b.Average(axis=1)(X=stack)
   argmax = b.Argmax(axis=1)(X=average)
   rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
   roc_auc = b.RocAuc()(gt_y=y, pred_probas=average)
   # after defining the graph structure we obtain
   # the pipeline object from the builder
   pipeline = b.build(
      {'X': X, 'y': y},
      {'labels': argmax, 'probas': average, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
   )
   # we can set a random state for the pipeline
   pipeline.set_random_state(random_state)

   # let's get some data
   # than train and test our pipeline
   all_X, all_y = make_moons(noise=1, random_state=random_state)
   train_X, test_X, train_y, test_y = train_test_split(
      all_X, all_y, test_size=0.2, random_state=random_state)
   # creating executors
   fit_executor = TopologicalExecutor(
      pipeline,
      stage=Stage.FIT,
      inputs=['X', 'y'],
      outputs=['probas', 'rf_1_roc-auc', 'roc-auc']
   )
   transform_executor = TopologicalExecutor(
      pipeline,
      stage=Stage.TRANSFORM,
      inputs=['X'],
      outputs=['probas', 'labels']
   )
   # executing our pipeline and obtaining metrics
   fit_result = fit_executor({'X': train_X, 'y': train_y}).numpy()
   print(
      "Train ROC-AUC calculated by fit executor:",
      fit_result['roc-auc']
   )
   test_result = transform_executor({'X': test_X}).numpy()
   print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'][:, 1]))


.. toctree::
   :maxdepth: 2
   :caption: Bosk usage examples:

   basic_examples
   architectures
   advanced_api
