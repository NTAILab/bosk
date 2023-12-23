Примеры использования bosk
==========================

На этой странице вы можете найти различные примеры
использования bosk с комментариями и иллюстрациями. 
Все перечисленные ниже файлы (и не только) можно найти
в нашем репозитории `GitHub <https://github.com/NTAILab/bosk>`_ 
в папке ``examples``.

Ниже Вы можете увидеть краткий самодостаточный пример использования bosk.
Его можно скопировать на локальную машину и запустить после :doc:`установки пакета <ru_install>`.

.. code-block:: python
   
   from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
   from bosk.executor.topological import TopologicalExecutor
   from bosk.stages import Stage
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import roc_auc_score
   # подключение блоков для обеспечения подсказок IDE
   from bosk.block.zoo.input_plugs import Input, TargetInput
   from bosk.block.zoo.data_conversion import Concat
   from bosk.block.zoo.models.classification import RFC, ETC
   from bosk.block.zoo.output_plugs import Output

   n_estimators = 20
   random_state = 42

   # создание объекта, конструирующего конвейер
   with FunctionalPipelineBuilder() as b:
       # создание оберток вычислительных блоков
       # и их соединение в функциональном стиле
       X, y = Input()(), TargetInput()()
       rf_1 = RFC(n_estimators=n_estimators)(X=X, y=y)
       et_1 = ETC(n_estimators=n_estimators)(X=X, y=y)
       concat_1 = Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
       rf_2 = RFC(n_estimators=n_estimators)(X=concat_1, y=y)
       et_2 = ETC(n_estimators=n_estimators)(X=concat_1, y=y)
       stack = b.Stack(['rf_2', 'et_2'], axis=1)(rf_2=rf_2, et_2=et_2)
       average = b.Average(axis=1)(X=stack)
       argmax = b.Argmax(axis=1)(X=average)
       rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
       roc_auc = b.RocAuc()(gt_y=y, pred_probas=average)
   # после задания структуры вычислительного графа
   # мы создаем конвейер
   pipeline = b.build(
      {'X': X, 'y': y},
      {'labels': argmax, 'probas': average, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
   )
   # у нашей модели можно задать случайное зерно
   pipeline.set_random_state(random_state)

   # давайте сгенерируем данные
   # для обучения и тестирования конвейера
   all_X, all_y = make_moons(noise=1, random_state=random_state)
   train_X, test_X, train_y, test_y = train_test_split(
      all_X, all_y, test_size=0.2, random_state=random_state)
   # создание исполнителей
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
   # выполнение нашего конвейера и расчет метрик
   fit_result = fit_executor({'X': train_X, 'y': train_y}).numpy()
   print(
      "Train ROC-AUC calculated by fit executor:",
      fit_result['roc-auc']
   )
   test_result = transform_executor({'X': test_X}).numpy()
   print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'][:, 1]))

.. toctree::
   :maxdepth: 2
   :caption: Примеры использования bosk:

   ru_basic_examples
   ru_architectures
   ru_advanced_api
