Приступая к работе
==================

После `установки <ru_install>`_ фреймворка bosk, предлагаем
обсудить базовые концепции.

Базовые примитивы
-----------------

- :ref:`Блок <block>` – базовый вычислительный узел, у которого определены входы и выходы (слоты). \
  Может быть обучен методом `fit`, после чего возможно использование метода `transform` \
  для преобразования входных данных и получения выходов;
- :ref:`Конвейер <pipeline>` – модель Глубокого леса, представленная в форме вычислительного \
  графа, определенного посредством указания связей между слотами разных блоков. \
  Конвейер реализует как стадию `обучения <fit>`_ (fit), так и `преобразования <transform>`_;
- :ref:`Исполнитель <executor>` – менеджер, который обслуживает конвейер и запускает вычислительный граф \
  для получения значений на выходах исходя из поданных на вход данных. Может быть использован для \
  обучения Глубокого леса или для вычисления предсказаний на обученной модели.

.. _block:

Блок
~~~~

Каждый блок может быть обучен на некотором множестве данных и после чего быть
использован для преобразования новых данных.

У блока может быть определено несколько входов и выходов.
Информация о них, а также прочие параметры блока находятся в
классе :py:class:`bosk.block.meta.BlockMeta`.
Эта метаинформация может быть общей у разных экземпляров одного типа блока.
Каждый вход описывается классом :py:class:`bosk.block.meta.InputSlotMeta`,
а каждый выход - :py:class:`bosk.block.meta.OutputSlotMeta`.

Для соединения блоков между собой у них есть слоты:
:py:class:`BlockInputSlot` и :py:class:`BlockOutputSlot`,
которые соответствуют описанной выше метаинформации.
Отличие в том, что слоты уникальны для каждого экземпляра блока.

.. note::
  Предсказания блока всегда вычисляются с помощью метода :py:meth:`transform`,
  а не "predict" или каким-либо еще.

Все доступные блоки содержатся в подмодуле :py:mod:`bosk.block.zoo`.

.. _pipeline:

Конвейер
~~~~~~~~

Конвейер определяется набором узлов (блоков) и связями между ними.

Есть три способа создать конвейер:

1. Вручную, задав список узлов и связей.
2. С помощью :ref:`построения в функциональном стиле <functional-pipeline-builder>`.
3. :ref:`Автоматически <automatic-pipeline>`, строя Глубокий лес слой за слоем
  с заранее определенной структурой последнего.

Далее мы опишем лишь последние два способа, так как они более удобны и предпочтительны.

.. _functional-pipeline-builder:

Создание конвейера в функциональном стиле
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Глубокий лес может быть создан в ручном режиме с помощью `FunctionalPipelineBuilder`.
Он позволяет создавать конвейеры любой сложности с помощью комбинаций оберток блоков.

Например, для создания Глубокого леса с двумя слоями можно использовать следующий код:

.. code-block:: python

   from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
   from bosk.executor.sklearn_interface import BoskPipelineClassifier
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split

   # создание построителя конвейера
   b = FunctionalPipelineBuilder()
   # блоки для маршрутизации входных данных:
   # `x` для вектора факторов и `y`
   # для откликов
   x_ = b.Input('X')()
   y_ = b.TargetInput('y')()
   # создание случайных лесов для первого слоя
   rf_ = b.RFC(random_state=123)(X=x_, y=y_)
   et_ = b.ETC(random_state=123)(X=x_, y=y_)
   # конкатенация предсказаний лесов с `X`
   concat_ = b.Concat(['X', 'rf', 'et'], axis=1)(X=x_, rf=rf_, et=et_)
   # создание второго слоя
   rf2_ = b.RFC(random_state=456)(X=concat_, y=y_)
   et2_ = b.ETC(random_state=456)(X=concat_, y=y_)
   concat2_ = b.Concat(['X', 'rf2', 'et2'], axis=1)(X=x_, rf2=rf2_, et2=et2_)
   # создание финальной модели
   proba_ = b.ETC(random_state=12345)(X=concat2_, y=y_)
   # используем ее вывод в качестве выхода конвейера
   b.Output('proba')(proba_)
   # создание конвейера
   pipeline = b.build()
   # сделаем модель scikit-learn из нашего конвейера
   model = BoskPipelineClassifier(pipeline)

   # для примера, сгенерируем набор обучающих и тестовых данных:
   all_X, all_y = make_moons(noise=0.1)
   X_train, X_test, y_train, y_test = train_test_split(
      all_X, all_y, test_size=0.2)

   # обучение модели
   model.fit(X_train, y_train)
   # использование модели для вычисления предсказаний
   test_preds = model.predict(X_test)

.. _automatic-pipeline:

Автоматическое создание конвейера слой за слоем
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Наш фреймворк также позволяет автоматизировать процесс создания Глубоких лесов,
требуется лишь задать данные для обучения и набор параметров.

Например, следующий код создает Глубокий лес с максимальной глубиной в три слоя
с блоками двух типов на каждом. Используется кросс-валидация с двумя фолдами.

.. code-block:: python

   from bosk.auto.deep_forest import ClassicalDeepForestConstructor
   from bosk.executor import TopologicalExecutor
   from bosk.executor.sklearn_interface import BoskPipelineClassifier
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
 
   constructor = ClassicalDeepForestConstructor(
       TopologicalExecutor,
       rf_params=dict(),
       max_iter=3,
       layer_width=2,
       cv=2,
       random_state=12345,
   )
   # для примера, сгенерируем набор обучающих и тестовых данных:
   all_X, all_y = make_moons(noise=0.1)
   X_train, X_test, y_train, y_test = train_test_split(
      all_X, all_y, test_size=0.2)
   # создание глубокого леса автоматически основываясь на данных
   pipeline = constructor.construct(X_train, y_train)
   # сделаем модель scikit-learn
   model = BoskPipelineClassifier(pipeline, executor_cls=TopologicalExecutor)
   model._classifier_init(y_train)
   test_preds = model.predict(X_test)

.. _executor:

Исполнитель
~~~~~~~~~~~

Исполнитель - это менеджер, который может обучить конвейер и вычислить его выходы.

Исполнитель конвейера ведет себя как функция и может быть применен к словарю
с входными данными.

Выход исполнителя - это специальный словарь, содержащий данные, обернутые в 
:py:class:`bosk.data.BaseData`. Для того, чтобы конвертировать их в
numpy массивы, Вы должны вызвать `.numpy()` метод.

Пример использования:

.. code-block:: python

  pipeline = make_pipeline()  # создание конвейера каким-либо образом
  fitter = TopologicalExecutor(pipeline, stage=Stage.FIT)
  fitter({'X': X_train, 'y': y_train})  # обучение на словаре, состоящем из входных numpy массивов
  predictor = TopologicalExecutor(pipeline, stage=Stage.TRANSFORM)
  predictions = predictor({'X': X_test}).numpy()  # результат: словарь выходных numpy массивов

Список исполнителей и описание каждого из них находятся в подмодуле :py:mod:`bosk.executor`.

Больше примеров может быть найдено на странице :doc:`ru_examples`.
