Примеры архитектур модели Глубокого леса
========================================

Основная идея модели Глубокого леса заключается в
композиции так называемых слоев, содержащих модели 
Лесов решений. Каждый следующий слой получает на вход
результат предыдущего, объединенный с входным вектором признаков.
Модификации, которые можно применить к глубокому лесу, разнообразны:
разные схемы взвешивания, цензурирование хорошо предсказанных данных (Confidence screening),
многозернистое сканирование (Multigrained scanning) изображений и так далее.
Bosk разработан с учетом этих требований к гибкости, и все описанные
выше алгоритмы могут быть легко реализованы в вычислительном конвейере bosk.
Кроме того, новые идеи могут быть разработаны с помощью введения новых
вычислительных блоков, применены и протестированы с помощью bosk.

Ссылки на исследования
----------------------

Чтобы лучше понять специфику моделей Глубокого леса, 
Вы можете ознакомиться со следующими статьями:

- `Z.-H. Zhou, J. Feng. Deep forest: Towards an alternative to deep neural networks <https://arxiv.org/pdf/1702.08835v1.pdf>`_
- `Wang C., Lu N., Cheng Y., Jiang, B. Deep forest based multivariate classification for diagnostic health monitoring <https://arxiv.org/pdf/1901.01334.pdf>`_
- `Yang F., Xu Q., Li B., Ji Y. Ship detection from thermal remote sensing imagery through region-based deep forest <https://ieeexplore.ieee.org/document/8277182>`_
- `Zheng W., Cao S., Jin X., Mo S., Gao H., Qu Y., Zhu Y. Deep forest with local experts based on elm for pedestrian detection <https://link.springer.com/chapter/10.1007/978-3-030-00767-6_74>`_
- `Utkin L. V., Ryabinin M. A. A Siamese deep forest <https://arxiv.org/abs/1704.08715>`_
- `Pang M., Ting K. M., Zhao P., Zhou Z. H. Improving deep forest by confidence screening <https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm18.pdf>`_
- `Utkin L., Konstantinov A., Meldo A., Ryabinin M., Chukanov V. A Deep Forest Improvement by Using Weighted Schemes <https://ieeexplore.ieee.org/document/8711886>`_

Примеры различных архитектур
----------------------------

.. toctree::
   :maxdepth: 1

   Базовый Глубокий лес <notebooks/ru_basic_forest.ipynb>
   Механизм confidence screening <notebooks/ru_conf_screening.ipynb>
   Многозернистое сканирование <notebooks/ru_mg_scanning.ipynb>
   notebooks/ru_regressor.ipynb
