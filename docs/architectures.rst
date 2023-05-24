Deep Forest architectures examples
===================================

The basic idea behind the deep forest is stacking of the so-called layers containing decision forest models. Each next layer gets his input as the outcome of the previous one concatenated with the input features.
The modifications that can be applied to the deep forest are various: different weighting schemes, confidence screening of the well-predicted data, multigrained scanning of the images and so on.
Bosk is designed with the respect to these flexibility requirements, and all the described above algorithms can be easily made in the bosk computational pipeline.
Moreover, the new ideas can be designed with the self-made compute blocks, applied and tested within the framework.

References
-----------

To deeply understand the deep forest specifics, consider reading the following papers:

- `Z.-H. Zhou, J. Feng. Deep forest: Towards an alternative to deep neural networks <https://arxiv.org/pdf/1702.08835v1.pdf>`_
- `Wang C., Lu N., Cheng Y., Jiang, B. Deep forest based multivariate classification for diagnostic health monitoring <https://arxiv.org/pdf/1901.01334.pdf>`_
- `Yang F., Xu Q., Li B., Ji Y. Ship detection from thermal remote sensing imagery through region-based deep forest <https://ieeexplore.ieee.org/document/8277182>`_
- `Zheng W., Cao S., Jin X., Mo S., Gao H., Qu Y., Zhu Y. Deep forest with local experts based on elm for pedestrian detection <https://link.springer.com/chapter/10.1007/978-3-030-00767-6_74>`_
- `Utkin L. V., Ryabinin M. A. A Siamese deep forest <https://arxiv.org/abs/1704.08715>`_
- `Pang M., Ting K. M., Zhao P., Zhou Z. H. Improving deep forest by confidence screening <https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm18.pdf>`_
- `Utkin L., Konstantinov A., Meldo A., Ryabinin M., Chukanov V. A Deep Forest Improvement by Using Weighted Schemes <https://ieeexplore.ieee.org/document/8711886>`_

Architectures examples
----------------------

.. toctree::
   :maxdepth: 1

   Basic deep forest <notebooks/basic_forest.ipynb>
   Confidence screening <notebooks/conf_screening.ipynb>
   Multigrained scanning <notebooks/mg_scanning.ipynb>
   Regression <notebooks/regressor.ipynb>
