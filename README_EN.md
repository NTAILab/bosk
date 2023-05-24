# bosk

[[Russian version]](README.md)
[[Documentation]](https://ntailab.github.io/bosk/en_index.html)

Bosk is a Python framework for Deep Forest construction. Compatible with [scikit-learn](https://scikit-learn.org).

## About framework

**Bosk** is an open source library for developing and applying new machine learning algorithms based on *deep forests*.

*Deep forests* - are machine learning models,
combining representation learning (automatic generation of more appropriate feature vectors) with algorithms that do not require backpropagation, such as random forests and gradient boosting.
In other words, *deep forests* are analogous to deep neural networks, but their construction is carried out in block-by-block fashion, and any machine learning models can act as blocks.

**Bosk** models are *deep forests* represented as graphs whose nodes are *computing blocks*,
for example: base classifier forests, concatenation operations, etc.
Each model corresponds to two different computational
graphs: the first one is for the training stage of the model, and the second one is for the prediction stage.

Without using a framework, deep forest developers need to
manually implement and maintain consistency of code sections for each stage.
**Bosk** allows you to develop new architectures of deep
forests without resorting to writing routine code separately for the training and prediction stages:
the **Bosk** model is defined **once**,
the framework **automatically** infers corresponding computational graphs that are required for each step.

**Bosk** contains useful tools for executing and debugging computational graphs.
In addition, **Bosk** provides a large number of ready-to-use
standard blocks. The framework supports both manual construction of a *deep forest* structure,
and its automatic layerwise construction.

### List of areas of application

The main areas of application for **Bosk** are:

- Development of fundamentally new deep forest architectures - a flexible platform allows you to build complex schemes, easily expand the set of base blocks;
- Building (including automated) models for solving specific applied problems of machine learning - the platform allows you to easily build deep forests from a given data set, save and load models for subsequent prediction;
- Comparison of different deep forest architectures - the testing module allows you to evaluate the accuracy and performance of different models, avoiding the repetition of the same calculations, which can be used both for research purposes and for choosing the most appropriate solution to an applied problem;
- Development of high-level algorithms for solving new applied problems of machine learning - an extensible architecture is not limited to classification and regression problems, the corresponding basic blocks can be easily adapted for new types of problems.

## Installation

For the package to work correctly, Python 3.9+ is required.

Bosk uses the JAX library to perform computations on the GPU (video card), however its installation is not trivial and cannot be done automatically when the package is installed.

**Bosk can be run without JAX installed**,
however, blocks running on the GPU will not be available.
To install without JAX, just follow [package installation](#package-installation).

### JAX installation

Officially JAX is distributed only for Linux and Mac OS, so, unfortunatelly, Windows users should use [WSL](https://docs.microsoft.com/en-us/windows/wsl/about) to install JAX and use bosk with the GPU.

**CPU-only system**

If there is no GPU available, install the CPU JAX version

```bash
pip install --upgrade "jax[cpu]"
```

**GPU system**

If you are interested in GPU installation, please, visit our [install guide](https://ntailab.github.io/bosk/install.html#jax-installation) in the documentation.

### Package Installation

To install the bosk package directly from GitHub run:

```bash
pip install git+https://github.com/NTAILab/bosk.git
```

If you are interested in manual or developement-mode installation, please, visit our [install guide](https://ntailab.github.io/bosk/install.html#package-installation) in the documentation.

## Examples

### Examples of new code use cases

Examples of new code use cases include scripts and notebooks with code and text explanations and diagrams, located in the [examples directory](examples/).
You can also refer to our [documentation](https://ntailab.github.io/bosk/en_examples.html) which includes these examples.

### Simple Example

For the quick overview let's make a Deep Forest with one layer, consisting of two forests (Random Forest and Extremely Randomized Trees), which output probabilities are concatenated with input feature vector and passed to the final forest. The following code could be used:

```python
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
```

For more examples visit our [documentation](https://ntailab.github.io/bosk/examples.html). Also, you can find example scripts and Jupyter notebooks in the [examples folder](examples/).

### Documentation

More information about bosk can be found in our [documentation](https://ntailab.github.io/bosk/en_index.html).

### Contribution

We are glad to see new contributors. Please, look at the [contribution guide](https://ntailab.github.io/bosk/contribution.html) to get started and read the guidelines.
