# bosk

[[Russian version]](README.md)

Bosk is a Python framework for Deep Forest construction. Compatible with [scikit-learn](https://scikit-learn.org).

## About bosk

Following common principle of deep neural network frameworks, we consider models as general computational graphs with some additional functionality, in contrast to defining strictly layerwise structure. In bosk a Deep Forest structure corresponds to two separate computational graphs: one for fitting (training) and one for transforming (predicting).

This framework helps to construct new Deep Forests avoiding writing error prone routine code, and provides tools for pipeline execution and debugging, as well as a wide set of ready to use building blocks. It supports both fully manual pipeline and automatical layerwise Deep Forest building.

## Installation

Python 3.9+ is required.

### JAX installation

Bosk uses JAX for GPU computations, but JAX installation is not trivial.
Officially JAX is distributed only for Linux and Mac OS, so, unfortunatelly, Windows users should use [WSL](https://docs.microsoft.com/en-us/windows/wsl/about) to install JAX and use bosk with the GPU.

**CPU-only system**

If there is no GPU available, install the CPU JAX version

    pip install --upgrade "jax[cpu]"

**GPU system**

If you are interested in GPU installation, please, visit our [install guide](https://ntailab.github.io/bosk/install.html#jax-installation) in the documentation.

### Package Installation

To install the bosk package directly from GitHub run:

    pip install git+ssh://git@github.com:NTAILab/bosk.git

If you are interested in manual or developement-mode installation, please, visit our [install guide](https://ntailab.github.io/bosk/install.html#package-installation) in the documentation.

### Examples

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
