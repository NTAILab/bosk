from bosk.executor.recursive import RecursiveExecutor
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.executor.sklearn_interface import BoskPipelineClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def main():
    all_X, all_y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, random_state=12345)
    # make a pipeline
    b = FunctionalPipelineBuilder()
    # we denote here block wrappers with underscore at the end
    x_ = b.Input('X')()
    y_ = b.TargetInput('y')()
    rf_ = b.RFC(random_state=123)(X=x_, y=y_)
    et_ = b.ETC(random_state=123)(X=x_, y=y_)
    concat_ = b.Concat(['X', 'rf', 'et'], axis=1)(X=x_, rf=rf_, et=et_)
    rf2_ = b.RFC(random_state=456)(X=concat_, y=y_)
    et2_ = b.ETC(random_state=456)(X=concat_, y=y_)
    concat2_ = b.Concat(['X', 'rf2', 'et2'], axis=1)(X=x_, rf2=rf2_, et2=et2_)
    proba_ = b.ETC(random_state=12345)(X=concat2_, y=y_)
    b.Output('proba')(proba_)  # create and label output
    pipeline = b.build()
    # make a scikit-learn model
    model = BoskPipelineClassifier(pipeline, executor_cls=RecursiveExecutor)
    # fit the model
    model.fit(X_train, y_train)
    # predict with the model
    test_preds = model.predict(X_test)
    print('Test f1 score:', f1_score(y_test, test_preds, average='macro'))


if __name__ == '__main__':
    main()
