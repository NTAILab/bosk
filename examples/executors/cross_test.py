"""Example of testing different executors with various deep forest architectures.

"""
from collections import defaultdict
from time import time

import numpy as np

from sklearn.datasets import make_moons, load_digits, load_iris
from sklearn.model_selection import train_test_split

from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.topological import TopologicalExecutor
from bosk.painter.topological import TopologicalPainter
from bosk.data import CPUData

from examples.deep_forests.casual.source import make_deep_forest, make_deep_forest_functional
from examples.deep_forests.cs.simple import make_deep_forest_functional_confidence_screening
from examples.deep_forests.mg_scanning.mg_scanning import (make_deep_forest_functional_multi_grained_scanning_1d, 
                                                           make_deep_forest_functional_multi_grained_scanning_2d)
from examples.deep_forests.weighted_cs.simple import make_deep_forest_weighted_confidence_screening


def get_moons_dataset():
    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    return CPUData(train_X), CPUData(test_X), CPUData(train_y), CPUData(test_y)


def get_digits_dataset():
    digits = load_digits()
    all_X = digits.data
    all_y = digits.target
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    return CPUData(train_X), CPUData(test_X), CPUData(train_y), CPUData(test_y)


def get_iris_dataset():
    iris = load_iris()
    all_X = iris.data
    all_y = iris.target
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    return CPUData(train_X), CPUData(test_X), CPUData(train_y),CPUData(test_y)


class DeepForestWrapper():
    def __init__(self, df_factory, dataset_factory):
        self.df_factory = df_factory
        self.dataset_factory = dataset_factory
        self.is_fitted = False
    
    def train(self, exec_cls, **exec_kw):
        train_X, self.test_X, train_y, _ = self.dataset_factory()
        self.fit_exec, self.tf_exec = self.df_factory(exec_cls, **exec_kw)
        fit_res = self.fit_exec({'X': train_X, 'y': train_y})
        self.is_fitted = True
        return fit_res
    
    def predict_test(self):
        assert(self.is_fitted == True)
        return self.tf_exec({'X': self.test_X})

    def paint(self, name):
        if isinstance(self.fit_exec, TopologicalExecutor) and isinstance(self.tf_exec, TopologicalExecutor):
            TopologicalPainter().from_executor(self.fit_exec).render('fit_' + name)
            TopologicalPainter().from_executor(self.tf_exec).render('tf_' + name)


def main():
    tol = 10 ** -6
    wrappers_dict = {
        'casual forest': DeepForestWrapper(make_deep_forest, get_moons_dataset),
        'casual func api forest': DeepForestWrapper(make_deep_forest_functional, get_moons_dataset),
        'CS forest': DeepForestWrapper(make_deep_forest_functional_confidence_screening, get_moons_dataset),
        'WCS forest': DeepForestWrapper(make_deep_forest_weighted_confidence_screening, get_moons_dataset),
        'MG 1d forest': DeepForestWrapper(make_deep_forest_functional_multi_grained_scanning_1d, get_iris_dataset),
        'MG 2d forest': DeepForestWrapper(make_deep_forest_functional_multi_grained_scanning_2d, get_digits_dataset),
    }
    executors_dict = {
        'recursive': (RecursiveExecutor, {}),
        'topological': (TopologicalExecutor, {}),
    }
    test_result_dict = dict()
    trouble_scores = []

    def print_scores(dictionary, postfix):
        for key, val in dictionary.items():
                if isinstance(val.data, float) or isinstance(val.data, int):
                    print(f'\t\t{key}:', val)
                else:
                    print(f'\t\t{key}: ... (mean {np.round(np.mean(val.data), 4)})', )
                score_dict[key + f' ({postfix})'].append(val)
    for df_name, df_wrapper in wrappers_dict.items():
        score_dict = defaultdict(list)
        for exec_name, (exec_cls, exec_kw) in executors_dict.items():
            print(f'--- {df_name} training with {exec_name} executor ---')
            time_stamp = time()
            train_res = df_wrapper.train(exec_cls, **exec_kw)
            print(f'\tTaken {round(time() - time_stamp, 4)} s.')
            print('\tTraining metrics:')
            print_scores(train_res, 'fit')
            print(f'--- {df_name} test with {exec_name} executor ---')
            time_stamp = time()
            test_res = df_wrapper.predict_test()
            print(f'\tTaken {round(time() - time_stamp, 4)} s.')
            print('\tTest metrics:')
            print_scores(test_res, 'test')
            if exec_cls is TopologicalExecutor:
                print(f'--- Drawing {df_name} with {exec_name} executor ---')
                time_stamp = time()
                df_wrapper.paint(f'{df_name}_{exec_name}.png')
                print(f'\tTaken {round(time() - time_stamp, 4)} s.')
            test_res = True
            for key, val in score_dict.items():
                res = all([np.sum(np.abs(score.data - val[0].data)) < tol for score in val])
                if not res:
                    trouble_scores.append(f'{df_name}, {key}')
                test_res *= res
            print(f'{df_name} test is', 'PASSED' if test_res else 'FAILED')
            test_result_dict[df_name] = test_res
    print('--- Summary ---')
    if all(test_result_dict.values()):
        print('Test is PASSED')
    else:
        print('Test is FAILED')
        print('The next scores were different:')
        for score in trouble_scores:
            print(score)


if __name__ == "__main__":
    main()
