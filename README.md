# bosk

[[English version]](README_EN.md)
[[Документация]](https://ntailab.github.io/bosk/index.html)

**Bosk** - это фреймворк для создания моделей Глубоких лесов, написанный на языке Python. Совместим с [scikit-learn](https://scikit-learn.org).

## О фреймворке

**Bosk** – открытая библиотека для разработки и применения новых алгоритмов машинного обучения на основе *глубоких лесов*.

*Глубокие леса* – вид моделей машинного обучения,
совмещающих обучение представлениям (автоматическое создание более удобных для обучения векторов признаков) с алгоритмами, не требующими обратного распространения ошибки, такими как случайные леса и градиентный бустинг.
Иными словами, *глубокие леса* являются аналогом глубоких нейронных сетей, причём их построение осуществляется послойно, а блоками могут выступать любые модели машинного обучения.

Модели **Bosk** – *глубокие леса*, представимые в виде графов, узлы которых составляют *вычислительные блоки*,
например: базовые леса-классификаторы, операции конкатенации и т.д.
Каждой модели соответствует два различных вычислительных
графа: первый для стадии обучения модели, а второй – для предсказания.

Без использования фреймворка разработчикам глубоких лесов требуется
вручную реализовывать и поддерживать согласованность участков кода для каждой стадии.
**Bosk** позволяет разрабатывать новые архитектуры глубоких
лесов, не прибегая к написанию рутинного кода отдельно для стадий обучения и предсказания:
модель **Bosk** задаётся **один раз**,
фреймворк **автоматически** определяет какие вычислительные графы
потребуются для каждого из этапов.

**Bosk** содержит полезные инструменты для выполнения и отладки вычислительных графов.
Помимо этого, в **Bosk** представлено большое количество готовых к использованию
стандартных блоков. Фреймворк поддерживает как ручное задание вычислительного графа *глубокого леса*,
так и его автоматическое послойное построение.

### Перечень направлений прикладного использования

Основными направлениями применения **Bosk** являются:

- Разработка принципиально новых архитектур глубокого леса – гибкая платформа позволяет строить сложные схемы, легко расширять набор базовых блоков;
- Построение (в том числе автоматизированное) моделей для решения конкретных прикладных задач машинного обучения – платформа позволяет легко строить глубокие леса по заданному набору данных, сохранять и загружать модели для последующего предсказания;
- Сравнение различных архитектур глубокого леса – модуль тестирования позволяет оценивать точность и производительность различных моделей, избегая повторения одинаковых вычислений, что может использоваться как в исследовательских целях, так и для выбора наиболее подходящего решения прикладной задачи;
- Разработка высокоуровневых алгоритмов для решения новых прикладных задач машинного обучения – расширяемая архитектура не ограничена задачами классификации и регрессии, для новых видов задач легко могут быть адаптированы соответствующие базовые блоки.


## Установка

Для корректной работы пакета необходим Python версии не ниже 3.9.

Bosk использует библиотеку Graphviz для визуализации вычислительных графов.
Для корректной работы данного инструмента требуется установка бинарной зависимости **graphviz**.

Инструкции по установке **graphviz** можно найти по [ссылке](https://graphviz.org/download/).

В Bosk используется библиотека JAX для выполнения вычислений на ГПУ (видеокарте), однако ее установка не тривиальна и не может быть выполнена автоматически при установке пакета.

**Bosk может быть запущен без установки JAX**,
однако блоки, выполняющиеся на ГПУ будут недоступны.
Для установки без JAX достаточно выполнить пункт [установка пакета](#установка-пакета).

### Установка JAX

Официально JAX распространяется только для Linux и Mac OS, поэтому, к сожалению,
пользователям Windows следует использовать [WSL](https://docs.microsoft.com/en-us/windows/wsl/about) для установки JAX и использования bosk с поддержкой вычислений на ГПУ.

**Установка только для ЦПУ**

Если на Вашей системе нет графического процессора, Вы можете установить версию JAX для ЦПУ:

```bash
pip install --upgrade "jax[cpu]==0.4.10"
```

**Установка с поддержкой вычислений на ГПУ**

Если Вы заинтересованы в установке bosk с поддержкой вычислений на ГПУ, пожалуйста, прочитайте наше [руководство по установке](https://ntailab.github.io/bosk/ru_install.html#jax-installation) в документации.

### Установка пакета

Для установки пакета bosk напрямую из GitHub выполните:

```bash
pip install git+https://github.com/NTAILab/bosk.git
```

Также Вы можете вручную склонировать репозиторий и установить bosk:

```bash
git clone https://github.com/NTAILab/bosk.git
cd bosk
pip install -r requirements.txt
python setup.py install
```

## Примеры

### Примеры новых вариантов использования кода

Примеры новых вариантов использования кода включают скрипты и блокноты с кодом и тектовыми пояснениями и диаграммами, располагаются в [директории examples](examples/).
Также вы можете обратиться к нашей [документации](https://ntailab.github.io/bosk/ru_examples.html), в которую включены данные примеры.

### Простой пример

Для краткого обзора возможностей фреймворка рассмотрим однослойный Глубокий лес, содержащий в себе два вида лесов: случайный лес (Random Forest) и модель сверхслучайных деревьев (Extremely Randomized Trees).
Вектора вероятностей, предсказываемые каждой из моделей, конкатенируются с вектором входных признаков и передаются в финальный лес.

```python
# создание построителя конвейера
b = FunctionalPipelineBuilder()
# блоки для маршрутизации входных данных:
# `x` для вектора факторов и `y`
# для откликов
x = b.Input('x')()
y = b.TargetInput('y')()
# блоки моделей лесов
random_forest = b.RFC(max_depth=5)
extra_trees = b.ETC(n_estimators=200)
# блок конкатенации
cat = b.Concat(['x', 'rf', 'et'])
# слой, конкатенирующий выходные векторы лесов
# и вектор входных признаков
layer_1 = cat(x=x, rf=rf(X=x, y=y), et=extra_trees(X=x, y=y))
# лес для осуществления итогового предсказания
final_extra_trees = b.ETC()
# выход конвейера
b.Output('proba')(final_extra_trees(X=layer_1, y=y))
# создание конвейера
pipeline = b.build()

# scikit-learn обертка для конвейера
model = BoskPipelineClassifier(pipeline, executor_cls=RecursiveExecutor)
# обучаем модель
model.fit(X_train, y_train)
# осуществляем предсказание
test_preds = model.predict(X_test)
```

Для того, чтобы увидеть больше примеров, Вы можете обратиться к нашей [документации](https://ntailab.github.io/bosk/ru_examples.html). Также Вы можете найти Jupyter блокноты с примерами в  [директории с примерами](examples/).

## Документация

Больше информации о bosk Вы можете найти в нашей [документации](https://ntailab.github.io/bosk/index.html).

## Как стать участником проекта

Мы всегда рады вкладу сообщества в наш проект. Пожалуйста, прочитайте [инструкцию](https://ntailab.github.io/bosk/ru_contribution.html), чтобы узнать, как стать участником bosk.
