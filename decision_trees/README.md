# Decision Trees

These examples demonstrate decision tree classification using a synthetic diabetes risk assessment scenario. The goal is
to predict whether a patient is at high risk of diabetes based on a small set of basic health indicators.

The algorithm is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, focusing on entropy and information gain
- A [reference implementation](sklearn_implementation.py) using `scikit-learn` for comparison

---

## Model

The model is a simple recursive decision tree classifier that splits the data based on feature values to minimize
entropy and maximize information gain.

## Training Data

Below is a preview of the synthetic training dataset (generated via AI):

| age |  bmi | fasting_glucose | family_history | physical_activity_hours_per_week | high_diabetes_risk |
|----:|-----:|----------------:|---------------:|---------------------------------:|-------------------:|
|  68 | 34.2 |             135 |              1 |                              1.0 |                  1 |
|  62 | 32.5 |             117 |              0 |                              0.5 |                  1 |                  
|  22 | 21.5 |              88 |              0 |                              6.0 |                  0 |
| ... |      |                 |                |                                  |                    |

Each sample represents a patient with the following features:

- `age` – age in years
- `bmi` – body mass index
- `fasting_glucose` – fasting blood glucose level (mg/dL)
- `family_history` – family history of diabetes (0 = no, 1 = yes)
- `physical_activity_hours_per_week` – weekly physical activity

Target variable:

- `high_diabetes_risk` – binary label (0 = low risk, 1 = high risk)

## Implementation

### Custom Implementation

The custom implementation focuses on understanding how decision trees work internally. It includes:

- Explicit computation of entropy to measure node impurity
- Information gain to evaluate the quality of splits
- Search over all features and possible thresholds
- Recursive tree construction
- Prediction by traversing the tree from root to leaf

No machine learning libraries are used in this implementation.

### scikit-learn Implementation

The scikit-learn implementation serves as a reference and comparison. It uses:

- `DecisionTreeClassifier` with the `entropy` criterion
- Built-in stopping criteria such as `max_depth` and `min_samples_split`

The same dataset and feature set are used to allow a direct comparison with the custom implementation.

## Run

To run the custom decision tree implementation, execute:

```bash
python3 custom_implementation.py
```

To run the scikit-learn decision tree implementation, execute:

```bash
python3 sklearn_implementation.py
```
