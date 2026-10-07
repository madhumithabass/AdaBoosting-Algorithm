#  AdaBoost Algorithm (Adaptive Boosting)

A clear and practical implementation of the **AdaBoost (Adaptive Boosting)** machine learning algorithm, demonstrating both core principles and practical usage in Python.

---

##  Table of Contents
- [What is AdaBoost?](#-what-is-adaboost)
- [How AdaBoost Works (Step-by-Step)](#-how-adaboost-works-step-by-step)
- [Core Mathematical Formulas](#-core-mathematical-formulas)
- [Key Hyperparameters](#-key-hyperparameters)
- [Pros and Cons](#-pros-and-cons)
- [Quick Start Code](#-quick-start-code)
- [Project Structure](#-project-structure)

---

##  What is AdaBoost?

**AdaBoost (Adaptive Boosting)** is a popular **ensemble learning** method that transforms a series of **weak learners** (models that perform only slightly better than random guessing) into a single **strong learner**.

In classification tasks, AdaBoost most commonly uses **Decision Stumps**—single-split decision trees with a depth of 1 (1 node and 2 leaves)—as its base estimators.

---

##  How AdaBoost Works (Step-by-Step)

1. **Equal Weight Initialization:**
   Every training sample starts with equal importance:
   $$w_i = \frac{1}{N}$$
   *(where $N$ is the total number of samples).*

2. **Sequential Weak Learner Training:**
   A weak learner is trained on the weighted dataset.

3. **Error Calculation:**
   The algorithm calculates the total error ($\epsilon$) by summing the weights of all misclassified samples.

4. **Calculate Amount of Say ($\alpha$):**
   The model determines how much influence this weak learner will have in the final decision. High accuracy = large say; low accuracy = small say.

5. **Update Sample Weights:**
   - **Incorrectly predicted samples:** Weights are **increased** (so the next tree focuses heavily on them).
   - **Correctly predicted samples:** Weights are **decreased**.
   - Weights are then normalized so their sum equals 1.

6. **Final Aggregate Prediction:**
   When predicting a new data point, all weak learners vote, weighted by their respective Amount of Say ($\alpha$).

---

##  Core Mathematical Formulas

### 1. Total Error ($\epsilon_t$)
$$\epsilon_t = \frac{\sum_{\text{misclassified}} w_i}{\sum_{i=1}^{N} w_i}$$

### 2. Amount of Say ($\alpha_t$)
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

### 3. Weight Update Rule
- For **incorrect** samples:
  $$w_{\text{new}} = w_{\text{old}} \cdot e^{\alpha_t}$$
- For **correct** samples:
  $$w_{\text{new}} = w_{\text{old}} \cdot e^{-\alpha_t}$$

---

##  Key Hyperparameters (`AdaBoostClassifier`)

| Parameter | Default | Meaning |
| :--- | :---: | :--- |
| `estimator` | `DecisionTreeClassifier(max_depth=1)` | The base weak model to train iteratively. |
| `n_estimators` | `50` | The maximum number of estimators to train sequentially. |
| `learning_rate` | `1.0` | Controls the step size / shrinkage applied to each estimator's contribution. |
| `random_state` | `None` | Seed for reproducibility of random state generation. |

---

##  Pros and Cons

###  Advantages
- Simple to understand and fast to train compared to complex deep learning models.
- Minimal hyperparameter tuning required out-of-the-box.
- Less prone to overfitting compared to single unpruned decision trees when `learning_rate` and `n_estimators` are properly tuned.

### Limitations
- **Sensitive to noisy data and outliers:** Because AdaBoost aggressively raises weights on incorrect points, it will over-prioritize extreme outliers.
- **No parallel training:** Sequential by design—each learner depends entirely on the errors of the preceding learner.

---

##  Quick Start Code

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define base estimator (decision stump)
base_estimator = DecisionTreeClassifier(max_depth=1)

# 4. Initialize and fit AdaBoost
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
