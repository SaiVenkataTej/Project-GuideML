# Techniques used (models)

## **1. Linear Regression**

### **🛠️ Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Regression |

---

### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer (Median) | Imputation |
| OneHotEncoder (`drop= 'first'` to avoid dummy variable trap) | Encoding |

---

### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| StandardScaler | Scaling |
| RobustScaler | Scaling |
| Log Transformation | Transformation |
| Box - Cox Transformation | Transformation |
| PowerTransformer | Transformation |
| SelectKBest(f_Regression) | Selection |
| VarianceThreshold | Selection |

---

### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| K - Fold Cross Validation (Shuffle = True) | Cross - Validation |
| Ridge(L2) | Regularization |
| Lasso(L1) | Regularization |
| ElasticNet | Regularization |
| GridSearch CV (`alpha`, `l1_ratio`) | Hyper Parameter Optimization |

---

### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| RMSE (Root Mean Squared Error) | **Formula**: $\sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$. |
| MAE (Mean Absolute Error) | **Formula**: $\frac{1}{N} \sum_{i=1}^{N} |
| Adjusted $R^2$ | **Interpretation** Proportion of variance explained by the model, adjusted for the number of predictors. Prevents inflation of $R^2$ with extraneous features. |
| $R^2$ Score | **Formula**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. **Interpretation**: Proportion of the variance in the dependent variable predictable by the model. Range $0$ to $1$. |
| Log Loss | **Formula**:$\frac{-1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$. **Interpretation**: Penalizes confident incorrect predictions (even if not primary metric for LR). Lower is better. |

---

### **📈 Visualization & Interpretation**

#### **Diagnostic**
* Residual Plots (Homoscedasticity check)
* Q-Q Plots (Normality check)
#### **Interpretation**
* Coefficient Bar Charts (Weights)
* Predicted vs. Actual Scatter Plot
---

## **2. Logistic Regression**

### **🛠️ Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Classification |

---

### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer (Median / Mode strategy) | Imputation |
| OneHotEncoder (`handle_unknown ='ignore'`) | Encoding |
| LabelEncoder | Encoding |

---

### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| StandardScaler | Scaling |
| Recursive Feature Elimination (RFE) | Selection |
| SelectFromModel | Selection | 

---

### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| Stratified K - Fold | Cross - Validation |
| Class Weights ('balanced') | Imbalance |
| GridSearch CV (`C`, `penalty`, `solver`) | Hyper Parameter Optimization |

---

### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| F1 - Score | **Formula**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$. **Interpretation**: Harmonic mean of precision and recall. Useful for imbalanced classes. |
| ROC - AUC Score | **Interpretation**: Area Under the Receiver Operating Characteristic Curve. Measures the model's ability to distinguish between classes across all possible thresholds. Range $0.5$ to $1$. |
| Accuracy | **Formula**: $\frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$. **Interpretation**: Ratio of correct predictions to the total number of predictions. |
| Log Loss | **Formula (Binary)**: $\frac{-1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$. **Interpretation**: Measures the uncertainty of the probability predictions. Lower is better. |
| Precision - Recall Score | **Interpretation**: Measures trade-off between Precision (correct positive predictions) and Recall (correctly identified positives). |

---

### **📈 Visualization & Interpretation**

#### **Diagnostic**
* ROC Curve
* Precision - Recall Score
* Confusion Matrix
#### **Interpretation**
* Odds Ratio Plot (Exponentiated Coefficients)

---

## **3. Naive Bayes**

### **🛠️ Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Classification |

---

### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer (Mean for Gaussian, Mode for Categorical) | Imputation |
| OrdinalEncoder | Encoding (Categorical NB) |
| OneHotEncoder | Encoding (Multinomial NB) |
| StandardScaler | Scaling |
| MinMaxScaler | Scaling |
---

### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| PowerTransformer (Yeo-Johnson for Gaussian assumption) | Transformation |
| SelectKBest (`chi2` or `mutual_info_classif`) | Selection |

---

### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| Stratified K - Fold | Cross - Validation |
| Class Priors Adjustment | Imbalance |
| GridSearch CV (`var_smoothing` for Gaussian, `alpha` for multinomial) | Hyper Parameter Optimization |

---

### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| Log Loss (Probabilistic Accuracy) | **Formula (Binary)**: $\frac{-1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$. **Interpretation**: Measures how close the predicted probability is to the true value. Lower is better. |
| Accuracy | **Interpretation**: General measure of correctness. |
| F1 - Score | **Interpretation**: Harmonic mean of precision and recall. |
| Precision | **Formula**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$. **Interpretation**: Out of all positive predictions, how many were correct. |

---

### **📈 Visualization & Interpretation**

#### **Diagnostic**
* Confusion Matrix Heatmap
#### **Interpretation**
* Feature Distribution Histograms (per class)
* Feature Log- Probabilities

---

## **4. Random Forest**

### **🛠️ Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Classification &  Regression |

---

#### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer (Constant, Mean or Median) | Imputation |
| OrdinalEncoder | Encoding |
| OneHotEncoder | Encoding |

---

#### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| SelectFromModel (Tree - based selection) | Selection |

---

#### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| Stratified K - Fold (Classification) | Cross-Validation |
| K - Fold (Regression) | Cross- Validation |
| Class Weights (`balanced` or `balanced_subsample`) | Imbalance |
| Randomized Search CV (`n_estimators`, `max_depth`, `max_features`) | Hyper Parameter Optimization |

---

#### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| F1 - Score | **Interpretation**: Harmonic mean of precision and recall (for classification). |
| RMSE (Root Mean Square Error) | **Formula**: $\sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$. **Interpretation**: Average magnitude of the errors in target units (for regression). |
| OOB Score (Out-of-Bag Error) | **Interpretation**: Internal estimate of generalization error. Calculated using samples not included in the bootstrap set for training a specific tree. |
---

#### **📈 Visualization & Interpretation**

##### **Diagnostic**
* Learning Curves
* ROC curves
##### **Interpretation**
* Feature Importance Plot (MDI or Permutation Importance)
* Tree Diagram (Single Estimator)

---

## **5. SVM**

### **🛠️ Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Classification & Regression |

---

#### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer (Median) | Imputation |
| OneHotEncoder | Encoding  |

---

#### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| MinMaxScaler | Scaling |
| StandardScaler | Scaling |
| PCA (Principal Component Analysis) | Dimensionality Reduction |

---

#### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| Stratified K-Fold | Cross - Validation |
| Class Wrights (`balanced`) | Imbalance |
| Randomized Search CV(`C`, `gamma`, `kernel`) | Hyper Parameter Optimization |

---

#### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| Accuracy | **Interpretation**: General measure of correctness. |
| F1 - Score | **Interpretation**: Harmonic mean of precision and recall. |

---

#### **📈 Visualization & Interpretation**

##### Diagnostic
* Confusion Matrix
* Decision Boundary Plot (2D projection)

##### Interpretation:
* Inspection of Support Vectors

---

## **6. Decision Trees**

### 🛠️ **Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Classification &  Regression |

---

#### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer (Mode / Median)| Imputation|
| OneHotEncoder | Encoding |
| OrdinalEncoder| Encoding |

---

#### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| VarianceThreshold | Selection |
| SelectKBest | Selection |
---

#### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| Stratified K - Fold | Cross - Validation  |
| Cost-Complexity Pruning (CCP) | Optimization|
| GridSearchCV (`maxDepth`, `min_samples_split`, `min_samples_leaf`) | Hyper Parameter Optimization |

---

#### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| Accuracy| **Interpretation**: General measure of correctness (for classification). |
| Root Mean Square Error | **Formula**: $\sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$. **Interpretation**: Average magnitude of the errors in target units (for regression). |
| Tree Depth | **Interpretation**: The length of the longest path from the root node to a leaf node. Indicator of model complexity. |
| Leaf Count | **Interpretation**: The total number of terminal nodes. Indicator of model complexity. | 
---

#### **📈 Visualization & Interpretation**

##### **Diagnostic**
* Validation Curve (Depth vs score)

##### **Interpretation**
* Graphviz Tree Diagram (Rule Visualization)
* Gini Importance Plot

--- 

## **7. K-Nearest Neighbors**

### **🛠️ Key Components and Techniques**

| Category | Technique Names |
| :--- | :--- |
| **Operation** | Classification & Regression |

---

#### **🧺 Data Preparation & Handling**

| Technique Names | Description / Purpose |
| :--- | :--- |
| SimpleImputer(Median) | Imputation |
| OneHotEncoder | Encoding |

---

#### **🔬 Preprocessing & Feature Engineering**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
|MinMaxScaler | Scaling|
|StandardScaler |Scaling |
|PCA | Dimensionality Reduction (Essential for high dimensions)|
|NCA (Neighbor Components Analysis) | Dimensionality Reduction|

---

#### **⚙️ Training & Optimization**

| Technique Names | Purpose / Parameters |
| :--- | :--- |
| Stratified K-Fold | Cross - Validation|
| Distance Weighting | Optimization |
| GridSearchCV |Hyper Parameter Optimization |
| Class adjustments | For classification |

---

#### **📊 Evaluation & Metrics**

| Metric | Formula / Interpretation |
| :--- | :--- |
| Accuracy | **Formula**: $\frac{1}{N} \sum_{i=1}^{N}. **Interpretation**: General measure of correctness (for classification) |
| MAE| **Formula**: $\frac{1}{N} \sum_{i=1}^{N} |
| Prediction Latency| **Interpretation**: The time taken for the model to make a prediction on a new sample (inference time). |

---

#### **📈 Visualization & Interpretation**

##### **Diagnostic**
* Elbow Method Plot (Error vs. K)
* Decision Boundary Plot

##### **Interpretation**
* Local Neighbor Inspection

# 🔬 Techniques Grouped by Functionality
## **1. 🧺 Data Handling & Preparation**
These techniques are used to clean the raw input data and prepare it for the model's consumption, primarily addressing missing values and categorical features.

### **Imputation (Missing Values)**:

* SimpleImputer (Median, Mode, Mean, Constant)

### **Encoding (Categorical Features)**:

* OneHotEncoder (with parameters like drop='first' or handle_unknown='ignore')
* OrdinalEncoder
* LabelEncoder (typically for the target variable y)

## **2. 🧩 Preprocessing & Feature Engineering**
These techniques are applied to the feature set (X) to satisfy model requirements (like scaling for distance-based models) or to improve model performance (like dimensionality reduction).

### **Scaling & Normalization**:

* StandardScaler
* MinMaxScaler
* RobustScaler

### **Transformation (Distribution/Form)**:

* Log Transformation
* Box - Cox Transformation
* PowerTransformer (e.g., Yeo-Johnson for Gaussian assumption)

### **Feature Selection & Dimensionality Reduction**:

* VarianceThreshold
* SelectKBest (with parameters like f_Regression, χ2 , mutual_info_classif)
* Recursive Feature Elimination (RFE)
* SelectFromModel (used to select features based on coefficients/importance)
* PCA (Principal Component Analysis)
* NCA (Neighbor Components Analysis)

## **3. ⚙️ Training & Optimization**
These techniques deal directly with the model fitting process, stability, parameter search, and handling specific data issues during training.

### **Cross-Validation**:

* K-Fold Cross Validation
* Stratified K-Fold (for classification)
* Regularization & Optimization:
* Ridge (L2)
* Lasso (L1)
* ElasticNet
* Cost-Complexity Pruning (CCP)
* Distance Weighting (in KNN)

### **Hyperparameter Optimization (HPO)**:

* GridSearch CV (with model-specific parameters like alpha, C, gamma, maxDepth, etc.)
* Randomized Search CV

### **Imbalance Handling**:

* Class Weights (balanced or balanced\_subsample)
* Class Priors Adjustment (in Naive Bayes)
* Class adjustments (general term for techniques like class weights or sampling)

## **4. 📊 Evaluation**
These techniques are used to measure the model's performance and provide diagnostic insight into its structure and behavior.

### **Regression Metrics**:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R<sup>2</sup> Score
* Adjusted R<sup>2</sup>
 
### **Classification Metrics**:

* Accuracy
* F1 - Score
* ROC - AUC Score
* Precision
* Log Loss
* Precision - Recall Score

### Model-Specific Metrics/Diagnostics:
* OOB Score (Out-of-Bag Error - for Random Forest)
* Tree Depth (Decision Trees)
* Leaf Count (Decision Trees)
* Prediction Latency (KNN)

## **5. 📊 Visualization & Interpretation**:
* Confusion Matrix (or Heatmap)
* Residual Plots (Homoscedasticity check)
* Q-Q Plots (Normality check)
* ROC Curve / Precision - Recall Curve
* Learning Curves / Validation Curve
* Coefficient Bar Charts / Odds Ratio Plot
* Feature Importance Plot (MDI, Gini)
* Tree Diagram (Graphviz)
* Decision Boundary Plot
* Elbow Method Plot (for KNN K selection)
* Inspection of Support Vectors / Local Neighbor Inspection