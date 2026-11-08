# Breast Tumour Classifier Lab

Machine Learning - Assignment 2 (M.Tech AIML, BITS WILP)

- **GitHub repository:** `<PASTE YOUR GITHUB REPO LINK HERE>`
- **Live Streamlit app:** `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`

## a. Problem statement

Predict whether a breast mass is **malignant** or **benign** from numeric measurements
of cell nuclei taken from a digitised fine-needle-aspirate image. This is a binary
classification problem. Five classifiers are trained on the same data and compared with
six metrics, then exposed through a Streamlit app that evaluates the saved models on an
uploaded test CSV.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository (bundled with scikit-learn as `load_breast_cancer`) |
| Instances | 569 (minimum required: 500) |
| Features | 30 numeric (minimum required: 12) |
| Target | `diagnosis` - 1 = benign (357), 0 = malignant (212) |
| Missing values | None |

The 30 features are the mean, standard error and "worst" value of 10 nuclear
measurements: radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry and fractal dimension.

**Split:** stratified 80/20 train/test split (`random_state=42`): 455 training rows,
114 test rows. The test rows are saved as `test_data.csv` and are what the app uses.
**Preprocessing:** `StandardScaler` is fitted on the training data inside each saved
`Pipeline`, so uploaded data needs no manual scaling.

## c. Repository layout

```
bc_classifier/
|-- app.py                # Streamlit app
|-- train_models.py       # trains + saves all models, writes metrics.csv and test_data.csv
|-- requirements.txt
|-- README.md
|-- test_data.csv         # 114-row held-out test set (features + target)
|-- full_dataset.csv      # complete dataset, for reference
`-- model/                # logistic_regression, decision_tree, knn, naive_bayes,
                          # support_vector_machine (.joblib) + metrics.csv
```

Run locally: `pip install -r requirements.txt`, then `python train_models.py`
(optional, models are already saved), then `streamlit run app.py`.

## d. Models used

Hyper-parameters: Logistic Regression (`max_iter=1000`); Decision Tree (`max_depth=5`);
KNN (`k=7`); Gaussian Naive Bayes (defaults); SVM (RBF kernel, `C=1`).
Metrics are computed on the 114-row test set (positive class = benign).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint best (2 errors in 114) and the highest AUC (0.995). The classes are close to linearly separable once features are scaled, so a simple linear model is enough. Fast, interpretable and well calibrated. |
| Decision Tree | Weakest model (9 errors, AUC 0.916). A single depth-5 tree is high-variance and overfits the 455 training rows; its probabilities are coarse (few leaf values), which also pulls AUC down. |
| KNN | Very good (3 errors) and perfect recall for benign cases, but precision is lower (0.96), meaning a few malignant tumours were labelled benign, which is the costlier error here. Depends heavily on scaling and `k`. |
| Naive Bayes | Good ranking ability (AUC 0.987) but only 0.930 accuracy. The features are strongly correlated (e.g. radius, perimeter, area), which breaks the independence assumption and makes its hard decisions less reliable. |
| Support Vector Machine | Ties Logistic Regression on every threshold metric (2 errors) with AUC 0.995. The RBF kernel handles the margin well, at higher training and storage cost than the linear model. |
| **Overall winner** | **Logistic Regression** - joint-best Accuracy/F1/MCC with SVM and the highest AUC, while being the simplest, smallest and most interpretable. Caveat: the test set has only 114 rows (one prediction is about 0.9 percentage points), so LR, SVM and KNN are statistically close; cross-validation would be needed to separate them with confidence. |

## Streamlit app features

1. Upload a test CSV (30 feature columns + `diagnosis`), or use the bundled sample.
2. Dropdown to select any of the five models (loaded from `model/`, never retrained).
3. Accuracy, AUC, Precision, Recall, F1 and MCC shown for the chosen model.
4. Confusion matrix, ROC curve and classification report.
5. "Compare all models" tab with the full metric table (best value per metric highlighted)
   and a predictions download button.
