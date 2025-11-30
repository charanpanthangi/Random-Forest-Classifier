# Random Forest Classifier Template

Beginner-friendly template that demonstrates how a Random Forest classifier works on the classic Iris dataset using scikit-learn.

## What is Random Forest Classification?
Random Forest is an **ensemble** of many decision trees. Each tree is trained on a **bootstrap** (bagged) sample of the data and only sees a random subset of features when splitting nodes. Combining many diverse trees through majority vote reduces variance and delivers strong accuracy compared to a single tree.

### Why it works well
- **Bagging (bootstrap aggregation):** each tree learns from a slightly different dataset, smoothing out noise.
- **Feature randomness:** using only a subset of features per split decorrelates trees, making the ensemble more robust.
- **Reduced overfitting:** a single decision tree can memorize the training set; averaging many trees lowers variance.

### Key hyperparameters to explore
- `n_estimators`: number of trees in the forest.
- `max_depth`: maximum depth of each tree (None lets trees grow until pure).
- `max_features`: number of features considered at each split.

## Dataset
The project uses the scikit-learn **Iris** dataset (150 samples, 4 numeric features, 3 species labels).

## Project Pipeline
1. Load data from scikit-learn.
2. Split into train and test sets (no scaling required for trees).
3. Train `RandomForestClassifier`.
4. Evaluate accuracy, precision, recall, F1, and confusion matrix.
5. Visualize confusion matrix, feature importance, and a PCA scatter plot.

## Repository Structure
```
app/
  data.py          # Load Iris dataset
  preprocess.py    # Train/test split
  model.py         # RandomForestClassifier factory
  evaluate.py      # Metrics helpers
  visualize.py     # Confusion matrix, feature importance, PCA plots
  main.py          # End-to-end pipeline
notebooks/
  demo_random_forest_classifier.ipynb
examples/
  README_examples.md
requirements.txt
Dockerfile
LICENSE
```

## Running the Pipeline
Install dependencies and run the script:
```bash
pip install -r requirements.txt
python app/main.py
```
Generated SVG plots and metrics are stored in the `outputs/` folder.

To explore interactively, open the notebook:
```bash
jupyter notebook notebooks/demo_random_forest_classifier.ipynb
```

## Why scaling is optional here
Decision trees split data based on thresholds and do not depend on feature scaling, so Random Forests inherit this property. You can still scale features for consistency with other models, but it is not required.

## Feature Importance
Random Forest provides a feature importance score per feature. Higher values mean the feature contributed more to reducing impurity across trees. The repository prints a sorted table and a bar plot for easy interpretation.

## Future Improvements
- Add out-of-bag (OOB) score reporting.
- Perform hyperparameter tuning (e.g., grid search or randomized search).
- Compare with gradient-boosted trees such as XGBoost.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
