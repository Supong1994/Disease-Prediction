import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('D:\\Datasets\\DiseaseDataset\\improved_disease_dataset.csv')

# Encode target
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# Features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Plot class distribution
plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.title("Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

# Resample using SMOTE + Tomek
resampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X, y)

print("Resampled Class Distribution:\n", pd.Series(y_resampled).value_counts())

# Encode categorical features
if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

# Fill missing values
X_resampled = X_resampled.fillna(0)

# Optional: Feature selection using Random Forest to reduce dimensionality
from sklearn.ensemble import RandomForestClassifier
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_resampled, y_resampled)

# Get top 20 features
feat_importances = pd.Series(rf_temp.feature_importances_, index=X_resampled.columns)
top_features = feat_importances.nlargest(20).index.tolist()
X_resampled = X_resampled[top_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Define XGBoost classifier
xgb_model = XGBClassifier(
    objective='multi:softmax',  # or 'multi:softprob' for probability outputs
    num_class=len(np.unique(y_resampled)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# XGBoost hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Evaluation
print("=" * 60)
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
