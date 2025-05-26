import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models


# Descarga el dataset desde Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
data = pd.read_csv("creditcard.csv")

# Escalamos la columna 'Amount' (las otras ya están PCA-transformadas)
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Separar features y etiquetas
X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Results:\n", classification_report(y_test, y_pred_rf))


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Results:\n", classification_report(y_test, y_pred_xgb))


model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=2048, validation_split=0.2, verbose=2)

y_pred_dl = (model.predict(X_test) > 0.5).astype("int32")
print("Deep Learning Results:\n", classification_report(y_test, y_pred_dl))


from sklearn.metrics import confusion_matrix

print("Confusion Matrix for Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
print("Confusion Matrix for XGBoost:\n", confusion_matrix(y_test, y_pred_xgb))
print("Confusion Matrix for Deep Learning:\n", confusion_matrix(y_test, y_pred_dl))
