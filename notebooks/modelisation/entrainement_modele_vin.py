# Importer les librairies
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.models import entrainer_modele_regression, sauvegarder_modele

# Charger les données prétraitées (optionnel)
# X_train_preprocessed = pd.read_csv("../../data/processed/X_train_vin.csv")
# X_test_preprocessed = pd.read_csv("../../data/processed/X_test_vin.csv")
# y_train = pd.read_csv("../../data/processed/y_train_vin.csv")
# y_test = pd.read_csv("../../data/processed/y_test_vin.csv")

# Entraîner le modèle
modele = entrainer_modele_regression(X_train_preprocessed, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = modele.predict(X_test_preprocessed)

# Évaluer le modèle
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Sauvegarder le modèle
sauvegarder_modele(modele, "../../models/regression_lineaire/modele_vin.pkl")