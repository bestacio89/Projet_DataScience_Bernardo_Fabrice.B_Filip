import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessor_donnees_vin(chemin_fichier, variables_numeriques, variables_categorielles, test_size=0.2, random_state=42):
  """
  Préprocesse les données du vin.

  Args:
    chemin_fichier: Chemin d'accès au fichier CSV contenant les données du vin.
    variables_numeriques: Liste des noms des variables numériques.
    variables_categorielles: Liste des noms des variables catégorielles.
    test_size: Proportion des données à utiliser pour l'ensemble de test.
    random_state: Graine aléatoire pour la reproductibilité.

  Returns:
    X_train_preprocessed: Données d'entraînement prétraitées.
    X_test_preprocessed: Données de test prétraitées.
    y_train: Variables cibles d'entraînement.
    y_test: Variables cibles de test.
  """
  try:
    # Charger les données
    data = pd.read_csv(chemin_fichier)

    # Séparer les features et la variable cible
    X = data.drop('target', axis=1)
    y = data['target']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Créer le preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), variables_numeriques),
            ('cat', OneHotEncoder(), variables_categorielles)
        ])

    # Appliquer le preprocessing sur les données d'entraînement
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # Appliquer le preprocessing sur les données de test
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

  except FileNotFoundError:
    print(f"Erreur: Fichier non trouvé: {chemin_fichier}")
    return None, None, None, None
  except Exception as e:
    print(f"Une erreur est survenue: {e}")
    return None, None, None, None

if __name__ == "__main__":
  chemin_fichier_vin = "../../data/raw/vin.csv"  # Adaptez le chemin si nécessaire
  variables_numeriques = ['fixed acidity', 'volatile acidity']  # Adaptez les variables
  variables_categorielles = ['type']  # Adaptez les variables

  X_train_preprocessed, X_test_preprocessed, y_train, y_test = preprocessor_donnees_vin(
      chemin_fichier_vin, variables_numeriques, variables_categorielles
  )

  if X_train_preprocessed is not None:
    print("Données prétraitées avec succès!")
    # ... (Code pour sauvegarder les données prétraitées si nécessaire)