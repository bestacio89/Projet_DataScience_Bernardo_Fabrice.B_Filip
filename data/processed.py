import pandas as pd

def nettoyer_donnees(data):
  """
  Nettoie et transforme un DataFrame pandas.

  Args:
    data: DataFrame pandas à nettoyer.

  Returns:
    Un DataFrame pandas nettoyé et transformé.
  """
  # Supprimer les doublons
  data.drop_duplicates(inplace=True)

  # Remplacer les valeurs manquantes par la moyenne
  data.fillna(data.mean(), inplace=True)

  # Convertir une colonne en type datetime
  data['date'] = pd.to_datetime(data['date'])

  # ... autres transformations ...

  return data