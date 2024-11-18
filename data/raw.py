import pandas as pd

def charger_donnees_brutes(chemin_fichier):
  """
  Charge des données brutes depuis un fichier CSV.

  Args:
    chemin_fichier: Chemin d'accès au fichier CSV.

  Returns:
    Un DataFrame pandas contenant les données brutes.
  """
  try:
    data = pd.read_csv(chemin_fichier)
    return data
  except FileNotFoundError:
    print(f"Erreur: Fichier non trouvé: {chemin_fichier}")
    return None