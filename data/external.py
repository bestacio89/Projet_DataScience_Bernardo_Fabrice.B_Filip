import requests

def recuperer_donnees_api(url):
  """
  Récupère des données depuis une API.

  Args:
    url: URL de l'API.

  Returns:
    Un dictionnaire ou une liste contenant les données récupérées.
  """
  try:
    response = requests.get(url)
    response.raise_for_status()  # Lever une exception si la requête a échoué
    data = response.json()  # Convertir la réponse en JSON
    return data
  except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête à l'API: {e}")
    return None