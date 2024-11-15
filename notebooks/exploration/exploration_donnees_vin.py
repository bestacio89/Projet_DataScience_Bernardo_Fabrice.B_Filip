
# Importer les librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data = pd.read_csv("../../data/raw/vin.csv")

# Afficher les premières lignes
print(data.head())

# Afficher les informations sur les colonnes
print(data.info())

# Statistiques descriptives
print(data.describe())

# Histogrammes des variables numériques
data.hist(figsize=(12, 8))
plt.show()

# Nuage de points entre deux variables
plt.scatter(data['fixed acidity'], data['volatile acidity'])
plt.xlabel('Fixed Acidity')
plt.ylabel('Volatile Acidity')
plt.show()

# Matrice de corrélation
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()