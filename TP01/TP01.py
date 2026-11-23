import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# 1. Charger le dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame
# 2. Définir X et y
X = df[["bmi"]]
y = df["target"]
# 3. Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)
print("pente a=",model.coef_[0])
print ("ordonnee b ",model.intercept_)
# 5. Faire des prédictions
y_pred = model.predict(X_test)
# 6. Évaluer le modèle
print("R2 :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
# 7. Afficher les graphiques
plt.figure(figsize=(12, 8))
# Graphique 1 : Entraînement
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train)
plt.title("Données d'entraînement")
plt.xlabel("BMI")
plt.ylabel("Target")
# Graphique 3 : Droite de régression
plt.subplot(2, 2, 3)
plt.scatter(X, y)
plt.plot(X_test, y_pred, color="red")
plt.title("Droite de régression")
plt.tight_layout()
plt.show()