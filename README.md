# Prédiction du churn client (Telco)

Modèle qui prédit quels clients d'un opérateur télécom vont résilier leur abonnement, pour aider à les retenir.

## Données
*Telco Customer Churn* (IBM, via Kaggle) : 7 043 clients, 21 colonnes, environ 26,5 % de churn. (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Méthode
- Analyse exploratoire : qualité des données, nettoyage de `TotalCharges`, churn selon les variables.
- Séparation train/test stratifiée (80/20).
- `Pipeline` avec `ColumnTransformer` (StandardScaler et OneHotEncoder) : le préprocesseur apprend seulement sur le train.
- Classes déséquilibrées : `class_weight="balanced"`.
- Comparaison de 4 modèles par validation croisée (5 plis) sur le train.
- Choix du seuil de décision sur le train, puis vérification sur le test.

## Résultats
| Modèle | AUC (CV) | Recall (CV) |
|---|---|---|
| Régression logistique | 0.846 | 0.801 |
| XGBoost | 0.846 | 0.799 |
| Random Forest | 0.846 | 0.789 |
| Arbre (profondeur 5) | 0.828 | 0.777 |

**Modèle final :** régression logistique, seuil 0.4. Sur le test : AUC 0.841, recall 0.866, précision 0.463.
Le modèle trouve environ 87 % des clients qui partent, au prix de fausses alertes.

## Ce que montre le modèle
- La fibre optique, un contrat mensuel et une faible ancienneté vont avec un risque de churn plus élevé.
- Le DSL, un contrat de deux ans et une grande ancienneté vont avec un risque plus faible.


## Structure du projet
- `Notebooks/01_eda.ipynb` : analyse exploratoire
- `Notebooks/02_modeling.ipynb` : modélisation
- `models/churn_model.joblib` : pipeline et seuil
- `data/` : jeu de données

## Lancer le projet
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Ouvrir les notebooks dans l'ordre.

## À venir
- [ ] Application Streamlit
