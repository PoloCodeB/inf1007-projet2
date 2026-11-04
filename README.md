# Projet INF1007 — De la donnée à l'application

## Description du projet
<!-- À compléter : domaine, question à laquelle l'application répond, type de problème (classification / régression) -->

## Jeu de données
- Source :
- Licence :
- Fichier : `donnees/` (si < 25 Mo et licence compatible, sinon lien + instructions de téléchargement ici)

## Installation et lancement
```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

## Structure du code
- `src/donnees.py` : lecture, nettoyage, classes de variables
- `src/modele.py` : entraînement, évaluation, prédiction
- `src/exceptions.py` : exceptions personnalisées
- `src/interface.py` : interface graphique (appelle la logique, ne la contient pas)
- `src/main.py` : point d'entrée
- `donnees/` : jeu de données (CSV)
- `modeles/` : modèle sauvegardé (.joblib) + métadonnées (.json)
- `tests/` : tests pytest (logique, hors interface)

## Résultat du modèle
- Métrique choisie :
- Score sur l'ensemble de test :
- Référence (modèle trivial) :
- Limites :

## Tests et couverture
```bash
pytest
pytest --cov=src
```
- Couverture obtenue :

## Documentation consultée
- pandas :
- scikit-learn :
- Technologie d'interface :

## Outil du chapitre 7 utilisé
<!-- À compléter en une phrase : décorateur / générateur / lambda, et où il a du sens -->
