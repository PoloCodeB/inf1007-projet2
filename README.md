# Projet INF1007 — De la donnée à l'application

## Description du projet
**Domaine : éducation.** L'application estime le risque qu'un étudiant abandonne ses études à partir de quelques informations sur son parcours (âge, moyenne du 1er semestre, taux de présence, frais de scolarité à jour, bourse, emploi à temps partiel, mode d'études).

- **Type de problème :** classification binaire (abandon : oui / non).
- **Utilisateur visé :** un conseiller pédagogique qui voudrait repérer les profils à risque pour proposer un accompagnement.
- **Avertissement :** il s'agit d'un exercice pédagogique, pas d'une aide à la décision (rappelé dans l'interface).

## Jeu de données
- **Source :** [Student Academic Performance & Dropout Risk](https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk) (Kaggle).
- **Licence :** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- **Fichier :** `donnees/etudiants.csv` (3,9 Mo).
- **Échantillonnage :** le fichier original contient 100 000 lignes, alors que l'énoncé demande entre 500 et 50 000. On garde donc un échantillon aléatoire de **30 000 lignes** tiré avec une graine fixe (`random_state=42`). La proportion d'abandons est conservée (32 %). Pour le recréer, télécharger le CSV original dans `donnees/dataset.csv`, puis :
  ```python
  import pandas as pd
  pd.read_csv("donnees/dataset.csv").sample(n=30000, random_state=42).to_csv("donnees/etudiants.csv", index=False)
  ```
- **Variables utilisées :** 3 numériques (`age`, `gpa_semester_1`, `attendance_rate`) et 4 catégorielles (`tuition_up_to_date`, `scholarship_holder`, `part_time_job` converties en Oui/Non, et `study_mode`). Cible : `dropout`.
- **Données personnelles :** aucune (pas de nom, d'adresse ni de courriel ; la colonne `person_id` n'est pas utilisée).

## Installation et lancement
```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python src/modele.py   # entraîne et sauvegarde le modèle (déjà fait : modeles/ est fourni)
python src/main.py     # lance l'interface Streamlit (équivalent à : streamlit run src/main.py)
```
L'interface s'ouvre dans le navigateur à l'adresse http://localhost:8501.

## Structure du code
- `src/donnees.py` : lecture, nettoyage et séparation des données (`JeuDeDonnees`), classes de variables (`Variable` → `VariableNumerique`, `VariableCategorielle`)
- `src/modele.py` : entraînement, évaluation, prédiction, sauvegarde (`Modele`), F1 calculé avec NumPy (`f1_numpy`)
- `src/exceptions.py` : exception personnalisée `DonneeInvalideError`
- `src/interface.py` : interface Streamlit (`Application`) ; appelle la logique, ne la contient pas
- `src/main.py` : point d'entrée
- `donnees/` : jeu de données (CSV)
- `modeles/` : modèle sauvegardé (`modele_abandon.joblib`) + métadonnées (`metadonnees.json`)
- `docs/figures/exploration.png` : graphique d'exploration
- `tests/` : tests pytest (logique, hors interface)

## Résultat du modèle
- **Modèle :** arbre de décision (`max_depth=5`, `min_samples_leaf=200`, `class_weight="balanced"`), dans un `Pipeline` avec `OneHotEncoder` pour les variables catégorielles.
- **Métrique choisie :** F1 de la classe « abandon ». Les classes sont déséquilibrées (68 % / 32 %) : l'exactitude seule récompenserait un modèle qui ne prédit jamais l'abandon.
- **Score sur l'ensemble de test (6 000 étudiants) :** F1 = **0,708**, exactitude = 0,789.
- **Référence (classe majoritaire, `DummyClassifier`) :** F1 = 0,000, exactitude = 0,679.
- **Matrice de confusion :** 3 198 vrais négatifs, 878 fausses alertes, 390 abandons manqués, 1 534 abandons détectés.
- **Limites :**
  - Le modèle repose presque entièrement sur la moyenne du 1er semestre (89 % de l'importance) ; le taux de présence et les frais à jour comptent un peu, l'âge, la bourse et le mode d'études pas du tout.
  - `class_weight="balanced"` favorise la détection des abandons au prix de fausses alertes (878).
  - Les données viennent de Kaggle et leur origine réelle n'est pas documentée : les résultats ne se transposent pas à un vrai établissement.

## Tests et couverture
```bash
pytest
pytest --cov=src
```
- **27 tests** (`tests/test_donnees.py`, `tests/test_modele.py`) : cas usuels, cas limites (bornes exactes, aucun positif) et cas d'erreur (`pytest.raises`). Un test vérifie que le F1 calculé avec NumPy est égal à celui de scikit-learn (`pytest.approx`).
- **Couverture obtenue :** `donnees.py` 97 %, `modele.py` 80 %, total 64 %. Le total est plus bas parce que `interface.py` et `main.py` ne sont pas testés (l'énoncé demande des tests hors interface).

## Documentation consultée
<!-- À VÉRIFIER : garder seulement les pages que vous avez réellement lues -->
- pandas : [`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [`pandas.cut`](https://pandas.pydata.org/docs/reference/api/pandas.cut.html)
- scikit-learn : [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- Streamlit : [`st.form`](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form), [`st.pyplot`](https://docs.streamlit.io/develop/api-reference/charts/st.pyplot)

## Outil du chapitre 7 utilisé
Un **décorateur** `chronometrer` (dans `src/modele.py`) mesure et affiche la durée de `Modele.entrainer`, ce qui permet de suivre le temps d'entraînement sans mélanger ce code avec la logique du modèle.
