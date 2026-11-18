# Projet INF1007 — De la donnée à l'application

## Description du projet
**Domaine : éducation.** L'application estime le risque qu'un étudiant abandonne ses études à partir de 5 informations sur son parcours (âge, moyenne du 1er semestre, taux de présence, frais de scolarité à jour, mode d'études).

- **Type de problème :** classification binaire (abandon : oui / non).
- **Utilisateur visé :** un conseiller pédagogique souhaitant identifier les profils à risque pour un accompagnement préventif.
- **Avertissement :** il s'agit d'un exercice pédagogique, pas d'une aide à la décision (rappelé dans l'interface).

## Jeu de données
- **Source :** [Student Academic Performance & Dropout Risk](https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk) (Kaggle).
- **Licence :** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- **Fichier :** `donnees/etudiants.csv` (3,9 Mo).
- **Échantillonnage :** le fichier original contient 100 000 lignes ; on conserve un échantillon représentatif de **30 000 lignes** avec graine fixe (`random_state=42`) respectant la proportion d'abandons (32 %).
- **Variables retenues (5 caractéristiques) :** 3 numériques (`age`, `gpa_semester_1`, `attendance_rate`) et 2 catégorielles (`tuition_up_to_date` converti en Oui/Non, et `study_mode`). Cible : `dropout`.
- **Données personnelles :** aucune (pas de nom, prénom ou courriel ; identifiant exclu).

## Installation et lancement
```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python src/modele.py        # entraîne et sauvegarde le modèle (modeles/ fourni)
streamlit run src/main.py   # lance l'interface Streamlit
```
L'interface s'ouvre localement sur http://localhost:8501.

## Structure du code
- `src/donnees.py` : lecture, nettoyage et découpage (`JeuDeDonnees`), classes de variables (`Variable` → `VariableNumerique`, `VariableCategorielle`)
- `src/modele.py` : entraînement, évaluation, prédiction, persistance (`Modele`), calcul F1 vectorisé (`f1_numpy`)
- `src/exceptions.py` : exception personnalisée `DonneeInvalideError`
- `src/interface.py` : interface Streamlit (`Application`) ; appelle la logique sans la contenir
- `src/main.py` : point d'entrée
- `donnees/` : jeu de données CSV
- `modeles/` : modèle sauvegardé (`modele_abandon.joblib`) + métadonnées (`metadonnees.json`)
- `docs/figures/exploration.png` : graphique d'exploration
- `tests/` : tests unitaires pytest (logique, hors interface)

## Résultat du modèle
- **Modèle :** arbre de décision (`max_depth=5`, `min_samples_leaf=200`, `class_weight="balanced"`). Variables catégorielles encodées avec `pd.get_dummies`.
- **Métrique choisie :** F1 de la classe « abandon » (classes déséquilibrées : 68 % poursuite / 32 % abandon).
- **Score sur l'ensemble de test (6 000 étudiants) :** F1 = **0,698**, exactitude = 0,774.
- **Référence (classe majoritaire, `DummyClassifier`) :** F1 = 0,000, exactitude = 0,679.
- **Matrice de confusion :** 3 074 vrais négatifs, 1 002 fausses alertes, 357 abandons manqués, 1 567 abandons détectés.

## Tests et couverture
```bash
pytest
pytest --cov=src
```
- **27 tests** (`tests/test_donnees.py`, `tests/test_modele.py`) couvrant cas usuels, limites et cas d'erreur (`pytest.raises`). Vérification de l'équivalence exacte entre `f1_numpy` et scikit-learn via `pytest.approx`.
- **Couverture obtenue :** `donnees.py` 97 %, `modele.py` 86 %, total 68 % (l'interface Streamlit n'est pas couverte par les tests unitaires selon l'énoncé).

## Documentation consultée
- pandas : [`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [`pandas.cut`](https://pandas.pydata.org/docs/reference/api/pandas.cut.html), [`pandas.get_dummies`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- scikit-learn : [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- Streamlit : [`st.form`](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form), [`st.pyplot`](https://docs.streamlit.io/develop/api-reference/charts/st.pyplot)

## Outil du chapitre 7 utilisé
Un **décorateur** `@chronometrer` (dans `src/modele.py`) mesure et affiche en console la durée d'exécution de `Modele.entrainer`.
