# Fiche de décisions

<!-- Les colonnes « Validé par l'auxiliaire » et « Date » sont à remplir après chaque validation (A à D).
     Les justifications sont une première version : adaptez-les à vos propres mots. -->

| Choix | Options envisagées | Choix retenu et justification | Validé par l'auxiliaire | Date |
|-------|--------------------|-------------------------------|-------------------------|------|
| Domaine | <!-- À COMPLÉTER : autres domaines envisagés --> | **Éducation (abandon scolaire).** Sujet proche de notre réalité d'étudiants ; un conseiller pourrait s'en servir pour repérer les profils à risque. Domaine sensible : avertissement affiché dans l'interface. | | |
| Jeu de données | <!-- À COMPLÉTER : autres jeux envisagés --> | **Student Academic Performance & Dropout Risk (Kaggle).** Tabulaire (CSV), aucune valeur manquante, cible binaire claire, variables numériques et catégorielles, aucune donnée identifiante. | | |
| Taille du jeu de données | Garder les 100 000 lignes ; échantillon de 30 000 lignes | **Échantillon de 30 000 lignes** (`random_state=42`) : l'énoncé limite à 50 000. 30 000 reste largement suffisant pour un arbre de décision et la proportion d'abandons (32 %) est conservée. | | |
| Variables d'entrée | Toutes les colonnes ; une sélection simple | **7 variables** : âge, moyenne du 1er semestre, taux de présence, frais à jour, bourse, emploi à temps partiel, mode d'études. Peu de champs à remplir dans l'interface et des informations qu'un conseiller connaît. `study_mode` ajouté pour avoir une vraie variable catégorielle. | | |
| Variables 0/1 | Garder 0/1 (numérique) ; convertir en Oui/Non (catégorielle) | **Oui/Non** : plus clair pour l'utilisateur (liste déroulante) et cohérent avec l'encodage `OneHotEncoder`. | | |
| Variable cible et métrique | Exactitude ; F1 ; rappel | **Cible `dropout`, métrique F1 de la classe abandon.** Classes déséquilibrées (68 % / 32 %) : un modèle qui ne prédit jamais l'abandon aurait 68 % d'exactitude mais un F1 de 0. | | |
| Modèle | Régression logistique ; k-NN ; arbre de décision ; forêt aléatoire | **Arbre de décision** (`max_depth=5`, `min_samples_leaf=200`, `class_weight="balanced"`) : simple à expliquer, gère bien les seuils (ex. GPA < 2), donne l'importance des variables. Profondeur limitée contre le surapprentissage ; `balanced` pour mieux détecter les abandons. | | |
| Référence | Classe majoritaire ; prédiction aléatoire | **Classe majoritaire** (`DummyClassifier`) : F1 = 0, exactitude = 0,679. Le modèle (F1 = 0,708, exactitude = 0,789) fait nettement mieux. | | |
| Technologie d'interface | Tkinter ; Streamlit | **Streamlit** : formulaire et graphiques Matplotlib (`st.pyplot`) en peu de code, interface claire dans le navigateur, fonctionne en local. | | |
| Saisie des nombres | Curseurs (sliders) ; champs texte | **Champs texte** : un curseur empêche toute saisie invalide, alors que l'énoncé demande de valider les saisies vides, non numériques ou hors intervalle et d'afficher un message clair. | | |
