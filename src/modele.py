import os

import joblib
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


class Modele:
    def pretrain_data(self):
        """
        Fait le prétraitement du dataset, qui est déja bien formé.
        Donc ont fait principalement que gardé les données dont nous avont besoin.
        """
        dataset_path = os.path.join(os.getcwd(), "dataset.csv")
        dataset_work_path = os.path.join(os.getcwd(), "dataset_work.csv")
        data = pd.read_csv(dataset_path, usecols=["age", "gpa_semester_1", "attendance_rate", "tuition_up_to_date", "scholarship_holder", "part_time_job", "dropout"])
        data.to_csv(dataset_work_path, index=False)
        return data

    def train_data(self, data) -> None:
        """
        Fait l'entrainement du model de prediction.
        """
        x = data[["age", "gpa_semester_1", "attendance_rate", "tuition_up_to_date", "scholarship_holder", "part_time_job"]]
        y = data["dropout"]

        # Limites de profondeur et taille minimale des feuilles pour éviter l'overfitting,
        # et tirage stratisfié pour conserver la répartition 68% / 32% de la variable cible.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=200, class_weight="balanced")
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        #y_prob = model.predict_proba(x_test) # Utile pour récupérer les probabilités/confiance

        exactitude = accuracy_score(y_test, y_pred)
        score_f1 = f1_score(y_test, y_pred)

        print(f"Exactitude (Accuracy) : {exactitude:.2f}")
        print(f"Score F1 : {score_f1:.2f}")

        # Validation croisée pour vérifier que les métriques ne sont pas dues au hasard d'un seul split.
        scores_cv = cross_val_score(model, x, y, cv=5, scoring="f1")
        print(f"F1 en validation croisée (5 folds) : {scores_cv.mean():.2f} (+/- {scores_cv.std():.2f})")

        os.makedirs("modeles", exist_ok=True)
        joblib.dump(model, "modeles/model_dropout.joblib")

        metadonnees = {"colonnes": list(x.columns)}
        with open("modeles/metadonnees.json", "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, indent=4, ensure_ascii=False)
