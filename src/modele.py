"""Entraînement, évaluation, prédiction et sauvegarde du modèle.

Lancer `python src/modele.py` pour entraîner le modèle une fois
et l'enregistrer dans modeles/ (l'interface ne fait que le charger).
"""

import json
import os
import time
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from donnees import DOSSIER_PROJET, Variable, VariableCategorielle, variable_depuis_dict

DOSSIER_MODELES = os.path.join(DOSSIER_PROJET, "modeles")
NOM_MODELE = "modele_abandon.joblib"
NOM_METADONNEES = "metadonnees.json"


def chronometrer(fonction):
    """Décorateur qui affiche la durée d'exécution de la fonction décorée."""

    @wraps(fonction)
    def enveloppe(*args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        print(f"{fonction.__name__} : {time.perf_counter() - debut:.2f} s")
        return resultat

    return enveloppe


def f1_numpy(y_vrai, y_predit) -> float:
    """Calcule le score F1 de la classe positive (1) avec NumPy, sans boucle.

    Reçoit deux séquences de 0 et de 1 de même longueur.
    Renvoie un float entre 0 et 1 (0 s'il n'y a aucun positif, ni réel ni prédit).
    Lève ValueError si les longueurs diffèrent.
    """
    y_vrai = np.asarray(y_vrai)
    y_predit = np.asarray(y_predit)
    if y_vrai.shape != y_predit.shape:
        raise ValueError("y_vrai et y_predit doivent avoir la même longueur.")
    vrais_positifs = np.sum((y_vrai == 1) & (y_predit == 1))
    faux_positifs = np.sum((y_vrai == 0) & (y_predit == 1))
    faux_negatifs = np.sum((y_vrai == 1) & (y_predit == 0))
    denominateur = 2 * vrais_positifs + faux_positifs + faux_negatifs
    if denominateur == 0:
        return 0.0
    return float(2 * vrais_positifs / denominateur)


class Modele:
    """Arbre de décision qui prédit le risque d'abandon d'un étudiant."""

    def __init__(self, variables: list[Variable]) -> None:
        self._variables = list(variables)
        self._arbre: DecisionTreeClassifier | None = None
        self._colonnes_encodees: list[str] = []
        self._reference: DummyClassifier | None = None
        self._resultats: dict = {}

    @property
    def variables(self) -> list[Variable]:
        """Variables d'entrée du modèle (copie de la liste)."""
        return list(self._variables)

    @property
    def resultats(self) -> dict:
        """Métriques calculées par evaluer() (ou lues dans les métadonnées)."""
        return dict(self._resultats)

    @property
    def colonnes(self) -> list[str]:
        """Noms des colonnes d'entrée, dans l'ordre des variables."""
        return [variable.nom for variable in self._variables]

    def _encoder(self, x: pd.DataFrame) -> pd.DataFrame:
        """Remplace chaque variable catégorielle par une colonne 0/1 par valeur possible.

        Exemple : study_mode devient study_mode_Full-time, study_mode_Online, study_mode_Part-time.
        Les colonnes sont remises dans l'ordre de l'entraînement ; une colonne absente
        (ex. une seule ligne à prédire) est ajoutée avec des 0.
        """
        categorielles = [v.nom for v in self._variables if isinstance(v, VariableCategorielle)]
        encode = pd.get_dummies(x[self.colonnes], columns=categorielles, dtype=int)
        return encode.reindex(columns=self._colonnes_encodees, fill_value=0)

    @chronometrer
    def entrainer(self, x_entrainement: pd.DataFrame, y_entrainement: pd.Series) -> None:
        """Entraîne l'arbre de décision et la référence triviale (classe majoritaire)."""
        categorielles = [v.nom for v in self._variables if isinstance(v, VariableCategorielle)]
        self._colonnes_encodees = list(
            pd.get_dummies(x_entrainement[self.colonnes], columns=categorielles, dtype=int).columns
        )
        # Profondeur et taille minimale des feuilles limitées pour éviter le surapprentissage ;
        # class_weight="balanced" compense la répartition 68 % / 32 % de la cible.
        self._arbre = DecisionTreeClassifier(max_depth=5, min_samples_leaf=200,
                                             class_weight="balanced", random_state=42)
        self._arbre.fit(self._encoder(x_entrainement), y_entrainement)
        self._reference = DummyClassifier(strategy="most_frequent")
        self._reference.fit(x_entrainement[self.colonnes], y_entrainement)

    def _verifier_entraine(self) -> DecisionTreeClassifier:
        """Renvoie l'arbre. Lève RuntimeError si le modèle n'est pas entraîné."""
        if self._arbre is None:
            raise RuntimeError("Le modèle n'est pas entraîné ni chargé.")
        return self._arbre

    def evaluer(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Calcule F1 et exactitude sur l'ensemble de test, pour le modèle et la référence.

        Renvoie un dict avec les métriques et la matrice de confusion.
        Lève RuntimeError si le modèle n'est pas entraîné.
        """
        arbre = self._verifier_entraine()
        y_predit = arbre.predict(self._encoder(x_test))
        y_reference = self._reference.predict(x_test[self.colonnes])
        self._resultats = {
            "metrique": "F1 (classe abandon)",
            "f1": f1_numpy(y_test, y_predit),
            "exactitude": float(accuracy_score(y_test, y_predit)),
            "f1_reference": f1_numpy(y_test, y_reference),
            "exactitude_reference": float(accuracy_score(y_test, y_reference)),
            "matrice_confusion": confusion_matrix(y_test, y_predit).tolist(),
        }
        return self.resultats

    def predire(self, saisies: dict) -> tuple[int, float]:
        """Prédit l'abandon pour un étudiant.

        Reçoit un dict {nom de colonne: saisie brute}.
        Renvoie (classe prédite 0/1, probabilité d'abandon entre 0 et 1).
        Lève DonneeInvalideError si une saisie est invalide,
        RuntimeError si le modèle n'est pas entraîné.
        """
        arbre = self._verifier_entraine()
        valeurs = {v.nom: v.valider(saisies.get(v.nom)) for v in self._variables}
        ligne = self._encoder(pd.DataFrame([valeurs]))
        classe = int(arbre.predict(ligne)[0])
        probabilite = float(arbre.predict_proba(ligne)[0][1])
        return classe, probabilite

    def importances(self) -> pd.Series:
        """Importance de chaque variable d'origine (somme de ses colonnes 0/1), triée.

        Lève RuntimeError si le modèle n'est pas entraîné.
        """
        arbre = self._verifier_entraine()
        par_colonne = pd.Series(arbre.feature_importances_, index=self._colonnes_encodees)
        importances = {}
        for nom in self.colonnes:
            # Colonne numérique : même nom ; colonne encodée : « nom_valeur ».
            colonnes = [c for c in self._colonnes_encodees if c == nom or c.startswith(nom + "_")]
            importances[nom] = float(par_colonne[colonnes].sum())
        return pd.Series(importances).sort_values(ascending=False)

    def sauvegarder(self, dossier: str = DOSSIER_MODELES) -> None:
        """Enregistre l'arbre (joblib) et ses métadonnées (JSON) dans le dossier."""
        arbre = self._verifier_entraine()
        os.makedirs(dossier, exist_ok=True)
        joblib.dump(arbre, os.path.join(dossier, NOM_MODELE))
        metadonnees = {
            "colonnes": self.colonnes,
            "colonnes_encodees": self._colonnes_encodees,
            "variables": [variable.vers_dict() for variable in self._variables],
            "resultats": self._resultats,
        }
        with open(os.path.join(dossier, NOM_METADONNEES), "w", encoding="utf-8") as fichier:
            json.dump(metadonnees, fichier, indent=4, ensure_ascii=False)

    @classmethod
    def charger(cls, dossier: str = DOSSIER_MODELES) -> "Modele":
        """Recrée un Modele à partir des fichiers enregistrés par sauvegarder().

        Lève FileNotFoundError si le modèle ou les métadonnées sont absents.
        """
        chemin_modele = os.path.join(dossier, NOM_MODELE)
        chemin_metadonnees = os.path.join(dossier, NOM_METADONNEES)
        for chemin in (chemin_modele, chemin_metadonnees):
            if not os.path.exists(chemin):
                raise FileNotFoundError(f"Fichier introuvable : {chemin}. Lancer d'abord `python src/modele.py`.")
        with open(chemin_metadonnees, encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)
        modele = cls([variable_depuis_dict(d) for d in metadonnees["variables"]])
        modele._arbre = joblib.load(chemin_modele)
        modele._colonnes_encodees = metadonnees["colonnes_encodees"]
        modele._resultats = metadonnees["resultats"]
        return modele


def entrainer_et_sauvegarder() -> dict:
    """Prépare les données, entraîne, évalue et sauvegarde le modèle. Renvoie les résultats."""
    from donnees import JeuDeDonnees  # Import local : seulement utile pour l'entraînement.

    jeu = JeuDeDonnees()
    jeu.charger()
    print(f"Lignes retirées au nettoyage : {jeu.nettoyer()}")
    jeu.graphique_exploration(os.path.join(DOSSIER_PROJET, "docs", "figures", "exploration.png"))

    x_entrainement, x_test, y_entrainement, y_test = jeu.separer()
    modele = Modele(jeu.construire_variables(x_entrainement))
    modele.entrainer(x_entrainement, y_entrainement)
    resultats = modele.evaluer(x_test, y_test)
    modele.sauvegarder()
    return resultats


if __name__ == "__main__":
    resultats = entrainer_et_sauvegarder()
    print(f"F1 modèle : {resultats['f1']:.3f}  (référence : {resultats['f1_reference']:.3f})")
    print(f"Exactitude modèle : {resultats['exactitude']:.3f}  (référence : {resultats['exactitude_reference']:.3f})")
    print(f"Matrice de confusion [[VN, FP], [FN, VP]] : {resultats['matrice_confusion']}")
