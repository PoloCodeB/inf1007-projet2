"""Entraînement, évaluation, prédiction et sauvegarde du modèle."""

import json
import os
import time
from collections.abc import Callable, Sequence
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


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur mesurant la durée d'une fonction. Reçoit : fonction. Renvoie : enveloppe."""
    @wraps(fonction)
    def enveloppe(*args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        print(f"{fonction.__name__} : {time.perf_counter() - debut:.2f} s")
        return resultat

    return enveloppe


def f1_numpy(y_vrai: Sequence, y_predit: Sequence) -> float:
    """Calcule le F1 vectorisé avec NumPy sans boucle. Reçoit : y_vrai, y_predit. Renvoie : float. Lève : ValueError."""
    vrai, pred = np.asarray(y_vrai), np.asarray(y_predit)
    if vrai.shape != pred.shape:
        raise ValueError("y_vrai et y_predit doivent avoir la même longueur.")
    vp = np.sum((vrai == 1) & (pred == 1))
    fp = np.sum((vrai == 0) & (pred == 1))
    fn = np.sum((vrai == 1) & (pred == 0))
    denominateur = 2 * vp + fp + fn
    return float(2 * vp / denominateur) if denominateur > 0 else 0.0


class Modele:
    """Modèle d'arbre de décision pour prédire le risque d'abandon scolaire."""

    def __init__(self, variables: list[Variable]) -> None:
        self._variables = list(variables)
        self._arbre: DecisionTreeClassifier | None = None
        self._colonnes_encodees: list[str] = []
        self._resultats: dict = {}

    @property
    def variables(self) -> list[Variable]:
        return list(self._variables)

    @property
    def resultats(self) -> dict:
        return dict(self._resultats)

    @property
    def colonnes(self) -> list[str]:
        return [v.nom for v in self._variables]

    def _encoder(self, x: pd.DataFrame) -> pd.DataFrame:
        """Encode les variables catégorielles en indicatrices binaires (0/1)."""
        cats = [v.nom for v in self._variables if isinstance(v, VariableCategorielle)]
        encode = pd.get_dummies(x[self.colonnes], columns=cats, dtype=int)
        return encode.reindex(columns=self._colonnes_encodees, fill_value=0)

    def _verifier_entraine(self) -> DecisionTreeClassifier:
        """Vérifie l'entraînement. Renvoie : DecisionTreeClassifier. Lève : RuntimeError."""
        if self._arbre is None:
            raise RuntimeError("Le modèle n'est pas entraîné ni chargé.")
        return self._arbre

    @chronometrer
    def entrainer(self, x_entrainement: pd.DataFrame, y_entrainement: pd.Series) -> None:
        """Entraîne l'arbre de décision. Reçoit : x_entrainement, y_entrainement. Renvoie : None."""
        cats = [v.nom for v in self._variables if isinstance(v, VariableCategorielle)]
        self._colonnes_encodees = list(pd.get_dummies(x_entrainement[self.colonnes], columns=cats, dtype=int).columns)
        self._arbre = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=200, class_weight="balanced", random_state=42
        )
        self._arbre.fit(self._encoder(x_entrainement), y_entrainement)

    def evaluer(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Évalue le modèle et la référence triviale. Reçoit : x_test, y_test. Renvoie : dict des métriques."""
        arbre = self._verifier_entraine()
        y_predit = arbre.predict(self._encoder(x_test))
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(x_test[self.colonnes], y_test)
        y_ref = dummy.predict(x_test[self.colonnes])
        self._resultats = {
            "metrique": "F1 (classe abandon)",
            "f1": f1_numpy(y_test, y_predit),
            "exactitude": float(accuracy_score(y_test, y_predit)),
            "f1_reference": f1_numpy(y_test, y_ref),
            "exactitude_reference": float(accuracy_score(y_test, y_ref)),
            "matrice_confusion": confusion_matrix(y_test, y_predit).tolist(),
        }
        return self.resultats

    def predire(self, saisies: dict) -> tuple[int, float]:
        """Prédit le risque d'abandon. Reçoit : saisies (dict). Renvoie : (classe, probabilite). Lève : DonneeInvalideError, RuntimeError."""
        arbre = self._verifier_entraine()
        valeurs = {v.nom: v.valider(saisies.get(v.nom)) for v in self._variables}
        ligne = self._encoder(pd.DataFrame([valeurs]))
        classe = int(arbre.predict(ligne)[0])
        probabilite = float(arbre.predict_proba(ligne)[0][1])
        return classe, probabilite

    def sauvegarder(self, dossier: str = DOSSIER_MODELES) -> None:
        """Sauvegarde modèle joblib et métadonnées JSON. Reçoit : dossier (str). Renvoie : None. Lève : RuntimeError."""
        arbre = self._verifier_entraine()
        os.makedirs(dossier, exist_ok=True)
        joblib.dump(arbre, os.path.join(dossier, NOM_MODELE))
        metadonnees = {
            "colonnes": self.colonnes,
            "colonnes_encodees": self._colonnes_encodees,
            "variables": [v.vers_dict() for v in self._variables],
            "resultats": self._resultats,
        }
        with open(os.path.join(dossier, NOM_METADONNEES), "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, indent=4, ensure_ascii=False)

    @classmethod
    def charger(cls, dossier: str = DOSSIER_MODELES) -> "Modele":
        """Charge le modèle et ses métadonnées. Reçoit : dossier. Renvoie : Modele. Lève : FileNotFoundError."""
        ch_mod = os.path.join(dossier, NOM_MODELE)
        ch_meta = os.path.join(dossier, NOM_METADONNEES)
        for ch in (ch_mod, ch_meta):
            if not os.path.exists(ch):
                raise FileNotFoundError(f"Fichier introuvable : {ch}. Exécutez `python src/modele.py`.")
        with open(ch_meta, encoding="utf-8") as f:
            meta = json.load(f)
        modele = cls([variable_depuis_dict(d) for d in meta["variables"]])
        modele._arbre = joblib.load(ch_mod)
        modele._colonnes_encodees = meta["colonnes_encodees"]
        modele._resultats = meta["resultats"]
        return modele


def entrainer_et_sauvegarder() -> dict:
    """Prépare les données, entraîne, évalue et sauvegarde le modèle. Reçoit : rien. Renvoie : dict."""
    from donnees import JeuDeDonnees

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
    print(f"F1 modèle : {resultats['f1']:.3f} (référence : {resultats['f1_reference']:.3f})")
    print(f"Exactitude modèle : {resultats['exactitude']:.3f} (référence : {resultats['exactitude_reference']:.3f})")
    print(f"Matrice de confusion : {resultats['matrice_confusion']}")
