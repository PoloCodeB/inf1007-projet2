"""Lecture, nettoyage et variables du jeu de données."""

import os
from abc import ABC, abstractmethod

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from exceptions import DonneeInvalideError

DOSSIER_PROJET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_DONNEES = os.path.join(DOSSIER_PROJET, "donnees", "etudiants.csv")

CIBLE = "dropout"
COLONNES_NUMERIQUES = ["age", "gpa_semester_1", "attendance_rate"]
COLONNES_CATEGORIELLES = ["tuition_up_to_date", "study_mode"]
COLONNES_ENTREE = COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES

DOMAINES = {"age": (15, 100), "gpa_semester_1": (0.0, 4.0), "attendance_rate": (0.0, 100.0)}

LIBELLES = {
    "age": "Âge",
    "gpa_semester_1": "Moyenne (GPA) du 1er semestre",
    "attendance_rate": "Taux de présence (%)",
    "tuition_up_to_date": "Frais de scolarité à jour",
    "study_mode": "Mode d'études",
}


class Variable(ABC):
    """Classe parente abstraite d'une variable d'entrée."""

    def __init__(self, nom: str, libelle: str) -> None:
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        return self._nom

    @property
    def libelle(self) -> str:
        return self._libelle

    @abstractmethod
    def valider(self, valeur: object) -> object:
        """Valide et convertit la saisie. Reçoit : valeur brute. Renvoie : valeur convertie. Lève : DonneeInvalideError."""

    @abstractmethod
    def vers_dict(self) -> dict:
        """Exporte la description pour les métadonnées JSON. Reçoit : rien. Renvoie : dict."""

    def __str__(self) -> str:
        return f"{self._libelle} ({self._nom})"


class VariableNumerique(Variable):
    """Variable numérique bornée par un intervalle [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float) -> None:
        super().__init__(nom, libelle)
        self._minimum = float(minimum)
        self._maximum = float(maximum)

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    def valider(self, valeur: object) -> float:
        """Valide un nombre dans [minimum, maximum]. Reçoit : str ou nombre. Renvoie : float. Lève : DonneeInvalideError."""
        if valeur is None or str(valeur).strip() == "":
            raise DonneeInvalideError(f"« {self._libelle} » est obligatoire.")
        try:
            nombre = float(str(valeur).replace(",", "."))
        except ValueError:
            raise DonneeInvalideError(f"« {self._libelle} » doit être un nombre valide.")
        if not (self._minimum <= nombre <= self._maximum):
            raise DonneeInvalideError(f"« {self._libelle} » doit être entre {self._minimum} et {self._maximum}.")
        return nombre

    def vers_dict(self) -> dict:
        return {"type": "numerique", "nom": self._nom, "libelle": self._libelle,
                "minimum": self._minimum, "maximum": self._maximum}


class VariableCategorielle(Variable):
    """Variable catégorielle restreinte à des modalités définies."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        super().__init__(nom, libelle)
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list[str]:
        return list(self._valeurs)

    def valider(self, valeur: object) -> str:
        """Valide une modalité autorisée. Reçoit : texte saisi. Renvoie : str nettoyée. Lève : DonneeInvalideError."""
        texte = "" if valeur is None else str(valeur).strip()
        if not texte:
            raise DonneeInvalideError(f"« {self._libelle} » est obligatoire.")
        if texte not in self._valeurs:
            raise DonneeInvalideError(f"« {self._libelle} » doit être parmi : {', '.join(self._valeurs)}.")
        return texte

    def vers_dict(self) -> dict:
        return {"type": "categorielle", "nom": self._nom, "libelle": self._libelle, "valeurs": self._valeurs}


def variable_depuis_dict(description: dict) -> Variable:
    """Reconstitue une Variable depuis un dict. Reçoit : description (dict). Renvoie : Variable. Lève : DonneeInvalideError."""
    genre = description.get("type")
    if genre == "numerique":
        return VariableNumerique(description["nom"], description["libelle"],
                                 description["minimum"], description["maximum"])
    if genre == "categorielle":
        return VariableCategorielle(description["nom"], description["libelle"], description["valeurs"])
    raise DonneeInvalideError(f"Type de variable inconnu : {genre}")


class JeuDeDonnees:
    """Charge, nettoie et sépare le jeu de données pour l'apprentissage."""

    def __init__(self, chemin: str = CHEMIN_DONNEES) -> None:
        self._chemin = chemin
        self._donnees: pd.DataFrame | None = None

    @property
    def donnees(self) -> pd.DataFrame:
        if self._donnees is None:
            raise DonneeInvalideError("Les données n'ont pas encore été chargées.")
        return self._donnees

    def charger(self) -> pd.DataFrame:
        """Lit le CSV et retient les colonnes utiles. Reçoit : rien. Renvoie : DataFrame. Lève : FileNotFoundError, DonneeInvalideError."""
        if not os.path.exists(self._chemin):
            raise FileNotFoundError(f"Fichier de données introuvable : {self._chemin}")
        with open(self._chemin, encoding="utf-8") as fichier:
            donnees = pd.read_csv(fichier)
        manquantes = set(COLONNES_ENTREE + [CIBLE]) - set(donnees.columns)
        if manquantes:
            raise DonneeInvalideError(f"Colonnes manquantes dans le CSV : {sorted(manquantes)}")
        self._donnees = donnees[COLONNES_ENTREE + [CIBLE]].copy()
        return self._donnees

    def nettoyer(self) -> int:
        """Filtre les lignes incomplètes ou aberrantes. Reçoit : rien. Renvoie : nb lignes retirées (int)."""
        donnees = self.donnees.dropna()
        for col, (mini, maxi) in DOMAINES.items():
            donnees = donnees[donnees[col].between(mini, maxi)]
        donnees = donnees[donnees[CIBLE].isin([0, 1])].copy()
        donnees["tuition_up_to_date"] = donnees["tuition_up_to_date"].map({1: "Oui", 0: "Non"})
        donnees = donnees.dropna()
        retirees = len(self.donnees) - len(donnees)
        self._donnees = donnees
        return retirees

    def separer(self, taille_test: float = 0.2, graine: int = 42) -> tuple:
        """Sépare en train/test stratifié. Reçoit : taille_test (float), graine (int). Renvoie : (X_tr, X_te, y_tr, y_te)."""
        return train_test_split(self.donnees[COLONNES_ENTREE], self.donnees[CIBLE],
                                test_size=taille_test, random_state=graine, stratify=self.donnees[CIBLE])

    def construire_variables(self, x_entrainement: pd.DataFrame) -> list[Variable]:
        """Crée les variables d'après les données d'entraînement. Reçoit : x_entrainement. Renvoie : list[Variable]."""
        variables: list[Variable] = []
        for col in COLONNES_NUMERIQUES:
            variables.append(VariableNumerique(col, LIBELLES[col],
                                               float(x_entrainement[col].min()), float(x_entrainement[col].max())))
        for col in COLONNES_CATEGORIELLES:
            valeurs = sorted(x_entrainement[col].unique().tolist())
            variables.append(VariableCategorielle(col, LIBELLES[col], valeurs))
        return variables

    def graphique_exploration(self, chemin: str) -> str:
        """Enregistre le graphique du taux d'abandon par GPA. Reçoit : chemin (str). Renvoie : chemin (str)."""
        tranches = pd.cut(self.donnees["gpa_semester_1"], bins=[0, 1, 2, 2.5, 3, 3.5, 4], include_lowest=True,
                          labels=["0–1", "1–2", "2–2,5", "2,5–3", "3–3,5", "3,5–4"])
        taux = self.donnees.groupby(tranches, observed=True)[CIBLE].mean() * 100

        fig, axe = plt.subplots(figsize=(7, 4))
        axe.bar(taux.index.astype(str), taux.values, color="steelblue", label="Taux d'abandon")
        axe.axhline(self.donnees[CIBLE].mean() * 100, color="red", linestyle="--", label="Moyenne globale")
        axe.set_title("Taux d'abandon selon la moyenne (GPA) du 1er semestre")
        axe.set_xlabel("Tranche de GPA")
        axe.set_ylabel("Étudiants ayant abandonné (%)")
        axe.legend()
        fig.tight_layout()

        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        fig.savefig(chemin)
        plt.close(fig)
        return chemin
