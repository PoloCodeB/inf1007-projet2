"""Lecture, nettoyage et variables du jeu de données."""

import os
from abc import ABC, abstractmethod

import matplotlib

matplotlib.use("Agg")  # Pas de fenêtre : on enregistre le graphique dans un fichier.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from exceptions import DonneeInvalideError

DOSSIER_PROJET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_DONNEES = os.path.join(DOSSIER_PROJET, "donnees", "etudiants.csv")

CIBLE = "dropout"
COLONNES_NUMERIQUES = ["age", "gpa_semester_1", "attendance_rate"]
COLONNES_BINAIRES = ["tuition_up_to_date", "scholarship_holder", "part_time_job"]
COLONNES_CATEGORIELLES = COLONNES_BINAIRES + ["study_mode"]
COLONNES_ENTREE = COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES

# Domaine possible de chaque variable numérique : une valeur en dehors est aberrante.
DOMAINES = {"age": (15, 100), "gpa_semester_1": (0.0, 4.0), "attendance_rate": (0.0, 100.0)}

LIBELLES = {
    "age": "Âge",
    "gpa_semester_1": "Moyenne (GPA) du 1er semestre",
    "attendance_rate": "Taux de présence en cours (%)",
    "tuition_up_to_date": "Frais de scolarité à jour",
    "scholarship_holder": "Bénéficie d'une bourse",
    "part_time_job": "Travail à temps partiel",
    "study_mode": "Mode d'études",
}


class Variable(ABC):
    """Classe parente abstraite d'une variable d'entrée du modèle."""

    def __init__(self, nom: str, libelle: str) -> None:
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le jeu de données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché à l'utilisateur."""
        return self._libelle

    @abstractmethod
    def valider(self, valeur):
        """Valide une saisie et la renvoie convertie.

        Lève DonneeInvalideError si la saisie n'est pas acceptable.
        """

    @abstractmethod
    def vers_dict(self) -> dict:
        """Renvoie la description de la variable pour le fichier JSON de métadonnées."""

    def __str__(self) -> str:
        return f"{self._libelle} ({self._nom})"


class VariableNumerique(Variable):
    """Variable numérique bornée par un intervalle [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float) -> None:
        super().__init__(nom, libelle)
        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    def valider(self, valeur: str | float) -> float:
        """Convertit la saisie en float et vérifie qu'elle est dans l'intervalle.

        Reçoit un texte (saisie de l'interface) ou un nombre.
        Lève DonneeInvalideError si la saisie est vide, non numérique ou hors intervalle.
        """
        if valeur is None or str(valeur).strip() == "":
            raise DonneeInvalideError(f"« {self._libelle} » est vide.")
        try:
            nombre = float(str(valeur).replace(",", "."))
        except ValueError:
            raise DonneeInvalideError(f"« {self._libelle} » doit être un nombre (reçu : {valeur}).")
        if not self._minimum <= nombre <= self._maximum:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être entre {self._minimum} et {self._maximum} (reçu : {nombre})."
            )
        return nombre

    def vers_dict(self) -> dict:
        return {"type": "numerique", "nom": self._nom, "libelle": self._libelle,
                "minimum": self._minimum, "maximum": self._maximum}


class VariableCategorielle(Variable):
    """Variable qui prend une valeur parmi une liste fixe."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        super().__init__(nom, libelle)
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list[str]:
        return list(self._valeurs)

    def valider(self, valeur: str) -> str:
        """Vérifie que la saisie fait partie des valeurs possibles et la renvoie.

        Lève DonneeInvalideError si la saisie est vide ou inconnue.
        """
        if valeur is None or str(valeur).strip() == "":
            raise DonneeInvalideError(f"« {self._libelle} » est vide.")
        if valeur not in self._valeurs:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être l'une de ces valeurs : {', '.join(self._valeurs)}."
            )
        return valeur

    def vers_dict(self) -> dict:
        return {"type": "categorielle", "nom": self._nom, "libelle": self._libelle,
                "valeurs": self._valeurs}


def variable_depuis_dict(description: dict) -> Variable:
    """Recrée une Variable à partir de sa description JSON (voir vers_dict).

    Lève DonneeInvalideError si le type est inconnu.
    """
    if description["type"] == "numerique":
        return VariableNumerique(description["nom"], description["libelle"],
                                 description["minimum"], description["maximum"])
    if description["type"] == "categorielle":
        return VariableCategorielle(description["nom"], description["libelle"], description["valeurs"])
    raise DonneeInvalideError(f"Type de variable inconnu : {description['type']}")


class JeuDeDonnees:
    """Charge, nettoie et sépare le jeu de données des étudiants."""

    def __init__(self, chemin: str = CHEMIN_DONNEES) -> None:
        self._chemin = chemin
        self._donnees: pd.DataFrame | None = None

    @property
    def donnees(self) -> pd.DataFrame:
        """DataFrame chargé. Lève DonneeInvalideError s'il n'a pas encore été chargé."""
        if self._donnees is None:
            raise DonneeInvalideError("Les données n'ont pas encore été chargées.")
        return self._donnees

    def charger(self) -> pd.DataFrame:
        """Lit le CSV en ne gardant que les colonnes utiles.

        Lève FileNotFoundError si le fichier n'existe pas,
        DonneeInvalideError s'il manque une colonne attendue.
        """
        if not os.path.exists(self._chemin):
            raise FileNotFoundError(f"Fichier de données introuvable : {self._chemin}")
        with open(self._chemin, encoding="utf-8") as fichier:
            donnees = pd.read_csv(fichier)
        manquantes = set(COLONNES_ENTREE + [CIBLE]) - set(donnees.columns)
        if manquantes:
            raise DonneeInvalideError(f"Colonnes manquantes dans le CSV : {sorted(manquantes)}")
        self._donnees = donnees[COLONNES_ENTREE + [CIBLE]]
        return self._donnees

    def nettoyer(self) -> int:
        """Retire les lignes incomplètes ou aberrantes et rend les binaires lisibles (Oui/Non).

        Renvoie le nombre de lignes retirées.
        """
        donnees = self.donnees.dropna()
        for colonne, (minimum, maximum) in DOMAINES.items():
            donnees = donnees[donnees[colonne].between(minimum, maximum)]
        donnees = donnees[donnees[CIBLE].isin([0, 1])].copy()
        for colonne in COLONNES_BINAIRES:
            donnees[colonne] = donnees[colonne].map({1: "Oui", 0: "Non"})
        donnees = donnees.dropna()  # Binaires différents de 0/1.
        retirees = len(self.donnees) - len(donnees)
        self._donnees = donnees
        return retirees

    def separer(self, taille_test: float = 0.2, graine: int = 42) -> tuple:
        """Sépare en ensembles d'entraînement et de test, en gardant la proportion de la cible.

        Renvoie (x_entrainement, x_test, y_entrainement, y_test).
        """
        return train_test_split(self.donnees[COLONNES_ENTREE], self.donnees[CIBLE],
                                test_size=taille_test, random_state=graine, stratify=self.donnees[CIBLE])

    def construire_variables(self, x_entrainement: pd.DataFrame) -> list[Variable]:
        """Crée une Variable par colonne d'entrée, avec les bornes et valeurs vues à l'entraînement."""
        variables: list[Variable] = []
        for colonne in COLONNES_NUMERIQUES:
            variables.append(VariableNumerique(colonne, LIBELLES[colonne],
                                               float(x_entrainement[colonne].min()),
                                               float(x_entrainement[colonne].max())))
        for colonne in COLONNES_CATEGORIELLES:
            valeurs = sorted(x_entrainement[colonne].unique().tolist())
            variables.append(VariableCategorielle(colonne, LIBELLES[colonne], valeurs))
        return variables

    def graphique_exploration(self, chemin: str) -> str:
        """Enregistre le taux d'abandon selon la tranche de GPA et renvoie le chemin du fichier."""
        tranches = pd.cut(self.donnees["gpa_semester_1"], bins=[0, 1, 2, 2.5, 3, 3.5, 4], include_lowest=True,
                          labels=["0–1", "1–2", "2–2,5", "2,5–3", "3–3,5", "3,5–4"])
        taux = self.donnees.groupby(tranches, observed=True)[CIBLE].mean() * 100

        fig, axe = plt.subplots(figsize=(8, 5))
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
