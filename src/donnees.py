"""Lecture, nettoyage et variables du jeu de données.

À compléter : chargement du CSV avec pandas, exploration
(types, valeurs manquantes, valeurs aberrantes), classes Variable.
"""

class Variable:
    """Classe parente des variables d'entrée."""

    def __init__(self, nom: str) -> None:
        self._nom = nom

    def valider(self, valeur):
        """Valide une saisie. Redéfinie par chaque classe enfant."""
        raise NotImplementedError


class VariableNumerique(Variable):
    """Variable numérique : à compléter (intervalle autorisé, validation)."""

    def valider(self, valeur):
        raise NotImplementedError


class VariableCategorielle(Variable):
    """Variable catégorielle : à compléter (valeurs possibles, validation)."""

    def valider(self, valeur):
        raise NotImplementedError


class JeuDeDonnees:
    """À compléter : chargement, nettoyage et séparation entraînement/test."""
