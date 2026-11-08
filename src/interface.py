"""Interface graphique Streamlit.

Elle appelle la logique de src/donnees.py et src/modele.py, elle ne la contient pas.
"""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from donnees import DOSSIER_PROJET, VariableCategorielle
from exceptions import DonneeInvalideError
from modele import Modele

CHEMIN_HISTORIQUE = os.path.join(DOSSIER_PROJET, "historique_predictions.csv")

AVERTISSEMENT = (
    "Cet outil est un exercice pédagogique (cours INF1007). Il ne doit pas servir à prendre "
    "une décision sur un étudiant : le modèle peut se tromper et ne connaît que quelques variables."
)


class Application:
    """Application Streamlit : contient le modèle et la liste de ses variables (composition)."""

    def __init__(self, modele: Modele, chemin_historique: str = CHEMIN_HISTORIQUE) -> None:
        self._modele = modele
        self._variables = modele.variables
        self._chemin_historique = chemin_historique

    def afficher(self) -> None:
        """Construit la page : avertissement, formulaire, résultat, graphique et historique."""
        st.title("Risque d'abandon scolaire")
        st.warning(AVERTISSEMENT)

        with st.form("formulaire"):
            st.subheader("Profil de l'étudiant")
            saisies = {variable.nom: self._champ(variable) for variable in self._variables}
            envoye = st.form_submit_button("Prédire", type="primary")

        if envoye:
            self._predire(saisies)

        self._afficher_historique()

    def _champ(self, variable) -> str:
        """Affiche le champ adapté au type de la variable et renvoie la saisie brute."""
        if isinstance(variable, VariableCategorielle):
            return st.selectbox(variable.libelle, variable.valeurs)
        return st.text_input(variable.libelle,
                             placeholder=f"Entre {variable.minimum:g} et {variable.maximum:g}")

    def _predire(self, saisies: dict) -> None:
        """Lance la prédiction et affiche le résultat, ou un message clair si une saisie est invalide."""
        try:
            classe, probabilite = self._modele.predire(saisies)
        except DonneeInvalideError as erreur:
            st.error(f"Saisie invalide : {erreur}")
            return

        if classe == 1:
            st.error(f"### Risque d'abandon élevé\nProbabilité estimée d'abandon : **{probabilite:.1%}**")
        else:
            st.success(f"### Risque d'abandon faible\nProbabilité estimée d'abandon : **{probabilite:.1%}**")

        figure = self._figure(probabilite)
        st.pyplot(figure)
        plt.close(figure)
        self._ajouter_historique(saisies, classe, probabilite)

    def _figure(self, probabilite: float) -> Figure:
        """Graphique Matplotlib : probabilité de la prédiction et importance des variables."""
        fig, (axe_proba, axe_importance) = plt.subplots(1, 2, figsize=(11, 4))

        axe_proba.barh(["Poursuite", "Abandon"], [(1 - probabilite) * 100, probabilite * 100],
                       color=["seagreen", "firebrick"])
        axe_proba.axvline(50, color="gray", linestyle="--", label="Seuil de décision (50 %)")
        axe_proba.set_xlim(0, 100)
        axe_proba.set_title("Probabilité estimée pour cet étudiant")
        axe_proba.set_xlabel("Probabilité (%)")
        axe_proba.legend(loc="lower right")

        importances = self._modele.importances().sort_values()
        libelles = {variable.nom: variable.libelle for variable in self._variables}
        axe_importance.barh([libelles[nom] for nom in importances.index], importances.values * 100,
                            color="steelblue")
        axe_importance.set_title("Poids de chaque variable dans le modèle")
        axe_importance.set_xlabel("Importance (%)")

        fig.tight_layout()
        return fig

    def _ajouter_historique(self, saisies: dict, classe: int, probabilite: float) -> None:
        """Ajoute la prédiction à la fin du fichier CSV d'historique (le crée au besoin)."""
        ligne = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **saisies,
                 "resultat": "Abandon" if classe == 1 else "Poursuite",
                 "probabilite_abandon": round(probabilite, 3)}
        nouveau_fichier = not os.path.exists(self._chemin_historique)
        with open(self._chemin_historique, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.DictWriter(fichier, fieldnames=list(ligne))
            if nouveau_fichier:
                ecrivain.writeheader()
            ecrivain.writerow(ligne)

    def _afficher_historique(self) -> None:
        """Affiche les prédictions passées, de la plus récente à la plus ancienne."""
        st.subheader("Historique des prédictions")
        if not os.path.exists(self._chemin_historique):
            st.info("Aucune prédiction pour l'instant.")
            return
        with open(self._chemin_historique, encoding="utf-8") as fichier:
            historique = pd.read_csv(fichier)
        libelles = {variable.nom: variable.libelle for variable in self._variables}
        libelles.update({"date": "Date", "resultat": "Résultat", "probabilite_abandon": "Probabilité d'abandon"})
        st.dataframe(historique.iloc[::-1].rename(columns=libelles), hide_index=True)
