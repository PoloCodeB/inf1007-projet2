"""Interface graphique Streamlit pour la prédiction du risque d'abandon scolaire."""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from donnees import DOSSIER_PROJET, Variable, VariableCategorielle
from exceptions import DonneeInvalideError
from modele import Modele

CHEMIN_HISTORIQUE = os.path.join(DOSSIER_PROJET, "historique_predictions.csv")

AVERTISSEMENT = (
    "Cet outil est un exercice pédagogique (cours INF1007). Il ne doit pas servir à prendre "
    "une décision sur un étudiant : le modèle peut se tromper."
)


class Application:
    """Interface Streamlit contenant le modèle et ses variables d'entrée (composition)."""

    def __init__(self, modele: Modele, chemin_historique: str = CHEMIN_HISTORIQUE) -> None:
        self._modele = modele
        self._variables = modele.variables
        self._chemin_historique = chemin_historique

    def afficher(self) -> None:
        """Affiche la page Streamlit : avertissement, formulaire, résultat et historique."""
        st.title("Risque d'abandon scolaire")
        st.warning(AVERTISSEMENT)

        with st.form("formulaire"):
            st.subheader("Profil de l'étudiant")
            saisies = {v.nom: self._champ(v) for v in self._variables}
            if st.form_submit_button("Prédire", type="primary"):
                self._predire(saisies)

        self._afficher_historique()

    def _champ(self, variable: Variable) -> str:
        """Affiche le champ adapté au type de la variable. Reçoit : Variable. Renvoie : str."""
        if isinstance(variable, VariableCategorielle):
            return st.selectbox(variable.libelle, variable.valeurs)
        return st.text_input(variable.libelle, placeholder=f"Entre {variable.minimum:g} et {variable.maximum:g}")

    def _predire(self, saisies: dict) -> None:
        """Lance la prédiction et affiche le verdict, le graphique et sauvegarde l'historique."""
        try:
            classe, probabilite = self._modele.predire(saisies)
        except DonneeInvalideError as erreur:
            st.error(f"Saisie invalide : {erreur}")
            return

        if classe == 1:
            st.error(f"### Risque d'abandon élevé\nProbabilité estimée : **{probabilite:.1%}**")
        else:
            st.success(f"### Risque d'abandon faible\nProbabilité estimée : **{probabilite:.1%}**")

        fig = self._figure(probabilite)
        st.pyplot(fig)
        plt.close(fig)
        self._ajouter_historique(saisies, classe, probabilite)

    def _figure(self, probabilite: float) -> Figure:
        """Graphique horizontal comparant les probabilités. Reçoit : probabilite (float). Renvoie : Figure."""
        fig, axe = plt.subplots(figsize=(6, 2.2))
        axe.barh(["Poursuite", "Abandon"], [(1 - probabilite) * 100, probabilite * 100],
                 color=["seagreen", "firebrick"])
        axe.axvline(50, color="gray", linestyle="--", label="Seuil (50 %)")
        axe.set_xlim(0, 100)
        axe.set_title("Probabilité estimée pour cet étudiant")
        axe.set_xlabel("Probabilité (%)")
        axe.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def _ajouter_historique(self, saisies: dict, classe: int, probabilite: float) -> None:
        """Ajoute la prédiction au CSV d'historique. Reçoit : saisies (dict), classe (int), probabilite (float)."""
        ligne = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **saisies,
            "resultat": "Abandon" if classe == 1 else "Poursuite",
            "probabilite_abandon": round(probabilite, 3),
        }
        nouveau = not os.path.exists(self._chemin_historique)
        with open(self._chemin_historique, "a", newline="", encoding="utf-8") as f:
            ecrivain = csv.DictWriter(f, fieldnames=list(ligne))
            if nouveau:
                ecrivain.writeheader()
            ecrivain.writerow(ligne)

    def _afficher_historique(self) -> None:
        """Affiche le tableau des prédictions passées. Reçoit : rien. Renvoie : None."""
        st.subheader("Historique des prédictions")
        if not os.path.exists(self._chemin_historique):
            st.info("Aucune prédiction pour l'instant.")
            return
        with open(self._chemin_historique, encoding="utf-8") as f:
            historique = pd.read_csv(f)
        st.dataframe(historique.iloc[::-1], hide_index=True)
