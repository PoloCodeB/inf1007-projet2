"""Point d'entrée de l'application.

Lancer avec : streamlit run src/main.py
"""

import streamlit as st

from interface import Application
from modele import Modele


@st.cache_resource
def charger_modele() -> Modele:
    """Charge le modèle sauvegardé une seule fois pour toute la session Streamlit."""
    return Modele.charger()


def main() -> None:
    """Charge le modèle et affiche l'interface (message clair si le modèle est absent)."""
    st.set_page_config(page_title="Risque d'abandon", page_icon="🎓")
    try:
        modele = charger_modele()
    except FileNotFoundError as erreur:
        st.error(f"{erreur}")
        return
    Application(modele).afficher()


if __name__ == "__main__":
    main()
