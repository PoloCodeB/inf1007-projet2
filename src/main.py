"""Point d'entrée de l'application.

`python src/main.py` relance automatiquement le fichier avec Streamlit
(équivalent à `streamlit run src/main.py`).
"""

import sys

import streamlit as st
from streamlit import runtime
from streamlit.web import cli

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
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(cli.main())
