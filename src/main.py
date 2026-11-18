"""Point d'entrée de l'application Streamlit.

Lancement : streamlit run src/main.py
"""

import streamlit as st

from interface import Application
from modele import Modele


@st.cache_resource
def charger_modele() -> Modele:
    """Charge le modèle sérialisé pour la session Streamlit.

    Reçoit : rien.
    Renvoie : instance prête de Modele.
    Lève : FileNotFoundError si les fichiers de sauvegarde sont introuvables.
    """
    return Modele.charger()


def main() -> None:
    """Point d'entrée principal initialisant la page et l'application.

    Reçoit : rien.
    Renvoie : None.
    Lève : rien.
    """
    st.set_page_config(page_title="Risque d'abandon", page_icon="🎓")
    try:
        modele = charger_modele()
    except FileNotFoundError as erreur:
        st.error(f"{erreur}")
        return
    Application(modele).afficher()


if __name__ == "__main__":
    main()
