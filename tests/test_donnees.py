"""Tests de la logique des données (hors interface)."""

import os

import pandas as pd
import pytest

from donnees import (
    JeuDeDonnees,
    VariableCategorielle,
    VariableNumerique,
    variable_depuis_dict,
)
from exceptions import DonneeInvalideError


def creer_csv(dossier, nb_lignes: int = 20) -> str:
    """Écrit un petit CSV valide (moitié abandon, moitié non) et renvoie son chemin."""
    donnees = pd.DataFrame({
        "age": [18 + i % 10 for i in range(nb_lignes)],
        "gpa_semester_1": [round(i % 5 * 0.8, 1) for i in range(nb_lignes)],
        "attendance_rate": [50.0 + i for i in range(nb_lignes)],
        "tuition_up_to_date": [i % 2 for i in range(nb_lignes)],
        "scholarship_holder": [1] * nb_lignes,
        "part_time_job": [0] * nb_lignes,
        "study_mode": ["Full-time", "Online"] * (nb_lignes // 2),
        "dropout": [0, 1] * (nb_lignes // 2),
    })
    chemin = os.path.join(dossier, "test.csv")
    donnees.to_csv(chemin, index=False)
    return chemin


# --- Variables ---

def test_numerique_accepte_texte_avec_virgule():
    gpa = VariableNumerique("gpa_semester_1", "GPA", 0.0, 4.0)
    assert gpa.valider("2,5") == pytest.approx(2.5)


def test_numerique_accepte_les_bornes_exactes():
    age = VariableNumerique("age", "Âge", 15, 30)
    assert age.valider("15") == 15
    assert age.valider(30) == 30


@pytest.mark.parametrize("saisie", ["", "   ", None, "abc", "14.9", "31"])
def test_numerique_refuse_vide_texte_et_hors_intervalle(saisie):
    age = VariableNumerique("age", "Âge", 15, 30)
    with pytest.raises(DonneeInvalideError):
        age.valider(saisie)


def test_categorielle_valide_et_refuse():
    mode = VariableCategorielle("study_mode", "Mode", ["Full-time", "Online", "Part-time"])
    assert mode.valider("Online") == "Online"
    with pytest.raises(DonneeInvalideError):
        mode.valider("Distance")


def test_str_affiche_libelle_et_nom():
    assert str(VariableNumerique("age", "Âge", 15, 30)) == "Âge (age)"


def test_variable_recreee_depuis_dict():
    origine = VariableCategorielle("study_mode", "Mode", ["Full-time", "Online"])
    copie = variable_depuis_dict(origine.vers_dict())
    assert isinstance(copie, VariableCategorielle)
    assert copie.vers_dict() == origine.vers_dict()


# --- Jeu de données ---

def test_charger_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        JeuDeDonnees(os.path.join(tmp_path, "absent.csv")).charger()


def test_charger_colonne_manquante(tmp_path):
    chemin = os.path.join(tmp_path, "incomplet.csv")
    pd.DataFrame({"age": [20]}).to_csv(chemin, index=False)
    with pytest.raises(DonneeInvalideError):
        JeuDeDonnees(chemin).charger()


def test_nettoyer_retire_aberrants_et_convertit_binaires(tmp_path):
    jeu = JeuDeDonnees(creer_csv(tmp_path))
    jeu.charger()
    jeu.donnees.loc[0, "gpa_semester_1"] = 9.0   # GPA impossible
    jeu.donnees.loc[1, "attendance_rate"] = None  # valeur manquante
    assert jeu.nettoyer() == 2
    assert set(jeu.donnees["tuition_up_to_date"]) == {"Oui", "Non"}


def test_separer_est_reproductible_et_stratifie(tmp_path):
    jeu = JeuDeDonnees(creer_csv(tmp_path))
    jeu.charger()
    jeu.nettoyer()
    _, x_test_1, _, y_test_1 = jeu.separer()
    _, x_test_2, _, _ = jeu.separer()
    assert list(x_test_1.index) == list(x_test_2.index)
    assert y_test_1.mean() == pytest.approx(0.5)


def test_construire_variables_utilise_les_bornes_d_entrainement(tmp_path):
    jeu = JeuDeDonnees(creer_csv(tmp_path))
    jeu.charger()
    jeu.nettoyer()
    variables = {v.nom: v for v in jeu.construire_variables(jeu.donnees)}
    assert variables["age"].minimum == 18
    assert variables["age"].maximum == 27
    assert variables["study_mode"].valeurs == ["Full-time", "Online"]


def test_graphique_exploration_cree_le_fichier(tmp_path):
    jeu = JeuDeDonnees(creer_csv(tmp_path))
    jeu.charger()
    jeu.nettoyer()
    chemin = jeu.graphique_exploration(os.path.join(tmp_path, "figures", "exploration.png"))
    assert os.path.exists(chemin)
