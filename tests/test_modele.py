"""Tests de la logique du modèle (hors interface)."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from donnees import VariableCategorielle, VariableNumerique
from exceptions import DonneeInvalideError
from modele import NOM_METADONNEES, Modele, f1_numpy

VARIABLES = [
    VariableNumerique("gpa_semester_1", "GPA", 0.0, 4.0),
    VariableCategorielle("study_mode", "Mode", ["Full-time", "Online"]),
]
SAISIE_VALIDE = {"gpa_semester_1": "3.2", "study_mode": "Online"}


def modele_entraine() -> tuple[Modele, pd.DataFrame, pd.Series]:
    """Entraîne un modèle sur 400 lignes synthétiques : GPA bas → abandon."""
    gpa = np.linspace(0, 4, 400)
    x = pd.DataFrame({"gpa_semester_1": gpa, "study_mode": ["Full-time", "Online"] * 200})
    y = pd.Series((gpa < 2).astype(int))
    modele = Modele(VARIABLES)
    modele.entrainer(x, y)
    return modele, x, y


# --- F1 calculé avec NumPy ---

def test_f1_numpy_egal_a_scikit_learn():
    generateur = np.random.default_rng(0)
    y_vrai = generateur.integers(0, 2, 1000)
    y_predit = generateur.integers(0, 2, 1000)
    assert f1_numpy(y_vrai, y_predit) == pytest.approx(f1_score(y_vrai, y_predit))


def test_f1_numpy_prediction_parfaite():
    assert f1_numpy([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)


def test_f1_numpy_aucun_positif_vaut_zero():
    assert f1_numpy([0, 0, 0], [0, 0, 0]) == 0.0


def test_f1_numpy_longueurs_differentes():
    with pytest.raises(ValueError):
        f1_numpy([0, 1], [0, 1, 1])


# --- Modèle ---

def test_predire_avant_entrainement():
    with pytest.raises(RuntimeError):
        Modele(VARIABLES).predire(SAISIE_VALIDE)


def test_predire_renvoie_classe_et_probabilite():
    modele, _, _ = modele_entraine()
    classe, probabilite = modele.predire(SAISIE_VALIDE)
    assert classe == 0  # GPA élevé → pas d'abandon
    assert 0.0 <= probabilite <= 1.0


def test_predire_saisie_invalide():
    modele, _, _ = modele_entraine()
    with pytest.raises(DonneeInvalideError):
        modele.predire({"gpa_semester_1": "abc", "study_mode": "Online"})


def test_evaluer_compare_a_la_reference():
    modele, x, y = modele_entraine()
    resultats = modele.evaluer(x, y)
    assert resultats["f1"] > resultats["f1_reference"]
    assert resultats["f1_reference"] == 0.0  # La classe majoritaire ne prédit jamais l'abandon.


def test_sauvegarder_puis_charger(tmp_path):
    modele, x, y = modele_entraine()
    modele.evaluer(x, y)
    modele.sauvegarder(tmp_path)

    with open(os.path.join(tmp_path, NOM_METADONNEES), encoding="utf-8") as fichier:
        metadonnees = json.load(fichier)
    assert metadonnees["colonnes"] == ["gpa_semester_1", "study_mode"]
    assert metadonnees["variables"][1]["valeurs"] == ["Full-time", "Online"]

    copie = Modele.charger(tmp_path)
    assert copie.predire(SAISIE_VALIDE) == modele.predire(SAISIE_VALIDE)
    assert copie.resultats["f1"] == pytest.approx(modele.resultats["f1"])


def test_charger_dossier_vide(tmp_path):
    with pytest.raises(FileNotFoundError):
        Modele.charger(tmp_path)
