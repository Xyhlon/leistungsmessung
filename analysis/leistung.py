from labtool_ex2 import Project
from sympy import exp, pi
import numpy as np
import pandas as pd
import os

# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false


def test_leistung_protokoll():
    # zLuft / cps zPapier / cps zKunststoff / cps zAlu0_8 / cps zAlu1_5 / cps
    gm = {
        "t": r"t",
        "U": r"U",
        "I": r"I",
        "E": r"E_{\mathrm{kin}}",
    }
    gv = {
        "t": r"\si{\second}",
        "U": r"\si{\volt}",
        "I": r"\si{\cm}",
        "E": r"\si{\mega\electronvolt}",
    }

    P = Project("Leistung", global_variables=gv, global_mapping=gm, font=13)
    P.output_dir = "./"
    P.figure.set_size_inches((8, 6))
    ax = P.figure.add_subplot()
    M1 = 1.5 * 0.24
    M2 = 1.5 * 0.6
    m1 = 0.5 * 120
    m2 = 0.5 * 240
    Pm1 = 0.5 * 1 + 0.5 * 120
    Pm2 = 0.5 * 1 + 0.5 * 240

    # A1 qualitative Absorption Untersuchung mit und ohne Abschirmung
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe1.csv")
    P.load_data(filepath, loadnew=True)
    print(P.data)


if __name__ == "__main__":
    test_leistung_protokoll()
