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
        "a": r"a",
        "b": r"b",
        "p": r"P",
        "E": r"E_{\mathrm{kin}}",
    }
    gv = {
        "t": r"\si{\second}",
        "U": r"\si{\volt}",
        "I": r"\si{\ampere}",
        "E": r"\si{\mega\electronvolt}",
        "a": r"\si{\volt}",
        "p": r"\si{\watt}",
        "b": r"\si{\ohm}",
    }

    P = Project("Leistung", global_variables=gv, global_mapping=gm, font=13)
    P.output_dir = "./"
    P.figure.set_size_inches((8, 6))
    ax = P.figure.add_subplot()
    Im1 = 0.015 * 0.24
    Im2 = 0.015 * 0.6
    Um1 = 0.005 * 120
    Um2 = 0.005 * 240
    Pm1 = 0.005 * 1 * 120
    Pm2 = 0.005 * 1 * 240

    # A1
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe1.csv")
    P.load_data(filepath, loadnew=True)
    # Setting uncertainties
    P.data["dU"] = P.data["U"]
    P.data["dU"][P.data["U"] < 121] = Um1
    P.data["dU"][P.data["U"] > 121] = Um2
    P.data["dI"] = P.data["I"]
    P.data["dI"][P.data["U"] < 121] = Im1
    P.data["dI"][P.data["U"] > 121] = Im2
    P.data["dp"] = P.data["p"]
    P.data["dp"][P.data["U"] < 121] = Pm1
    P.data["dp"][P.data["U"] > 121] = Pm2

    p = a * I + b * I**2

    P.plot_data(
        ax,
        I,
        p,
        label="Gemessene Daten",
        style="#1cb2f5",
        errors=True,
    )

    P.plot_fit(
        axes=ax,
        x=I,
        y=p,
        eqn=p,
        style=r"#1cb2f5",
        label="Strom",
        offset=[0, 30],
        use_all_known=False,
        guess={"a": 10, "b": 18},
        bounds=[
            {"name": "b", "min": 000, "max": 9000},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        # scale_covar=True,
    )
    print(P.data)

    ax.set_title(f"Leistungskennlinie von Strom")
    P.ax_legend_all(loc=4)
    ax = P.savefig(f"pIkennlinie.pdf")

    p = a * U + b * U**2

    P.plot_data(
        ax,
        U,
        p,
        label="Gemessene Daten",
        style="#1cb2f5",
        errors=True,
    )

    P.plot_fit(
        axes=ax,
        x=U,
        y=p,
        eqn=p,
        style=r"#1cb2f5",
        label="Strom",
        offset=[0, 30],
        use_all_known=False,
        guess={"a": 10, "b": 18},
        bounds=[
            {"name": "b", "min": 000, "max": 9000},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        # scale_covar=True,
    )
    print(P.data)

    ax.set_title(f"Leistungskennlinie von Spannung")
    P.ax_legend_all(loc=4)
    ax = P.savefig(f"pUkennlinie.pdf")
    # A3 Darstellung der Zählstatistik
    P.vload()


if __name__ == "__main__":
    test_leistung_protokoll()
