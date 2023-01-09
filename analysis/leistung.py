from labtool_ex2 import Project
from sympy import exp, pi, sqrt, Abs
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt  # noqa
import os
import cmath

# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false


def plotComplexChain(vecs: NDArray, axes: plt.Axes):
    v_off = np.cumsum(vecs)
    vecs = np.hstack([vecs, v_off[-1]])
    v_off[-1] = 0
    v_off = np.hstack([np.zeros_like(v_off[0]), v_off])
    colors = ["#0fafaf"] * len(vecs)
    colors[-1] = "#f00f0E"
    print(vecs)
    print(v_off)
    print(vecs.real)
    print(vecs.imag)
    print(np.abs(v_off))
    print(np.angle(v_off))
    # If scale_units is 'x' then the vector will be 0.5 x-axis units.
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    axes.quiver(
        np.angle(v_off),
        np.abs(v_off),
        vecs.real,
        vecs.imag,
        scale_units="y",
        scale=2,
        facecolor=colors,
        width=5e-3,
    )


def VoltNorma(U):
    klasse = 0.005  # 0.5%
    if U < 120:
        return klasse * 120 + 120 / 120
    if U < 240:
        return klasse * 240 + 240 / 120
    if U < 480:
        return klasse * 480 + 5
    if U < 600:
        return klasse * 600 + 600 / 120


def AmpereNorma(I):
    klasse = 0.015  # 1.5%
    if I < 0.24:
        return klasse * 0.24 + 0.24 / 120
    if I < 0.6:
        return klasse * 0.6 + 0.6 / 120
    if I < 1.2:
        return klasse * 1.2 + 1.2 / 120
    if I < 2.4:
        return klasse * 2.4 + 2.4 / 120
    if I < 6:
        return klasse * 6 + 6 / 120


def AmpereDigital(I):
    if I <= 0.020:
        if I < 0.01:
            return 0.01 * I + 0.00003

        return 0.01 * I + 0.0003
    if I <= 2:
        return 0.018 * I + 0.003
    if I > 2:
        return 0.03 * I + 0.03


def VoltDigital(U):
    if U <= 200:
        return 0.008 * U + 0.3
    if U > 200:
        return 0.012 * U + 3


def PowerNorma(P):
    klasse = 0.005  # 0.5%
    if P < 120:
        return 120 * klasse + 120 / 120
    if P < 240:
        return 240 * klasse + 240 / 120
    if P < 480:
        return 480 * klasse + 480 / 120
    if P < 600:
        return 600 * klasse + 600 / 120


def PowerChauvin(P):
    return P * 0.01


def printVectorChain(vecs: NDArray, axes: plt.Axes):
    """vectorChain is a (2,) array in the first row are the x coordinates of the vector which shall be concateted and in second row are the y coordinates"""
    # vectorChain

    v_off = np.zeros_like(vecs)
    prev_vec = np.zeros_like(vecs[0])
    print(prev_vec.shape, v_off.shape)
    v_off = np.vstack([v_off, prev_vec])
    for i, vec in enumerate(vecs):
        v_off[i] = prev_vec
        prev_vec += vec

    vecs = np.vstack([vecs, prev_vec])

    colors = ["#0fafaf"] * len(vecs)
    colors[-1] = "#f00f0E"
    print(v_off[:, 0])
    print(v_off[:, 1])
    print(vecs[:, 0])
    print(vecs[:, 1])

    axes.quiver(
        v_off[:, 0],
        v_off[:, 1],
        vecs[:, 0],
        vecs[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        facecolor=colors,
        width=3e-3,
    )


def test_leistung_protokoll():
    # zLuft / cps zPapier / cps zKunststoff / cps zAlu0_8 / cps zAlu1_5 / cps
    gm = {
        "t": r"t",
        "U": r"U",
        "U1": r"U_1",
        "U2": r"U_2",
        "U3": r"U_3",
        "U4": r"U_4",
        "U5": r"U_5",
        "I": r"I",
        "I1": r"I_1",
        "I2": r"I_2",
        "I3": r"I_3",
        "I4": r"I_{31}",
        "a": r"a",
        "b": r"b",
        "c": r"c",
        "d": r"d",
        "p": r"P",
        "m": r"P_0",
        "p1": r"P_1",
        "p2": r"P_2",
        "p3": r"P_3",
        "E": r"E_{\mathrm{kin}}",
    }
    gv = {
        "t": r"\si{\second}",
        "U": r"\si{\volt}",
        "U1": r"\si{\volt}",
        "U2": r"\si{\volt}",
        "U3": r"\si{\volt}",
        "U4": r"\si{\volt}",
        "U5": r"\si{\volt}",
        "I": r"\si{\ampere}",
        "I1": r"\si{\ampere}",
        "I2": r"\si{\ampere}",
        "I3": r"\si{\ampere}",
        "I4": r"\si{\ampere}",
        "E": r"\si{\mega\electronvolt}",
        "a": r"\si{\volt}",
        "b": r"\si{\ohm}",
        "m": r"\si{\watt}",
        "c": r"\si{\ampere}",
        "d": r"\si{\siemens}",
        "p": r"\si{\watt}",
        "p1": r"\si{\watt}",
        "p2": r"\si{\watt}",
        "p3": r"\si{\watt}",
    }

    plt.rcParams["axes.axisbelow"] = True
    P = Project("Leistung", global_variables=gv, global_mapping=gm, font=13)
    P.output_dir = "./"
    P.figure.set_size_inches((8, 6))
    ax: plt.Axes = P.figure.add_subplot()
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

    p = a * I + b * I**2 + m

    P.plot_data(
        ax,
        I,
        p,
        label="Gemessene Daten",
        style="#FFA500",
        errors=True,
    )

    P.plot_fit(
        axes=ax,
        x=I,
        y=p,
        eqn=p,
        style=r"#FFA500",
        label="Strom",
        offset=[0, 30],
        use_all_known=False,
        guess={"a": 10, "b": 18, "m": 0},
        bounds=[
            {"name": "b", "min": 000, "max": 9000},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        scale_covar=True,
    )

    ax.set_title(f"Leistungskennlinie von Strom")
    P.ax_legend_all(loc=4)
    ax = P.savefig(f"pIkennlinie.pdf")

    p = c * U + d * U**2 + m

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
        label="Spannung",
        offset=[0, 30],
        use_all_known=False,
        guess={"c": 10, "d": 18, "m": 0},
        bounds=[
            {"name": "c", "min": 000, "max": 9000},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        scale_covar=True,
    )

    ax.set_title(f"Leistungskennlinie von Spannung")
    P.ax_legend_all(loc=4)
    ax = P.savefig(f"pUkennlinie.pdf")
    # A2 Darstellung der Zählstatistik
    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe2.csv")
    P.load_data(filepath, loadnew=True)
    P.data["dI1"] = I1.data.apply(AmpereNorma)
    P.data["dI2"] = I2.data.apply(AmpereNorma)
    P.data["dI3"] = I3.data.apply(AmpereNorma)
    P.data["dI4"] = I4.data.apply(AmpereDigital)
    P.data["dU1"] = U1.data.apply(VoltNorma)
    P.data["dU2"] = U2.data.apply(VoltNorma)
    P.data["dU3"] = U3.data.apply(VoltDigital)
    P.data["dp1"] = p1.data.apply(PowerNorma)
    P.data["dp2"] = p2.data.apply(PowerNorma)
    P.data["dp3"] = p3.data.apply(PowerChauvin)
    P.print_table(I1, I2, I3, I4, U1, U2, U3, p1, p2, p3, name="alles")
    I1 = I1 + 0j
    I2 = I2 * cmath.exp(cmath.pi / 3 * 2j)
    I3 = I3 * cmath.exp(cmath.pi / 3 * 4j)
    I12 = sqrt(I1**2 - 3 * I4**2 / 4) - I4 / 2
    I23 = sqrt(I3**2 - 3 * I4**2 / 4) - I4 / 2
    I31 = I4
    U1 = U1 + 0j
    U2 = U2 * cmath.exp(cmath.pi / 3 * 2j)
    U3 = U3 * cmath.exp(cmath.pi / 3 * 4j)
    U31 = U3 - U1
    U12 = U1 - U2
    U23 = U2 - U3
    P.resolve(I1)
    P.resolve(I2)
    P.resolve(I3)
    P.resolve(I12)
    P.resolve(I23)
    P.resolve(I31)
    P.resolve(U1)
    P.resolve(U2)
    P.resolve(U3)
    P.resolve(U12)
    P.resolve(U23)
    P.resolve(U31)

    # print(
    #     P.data[
    #         [
    #             "I1",
    #             "I2",
    #             "I3",
    #             "I12",
    #             "I23",
    #             "I31",
    #             "U1",
    #             "U2",
    #             "U3",
    #             "U12",
    #             "U23",
    #             "U31",
    #         ]
    #     ]
    # )
    P.figure.clear()
    polar_ax = P.figure.add_subplot(polar=True)
    vecstuffen = np.stack((P.data.values.real, P.data.values.imag), axis=-1)
    # vecs = vecstuffen[0][4:7]
    # print(P.data.values[0][4:7])
    plotComplexChain(P.data.values[0][4:6], polar_ax)
    # printVectorChain(vecs, polar_ax)
    # ax.quiver(x_off, y_off, X, Y, angles="xy", scale_units="xy", scale=1)

    ax.set_title(f"Zeigerdiagramm")
    P.ax_legend_all(loc=4)
    ax = P.savefig(f"zeiger.pdf")
    print(P.data)
    # A3 Darstellung der Zählstatistik
    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe3.csv")
    P.load_data(filepath, loadnew=True)
    P.data = P.raw_data
    # print(P.data)


if __name__ == "__main__":
    test_leistung_protokoll()
