from labtool_ex2 import Project
from sympy import exp, pi, sqrt, Abs, conjugate, pi, acos, asin, atan
from sympy import I as jj
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt  # noqa
import matplotlib.colors
import os
import cmath
import math
from colorsys import rgb_to_hsv, hsv_to_rgb

# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false


def complementary(r, g, b):
    """returns RGB components of complementary color"""
    hsv = rgb_to_hsv(r, g, b)
    return hsv_to_rgb((hsv[0] + 0.2) % 1, hsv[1], hsv[2])


def plotComplexChain(vecs: NDArray, axes: plt.Axes, color: str = "#0fafaf", label=None):
    v_off = np.cumsum(vecs)
    vecs = np.hstack([vecs, v_off[-1]])
    v_off[-1] = 0
    v_off = np.hstack([np.zeros_like(v_off[0]), v_off])
    rgb = matplotlib.colors.to_rgb(color)
    colors = [rgb] * len(vecs)
    print(v_off)
    print(vecs)
    if len(vecs) > 2:
        colors[-1] = complementary(*rgb)
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
        label=label,
        width=5e-3,
    )


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def zeigerDreieck(axes, I, U):
    currentaxes = axes.figure.add_axes(
        axes.get_position(),
        projection="polar",
        label="twin",
        frameon=False,
        theta_direction=axes.get_theta_direction(),
        theta_offset=axes.get_theta_offset(),
        rlabel_position=170,
    )
    currentaxes.xaxis.set_visible(False)

    for leiterstrom, strangstrom, strangspannung in zip(
        I[["I1", "I2", "I3"]].values, I[["I12", "I23", "I31"]].values, U.values
    ):
        # currentaxes.set_rlabel_position(170)
        # currentaxes.set_ylabel("I", rotation=0)
        # axes.set_rlabel_position(22.5)
        # axes.set_ylabel("U", rotation=0)
        # axes.set_xlabel("$\\phi$")
        # Plot Strand Current
        plotComplexChain(
            np.hstack([strangstrom[0], -strangstrom[2]]),
            currentaxes,
            color="#FFAA66",
            label="Strangströme",
        )
        plotComplexChain(
            np.hstack([strangstrom[1], -strangstrom[0]]), currentaxes, color="#FFAA66"
        )
        plotComplexChain(
            np.hstack([strangstrom[2], -strangstrom[1]]), currentaxes, color="#FFAA66"
        )
        # Plot Composite Current

        currentaxes.quiver(
            np.zeros_like(leiterstrom, dtype=float),
            np.zeros_like(leiterstrom, dtype=float),
            leiterstrom.real,
            leiterstrom.imag,
            scale_units="y",
            scale=2,
            facecolor="#FF8C00",
            width=5e-3,
            label="Leiterströme",
        )
        # Plot Strand Voltages
        axes.quiver(
            np.zeros_like(strangspannung, dtype=float),
            np.zeros_like(strangspannung, dtype=float),
            strangspannung.real,
            strangspannung.imag,
            scale_units="y",
            scale=2,
            facecolor="#35baf6",
            width=5e-3,
            label="Strangspannungen",
        )

        cmax = abs(max(leiterstrom))
        smax = abs(max(strangspannung))
        ocmax = 10 ** orderOfMagnitude(cmax)
        osmax = 10 ** orderOfMagnitude(smax)
        if cmax / ocmax > smax / osmax:
            currentaxes.set_rlim([0, cmax])
            axes.set_rlim([0, cmax / ocmax * osmax])
        else:
            currentaxes.set_rlim([0, smax / osmax * ocmax])
            axes.set_rlim([0, smax])

        yield currentaxes


def zeigerStern(axes: plt.Axes, I, U):

    currentaxes = axes.figure.add_axes(
        axes.get_position(),
        projection="polar",
        label="twin",
        frameon=False,
        theta_direction=axes.get_theta_direction(),
        theta_offset=axes.get_theta_offset(),
        rlabel_position=170,
    )
    currentaxes.xaxis.set_visible(False)

    for strangstrom, strangspannung, compositespannung in zip(
        I.values, U[["U1", "U2", "U3"]].values, U[["U12", "U23", "U31"]].values
    ):
        # print(currentaxes._r_label_position)
        # currentaxes.set_rlabel_position(70)
        # currentaxes._r_label_position.invalidate()
        # print(currentaxes._r_label_position)
        # currentaxes.set_ylabel("I", rotation=0)
        # axes.set_rlabel_position(22.5)
        # axes.set_ylabel("U", rotation=0)
        axes.set_xlabel("$\\phi$")

        cmax = abs(max(strangstrom))
        smax = abs(max(compositespannung))
        ocmax = 10 ** orderOfMagnitude(cmax)
        osmax = 10 ** orderOfMagnitude(smax)
        if cmax / ocmax > smax / osmax:
            currentaxes.set_rlim([0, cmax])
            axes.set_rlim([0, cmax / ocmax * osmax])
            # currentaxes.set_rgrids([0, cmax], angle=22)
            # axes.set_rgrids([0, cmax / ocmax * osmax], angle=80)
        else:
            # currentaxes.set_rgrids([0, smax / osmax * ocmax], angle=22)
            # axes.set_rgrids([0, smax], angle=22)
            currentaxes.set_rlim([0, smax / osmax * ocmax])
            axes.set_rlim([0, smax])

        plotComplexChain(
            np.hstack([strangspannung[0], -strangspannung[1]]),
            axes,
            color="#35baf6",
            label="Strangspannungen",
        )
        plotComplexChain(
            np.hstack([strangspannung[1], -strangspannung[2]]), axes, color="#35baf6"
        )
        plotComplexChain(
            np.hstack([strangspannung[2], -strangspannung[0]]), axes, color="#35baf6"
        )
        axes.quiver(
            np.zeros_like(compositespannung, dtype=float),
            np.zeros_like(compositespannung, dtype=float),
            compositespannung.real,
            compositespannung.imag,
            scale_units="y",
            scale=2,
            facecolor="#0022ee",
            width=5e-3,
            label="Dreieckspannungen",
        )
        currentaxes.quiver(
            np.zeros_like(strangstrom, dtype=float),
            np.zeros_like(strangstrom, dtype=float),
            strangstrom.real,
            strangstrom.imag,
            scale_units="y",
            scale=2,
            facecolor="#FFAA66",
            width=5e-3,
            label="Strangströme",
        )

        yield currentaxes


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


def test_leistung_protokoll():
    # zLuft / cps zPapier / cps zKunststoff / cps zAlu0_8 / cps zAlu1_5 / cps
    gm = {
        "t": r"t",
        "U": r"U",
        "U1": r"U_1",
        "U2": r"U_2",
        "U3": r"U_3",
        "U12": r"U_{12}",
        "U23": r"U_{23}",
        "U31": r"U_{31}",
        "U4": r"U_{R3}",
        "U5": r"U_{L}",
        "I": r"I",
        "I1": r"I_1",
        "I2": r"I_2",
        "I3": r"I_3",
        "I12": r"I_{12}",
        "I23": r"I_{23}",
        "I31": r"I_{31}",
        "I4": r"I_{31}",
        "a": r"a",
        "b": r"b",
        "c": r"c",
        "d": r"d",
        "p": r"P",
        "m": r"P_0",
        "p1": r"P_1^{M}",
        "p2": r"P_2^{M}",
        "p3": r"P_3^{M}",
        "q1": r"Q_1^{M}",
        "q2": r"Q_2^{M}",
        "q3": r"Q_3^{M}",
        "p1c": r"P_1^{C}",
        "p2c": r"P_2^{C}",
        "p3c": r"P_3^{C}",
        "q1c": r"Q_1^{C}",
        "q2c": r"Q_2^{C}",
        "q3c": r"Q_3^{C}",
        "phi1": r"\phi_1",
        "phi2": r"\phi_2",
        "Pges": r"P_{ges}^{M}",
        "Qges": r"Q_{ges}^{M}",
        "Pgesc": r"P_{ges}^{C}",
        "Qgesc": r"Q_{ges}^{C}",
        "SgM": r"S_{ges}^{M}",
        "SgC": r"S_{ges}^{C}",
        "E": r"E_{\mathrm{kin}}",
    }
    gv = {
        "t": r"\si{\second}",
        "U": r"\si{\volt}",
        "U1": r"\si{\volt}",
        "U2": r"\si{\volt}",
        "U3": r"\si{\volt}",
        "U12": r"\si{\volt}",
        "U23": r"\si{\volt}",
        "U31": r"\si{\volt}",
        "U4": r"\si{\volt}",
        "U5": r"\si{\volt}",
        "I": r"\si{\ampere}",
        "I1": r"\si{\ampere}",
        "I2": r"\si{\ampere}",
        "I3": r"\si{\ampere}",
        "I12": r"\si{\ampere}",
        "I23": r"\si{\ampere}",
        "I31": r"\si{\ampere}",
        "I4": r"\si{\ampere}",
        "E": r"\si{\mega\electronvolt}",
        "a": r"\si{\volt}",
        "b": r"\si{\ohm}",
        "m": r"\si{\watt}",
        "c": r"\si{\ampere}",
        "d": r"\si{\ohm}",
        "p": r"\si{\watt}",
        "p1": r"\si{\watt}",
        "p2": r"\si{\watt}",
        "p3": r"\si{\watt}",
        "p1c": r"\si{\watt}",
        "p2c": r"\si{\watt}",
        "p3c": r"\si{\watt}",
        "q1c": r"\si{\Var}",
        "q2c": r"\si{\Var}",
        "q3c": r"\si{\Var}",
        "q1": r"\si{\Var}",
        "q2": r"\si{\Var}",
        "q3": r"\si{\Var}",
        "phi1": r"\si{\degree}",
        "phi2": r"\si{\degree}",
        "Pges": r"\si{\watt}",
        "Pgesc": r"\si{\watt}",
        "Qgesc": r"\si{\Var}",
        "Qges": r"\si{\Var}",
        "SgM": r"\si{\VA}",
        "SgC": r"\si{\VA}",
    }

    pd.set_option("display.max_columns", None)
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

    P.print_table(U, I, p, name="aufgabe1_raw", inline_units=True)
    p = a * I + b * I**2 + m
    P.print_expr(p)

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

    p = c * U + U**2 / d + m
    P.print_expr(p)

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

    # A2 Dreiecks Schaltung
    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe2dreieck.csv")
    P.load_data(filepath, loadnew=True)
    P.figure.clear()
    ax = P.figure.add_subplot(polar=True)
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 12, endpoint=False))

    P.data["dI1"] = I1.data.apply(AmpereNorma)
    P.data["dI2"] = I2.data.apply(AmpereNorma)
    P.data["dI3"] = I3.data.apply(AmpereNorma)
    P.data["dI4"] = I4.data.apply(AmpereDigital)
    P.data["dU12"] = U12.data.apply(VoltNorma)
    P.data["dU23"] = U23.data.apply(VoltNorma)
    P.data["dU31"] = U31.data.apply(VoltDigital)
    P.data["dp1"] = p1.data.apply(PowerNorma)
    P.data["dp2"] = p2.data.apply(PowerNorma)
    P.data["dp3"] = p3.data.apply(PowerChauvin)
    P.print_table(
        I1,
        I2,
        I3,
        I4,
        name="aufgabe2dreieck_1",
        inline_units=True,
    )

    P.print_table(
        U12,
        U23,
        U31,
        p1,
        p2,
        name="aufgabe2dreieck_2",
        inline_units=True,
    )
    P.data.fillna(0, inplace=True)
    I12 = sqrt(I1**2 - 3 * I4**2 / 4) - I4 / 2
    I23 = sqrt(I3**2 - 3 * I4**2 / 4) - I4 / 2
    I31 = I4
    P.resolve(I12)
    P.resolve(I23)
    P.resolve(I31)

    P.print_table(I12, I23, I31, name="dreieckStrangStrome", inline_units=True)

    P.data.I31 = P.data.I31 * cmath.exp(cmath.pi / 3 * 4j)
    P.data.I12 = P.data.I12 * cmath.exp(cmath.pi / 3 * 2j)
    P.data.I23 = P.data.I23 * cmath.exp(cmath.pi / 3 * 0j)

    I1 = I1 * exp(pi / 2 * jj)
    I2 = I2 * cmath.exp(cmath.pi * 11 / 6 * 1j)
    I3 = I3 * cmath.exp(cmath.pi * 7 / 6 * 1j)
    U12 = U12 * cmath.exp(cmath.pi / 3 * 2j)
    U23 = U23 * cmath.exp(cmath.pi / 3 * 0j)
    U31 = U31 * cmath.exp(cmath.pi / 3 * 4j)

    Pges = p1 + p2 + p3

    P.resolve(I1)
    P.resolve(I2)
    P.resolve(I3)
    P.resolve(U12)
    P.resolve(U23)
    P.resolve(U31)

    p1c = I12 * U12
    p2c = I23 * U23
    p3c = I31 * U31
    P.resolve(p1c)
    P.resolve(p2c)
    P.resolve(p3c)
    P.data.p1c = abs(P.data.p1c)
    P.data.p2c = abs(P.data.p2c)
    P.data.p3c = abs(P.data.p3c)
    Pgesc = p1c + p2c + p3c
    P.resolve(Pgesc)
    P.resolve(Pges)

    zeiger = zeigerDreieck(
        ax,
        I=P.data[
            [
                "I1",
                "I2",
                "I3",
                "I12",
                "I23",
                "I31",
            ]
        ],
        U=P.data[
            [
                "U12",
                "U23",
                "U31",
            ]
        ],
    )

    currentaxes = next(zeiger)

    ax.set_title(f"Zeigerdiagramm von Dreieckschaltung")
    P.ax_legend_all(loc=0)
    P.savefig(f"zeigerDreieck.pdf", clear=False)

    P.print_table(p1c, p2c, p3c, Pgesc, Pges, name="powerDreieck", inline_units=True)

    # A2 Stern Schaltung
    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe2.csv")
    P.load_data(filepath, loadnew=True)
    P.figure.clear()
    ax = P.figure.add_subplot(polar=True)
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 12, endpoint=False))

    # Calculate the Errors
    P.data["dI1"] = I1.data.apply(AmpereNorma)
    P.data["dI2"] = I2.data.apply(AmpereNorma)
    P.data["dI3"] = I3.data.apply(AmpereNorma)
    P.data["dI4"] = I4.data.apply(AmpereDigital)
    P.data["dU1"] = U1.data.apply(VoltNorma)
    P.data["dU2"] = U2.data.apply(VoltNorma)
    P.data["dU3"] = U3.data.apply(VoltDigital)
    P.data["dp1"] = p1.data.apply(PowerChauvin)
    P.data["dp2"] = p2.data.apply(PowerNorma)
    P.data["dp3"] = p3.data.apply(PowerNorma)

    # Export measurements
    P.print_table(I1, I2, I3, I4, name="aufgabe2stern_1", inline_units=True)
    P.print_table(U1, U2, U3, p1, p2, p3, name="aufgabe2stern_2", inline_units=True)

    P.data.fillna(0, inplace=True)
    gamma1 = acos((U1**2 + U3**2 - 400**2) / (2 * U1 * U3))
    gamma2 = acos((U2**2 + U3**2 - 400**2) / (2 * U2 * U3))
    gamma3 = acos((U2**2 + U1**2 - 400**2) / (2 * U2 * U1))
    phi2 = acos((U2**2 + 400**2 - U3**2) / (2 * U2 * 400))
    phi1 = acos((U1**2 + 400**2 - U2**2) / (2 * U1 * 400))
    phi3 = acos((U3**2 + 400**2 - U1**2) / (2 * U3 * 400))
    P.resolve(gamma1)
    P.resolve(gamma2)
    P.resolve(gamma3)
    P.resolve(phi2)
    P.resolve(phi1)
    P.resolve(phi3)
    # print(P.data.gamma1 - np.pi * 2 / 3)
    # print(P.data.gamma2 - np.pi * 2 / 3)
    # print(P.data.gamma3 - np.pi * 2 / 3)
    # print(1 / 3 - P.data.phi1 / (2 * np.pi))
    # print(P.data.phi2 / np.pi)
    # print(2 / 3 - P.data.phi3 / (2 * np.pi))

    I1 = I1 * cmath.exp(cmath.pi / 2 * 1j)
    I2 = I2 * cmath.exp(cmath.pi * 11 / 6 * 1j)
    I3 = I3 * cmath.exp(cmath.pi * 7 / 6 * 1j)
    # U1 = U1 * cmath.exp(cmath.pi / 2 * 1j)
    # U2 = U2 * cmath.exp(cmath.pi * 11 / 6 * 1j)
    # U3 = U3 * cmath.exp(cmath.pi * 7 / 6 * 1j)
    # TODO

    P.data.U1 = P.data.U1 * np.exp((2 * np.pi / 3 - P.data.phi1) * 1j)
    P.data.U2 = P.data.U2 * np.exp((-P.data.phi2) * 1j)
    P.data.U3 = P.data.U3 * np.exp((4 * np.pi / 3 - P.data.phi3) * 1j)
    U1N = 400 / np.sqrt(3) * cmath.exp(cmath.pi / 2 * 1j)
    U2N = 400 / np.sqrt(3) * cmath.exp(cmath.pi * 11 / 6 * 1j)
    U3N = 400 / np.sqrt(3) * cmath.exp(cmath.pi * 7 / 6 * 1j)
    print(
        np.asarray(
            [
                abs(P.data.U1.values[2] - U1N),
                abs(P.data.U2.values[2] - U2N),
                abs(P.data.U3.values[2] - U3N),
            ]
        ).mean()
    )

    Pges = p1 + p2 + p3
    # U0 = 4.45
    U0 = 40.8
    P.resolve(I1)
    P.resolve(I2)
    P.resolve(I3)

    U31 = U3 - U1
    U12 = U1 - U2
    U23 = U2 - U3
    P.resolve(U12)
    P.resolve(U23)
    P.resolve(U31)
    P.resolve(Pges)
    p1c = I1 * U1
    p2c = I2 * U2
    p3c = I3 * U3
    P.resolve(p1c)
    P.resolve(p2c)
    P.resolve(p3c)
    P.data.p1c = abs(P.data.p1c)
    P.data.p2c = abs(P.data.p2c)
    P.data.p3c = abs(P.data.p3c)
    Pgesc = p1c + p2c + p3c
    P.resolve(Pgesc)

    zeiger = zeigerStern(
        ax,
        I=P.data[
            [
                "I1",
                "I2",
                "I3",
            ]
        ],
        U=P.data[
            [
                "U1",
                "U2",
                "U3",
                "U12",
                "U23",
                "U31",
            ]
        ],
    )

    currentaxes = next(zeiger)
    plotComplexChain(
        P.data[
            [
                "I1",
                "I2",
                "I3",
            ]
        ].values[0],
        axes=currentaxes,
        color="#800000",
        label="Strangstromsumme",
    )
    ax.set_title(f"Symmetische Sternschaltung")
    P.ax_legend_all(loc=0)
    P.savefig(f"zeigerSternSym.pdf", clear=False)

    ax.clear()
    currentaxes.clear()
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 12, endpoint=False))
    currentaxes = next(zeiger)

    plotComplexChain(
        P.data[
            [
                "I1",
                "I2",
                "I3",
            ]
        ].values[1],
        axes=currentaxes,
        color="#800000",
        label="Strangstromsumme",
    )
    ax.set_title(f"Asymmetrische Sternschaltung ohne Bruch")
    P.ax_legend_all(loc=0)
    P.savefig(f"zeigerSternAsymOhneBruch.pdf", clear=False)

    ax.clear()
    currentaxes.clear()
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 12, endpoint=False))
    currentaxes = next(zeiger)
    plotComplexChain(
        np.hstack([P.data.U1[2], -U1N]),
        ax,
        color="#5f9ea0",
        label="$U_i - U_{iN} = U_0$",
    )
    plotComplexChain(
        np.hstack([P.data.U2[2], -U2N]),
        ax,
        color="#5f9ea0",
    )
    plotComplexChain(
        np.hstack([P.data.U3[2], -U3N]),
        ax,
        color="#5f9ea0",
    )

    ax.set_title(f"Asymmetrische Sternschaltung mit Bruch")
    P.ax_legend_all(loc=0)
    P.savefig(f"zeigerSternAsymBruch.pdf", clear=False)
    ax.clear()
    currentaxes.clear()
    P.print_table(p1c, p2c, p3c, Pgesc, Pges, name="powerSternAuf2", inline_units=True)

    # A3 Darstellung
    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe3.csv")
    P.load_data(filepath, loadnew=True)
    P.data = P.raw_data.droplevel("type", axis=1)
    P.vload()
    P.figure.clear()
    P.figure.set_size_inches(5, 5)
    ax = P.figure.add_subplot(polar=True)
    P.data["dI1"] = I1.data.apply(AmpereNorma)
    P.data["dI2"] = I2.data.apply(AmpereNorma)
    P.data["dI3"] = I3.data.apply(AmpereNorma)
    P.data["dI4"] = I4.data.apply(AmpereDigital)
    P.data["dU1"] = U1.data.apply(VoltNorma)
    P.data["dU2"] = U2.data.apply(VoltNorma)
    P.data["dU3"] = U3.data.apply(VoltNorma)
    P.data["dU4"] = U4.data.apply(VoltDigital)
    P.data["dU5"] = U5.data.apply(VoltDigital)
    P.data["dp1"] = p1.data.apply(PowerChauvin)
    P.data["dp2"] = p2.data.apply(PowerNorma)
    P.data["dp3"] = p3.data.apply(PowerNorma)
    P.gm["U1"] = "U_{1}"
    P.gm["U2"] = "U_{R2}"
    P.gm["U3"] = "U_{C}"
    P.gm["U4"] = "U_{R3}"
    P.gm["U5"] = "U_{L}"
    P.print_table(
        I1,
        I2,
        I3,
        I4,
        U1,
        U2,
        inline_units=True,
        name="aufgabe3mess_1",
    )
    P.print_table(
        U3,
        U4,
        U5,
        p1,
        p2,
        p3,
        inline_units=True,
        name="aufgabe3mess_2",
    )

    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe3leistung.csv")
    P.load_data(filepath, loadnew=True)
    P.data = P.raw_data.droplevel("type", axis=1)
    P.vload()
    P.data["dI1"] = I1.data.apply(AmpereNorma)
    P.data["dI2"] = I2.data.apply(AmpereNorma)
    P.data["dI3"] = I3.data.apply(AmpereNorma)
    P.data["dI4"] = I4.data.apply(AmpereDigital)
    P.data["dU1"] = U1.data.apply(VoltNorma)
    P.data["dU2"] = U2.data.apply(VoltNorma)
    P.data["dU3"] = U3.data.apply(VoltNorma)
    P.data["dU4"] = U4.data.apply(VoltDigital)
    P.data["dU5"] = U5.data.apply(VoltDigital)
    P.data["dp1"] = p1.data.apply(PowerChauvin)
    P.data["dp2"] = p2.data.apply(PowerNorma)
    P.data["dp3"] = p3.data.apply(PowerNorma)

    U23 = sqrt(U2**2 + U3**2)
    U12 = sqrt(U4**2 + U5**2)
    P.resolve(U23)
    P.resolve(U12)

    p1c = I1 * U1
    p2c = I2 * U2
    p3c = I3 * U4

    q1c = I1 * 0
    dq1c = I1 * 0
    q2c = -I2 * U3
    q3c = I3 * U5
    print(p1c)
    print(p2c)
    print(p3c)
    print(q1c)
    print(q2c)
    print(q3c)

    P.resolve(p1c)
    P.resolve(p2c)
    P.resolve(p3c)
    P.resolve(q1c)
    P.inject_err(dq1c)

    P.resolve(q2c)
    P.resolve(q3c)

    Pges = p1 + p2 + p3
    Pgesc = p1c + p2c + p3c
    Qgesc = q1c + q2c + q3c
    P.resolve(Pges)
    P.resolve(Pgesc)
    P.resolve(Qgesc)
    P.data = P.data.u.com
    pmesges = Pges.data
    P.data = P.data.u.sep
    P.print_table(
        p1c,
        p1,
        p2c,
        p2,
        p3c,
        p3,
        inline_units=True,
        table_type="tblr-x",
        options=r"cells={font=\footnotesize},row{1}={font=\mathversion{bold}\footnotesize}{cyan7},",
        name="aufgabe3power_1_1",
    )

    P.print_table(
        q1c,
        q2c,
        q3c,
        Qgesc,
        Pgesc,
        Pges,
        inline_units=True,
        table_type="tblr-x",
        options=r"cells={font=\footnotesize},row{1}={font=\mathversion{bold}\footnotesize}{cyan7},",
        name="aufgabe3power_2_1",
    )

    P.vload()
    filepath = os.path.join(os.path.dirname(__file__), "../data/aufgabe3blind.csv")
    P.load_data(filepath, loadnew=True)
    P.data = P.raw_data.droplevel("type", axis=1)
    P.vload()
    P.data["dI1"] = I1.data.apply(AmpereNorma)
    P.data["dI2"] = I2.data.apply(AmpereNorma)
    P.data["dI3"] = I3.data.apply(AmpereNorma)
    P.data["dI4"] = I4.data.apply(AmpereDigital)
    P.data["dU1"] = U1.data.apply(VoltNorma)
    P.data["dU2"] = U2.data.apply(VoltNorma)
    P.data["dU3"] = U3.data.apply(VoltNorma)
    P.data["dU4"] = U4.data.apply(VoltDigital)
    P.data["dU5"] = U5.data.apply(VoltDigital)
    P.data["dq1"] = q1.data.apply(PowerChauvin)
    P.data["dq2"] = q2.data.apply(PowerNorma)
    P.data["dq3"] = q3.data.apply(PowerNorma)
    q1 = q1 / ((3**0.5))
    q2 = q2 / ((3**0.5))
    q3 = q3 / ((3**0.5))

    U23 = sqrt(U2**2 + U3**2)
    U12 = sqrt(U4**2 + U5**2)
    P.resolve(U23)
    P.resolve(U12)

    p1c = I1 * U1
    p2c = I2 * U2
    p3c = I3 * U4

    q1c = I1 * 0
    dq1c = I1 * 0
    q2c = -I2 * U3
    q3c = I3 * U5

    P.resolve(p1c)
    P.resolve(p2c)
    P.resolve(p3c)

    P.resolve(q1c)
    P.resolve(q2c)
    P.resolve(q3c)

    P.inject_err(dq1c)

    P.resolve(q1)
    P.resolve(q2)
    P.resolve(q3)

    Qges = q1 + q2 + q3
    Pgesc = p1c + p2c + p3c
    Qgesc = q1c + q2c + q3c
    P.resolve(Qges)
    P.resolve(Pgesc)
    P.resolve(Qgesc)
    P.data = P.data.u.com
    P.data["Pges"] = pmesges
    P.data = P.data.u.sep
    SgC = sqrt(Pgesc**2 + Qgesc**2)
    SgM = sqrt(Pges**2 + Qges**2)
    P.resolve(SgC)
    P.resolve(SgM)
    P.print_table(
        q1c,
        q1,
        q2c,
        q2,
        q3c,
        q3,
        inline_units=True,
        options=r"cells={font=\footnotesize},row{1}={font=\mathversion{bold}\footnotesize}{blue7},",
        table_type="tblr-x",
        name="aufgabe3power_1_2",
    )

    P.print_table(
        p1c,
        p2c,
        p3c,
        Pgesc,
        Qgesc,
        Qges,
        inline_units=True,
        options=r"cells={font=\footnotesize},row{1}={font=\mathversion{bold}\footnotesize}{blue7},",
        table_type="tblr-x",
        name="aufgabe3power_2_2",
    )

    # phi1 = atan(U3 / U2)
    # phi2 = atan(-U5 / U4)
    phi1 = asin(U3 / 230)
    phi2 = asin(-U5 / 230)
    P.resolve(phi1)
    P.resolve(phi2)

    I1 = I1 * cmath.exp(cmath.pi * 0 * 1j)
    I2 = I2 * cmath.exp(((cmath.pi * 2 / 3) - phi1.data.values[0]) * 1j)
    I3 = I3 * cmath.exp(((cmath.pi * 4 / 3) - phi2.data.values[0]) * 1j)
    U1 = U1 * cmath.exp(cmath.pi * 0 * 1j)
    U2 = U2 * cmath.exp(((cmath.pi * 2 / 3) - phi1.data.values[0]) * 1j)
    U3 = U3 * cmath.exp(((cmath.pi * (2 / 3 + 0.5)) - phi1.data.values[0]) * 1j)
    U4 = U4 * cmath.exp(((cmath.pi * 4 / 3) - phi2.data.values[0]) * 1j)
    U5 = U5 * cmath.exp(((cmath.pi * (4 / 3 - 0.5)) - phi2.data.values[0]) * 1j)
    P.resolve(I1)
    P.resolve(I2)
    P.resolve(I3)
    P.resolve(U1)
    P.resolve(U2)
    P.resolve(U3)
    P.resolve(U4)
    P.resolve(U5)

    U31 = U4 + U5 - U1
    U12 = U1 - (U2 + U3)
    U23 = (U2 + U3) - (U4 + U5)
    P.resolve(U31)
    P.resolve(U12)
    P.resolve(U23)
    # ax.quiver(
    #     np.zeros_like(compositespannung, dtype=float),
    #     np.zeros_like(compositespannung, dtype=float),
    #     compositespannung.real,
    #     compositespannung.imag,
    #     scale_units="y",
    #     scale=2,
    #     facecolor="#0022ee",
    #     width=5e-3,
    #     label="Dreieckspannungen",
    # )

    # zeiger = zeigerStern(
    #     ax,
    #     I=P.data[
    #         [
    #             "I1",
    #             "I2",
    #             "I3",
    #         ]
    #     ],
    #     U=P.data[
    #         [
    #             "U12",
    #             "U23",
    #             "U31",
    #             "U1",
    #             "U2",
    #             "U3",
    #         ]
    #     ],
    # )
    # currentaxes = next(zeiger)
    currentaxes = ax.figure.add_axes(
        ax.get_position(),
        projection="polar",
        label="twin",
        frameon=False,
        theta_direction=ax.get_theta_direction(),
        theta_offset=ax.get_theta_offset(),
        rlabel_position=170,
    )
    currentaxes.xaxis.set_visible(False)
    ax.clear()
    currentaxes.clear()
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 12, endpoint=False))

    strangstrom = P.data[
        [
            "I1",
            "I2",
            "I3",
        ]
    ].values[0]
    cmax = abs(max(strangstrom))
    # smax = abs(max(compositespannung))
    smax = 230
    ocmax = 10 ** orderOfMagnitude(cmax)
    osmax = 10 ** orderOfMagnitude(smax)
    if cmax / ocmax > smax / osmax:
        currentaxes.set_rlim([0, cmax])
        ax.set_rlim([0, cmax / ocmax * osmax])
        # currentaxes.set_rgrids([0, cmax], angle=22)
        # ax.set_rgrids([0, cmax / ocmax * osmax], angle=80)
    else:
        # currentaxes.set_rgrids([0, smax / osmax * ocmax], angle=22)
        # ax.set_rgrids([0, smax], angle=22)
        currentaxes.set_rlim([0, smax / osmax * ocmax])
        ax.set_rlim([0, smax])
    currentaxes.quiver(
        np.zeros_like(strangstrom, dtype=float),
        np.zeros_like(strangstrom, dtype=float),
        strangstrom.real,
        strangstrom.imag,
        scale_units="y",
        scale=2,
        facecolor="#FFAA66",
        width=5e-3,
        label="Strangströme",
    )
    plotComplexChain(
        P.data[
            [
                "U2",
                "U3",
            ]
        ].values[0],
        axes=ax,
        color="#35baf6",
        label="Strangspannungen",
    )
    plotComplexChain(
        P.data[
            [
                "U4",
                "U5",
            ]
        ].values[0],
        axes=ax,
        color="#35baf6",
    )
    plotComplexChain(
        P.data[
            [
                "U1",
            ]
        ].values[0],
        axes=ax,
        color="#35baf6",
    )
    plotComplexChain(
        P.data[
            [
                "I1",
                "I2",
                "I3",
            ]
        ].values[0],
        axes=currentaxes,
        color="#800000",
        label="Strangstromsumme",
    )
    ax.set_title(f"Realler Verbraucher Sternschaltung")
    P.ax_legend_all(loc=0)
    P.savefig(f"zeigerSternReal.pdf", clear=False)
    ax.clear()
    currentaxes.clear()
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 12, endpoint=False))

    I2 = I2 * cmath.exp((cmath.pi * 2 / 3) * 1j)
    I3 = I3 * cmath.exp(-(cmath.pi * 2 / 3) * 1j)
    U2 = U2 * cmath.exp((cmath.pi * 2 / 3) * 1j)
    U3 = U3 * cmath.exp((cmath.pi * 2 / 3) * 1j)
    U4 = U4 * cmath.exp(-(cmath.pi * 2 / 3) * 1j)
    U5 = U5 * cmath.exp(-(cmath.pi * 2 / 3) * 1j)
    P.resolve(I1)
    P.resolve(I2)
    P.resolve(I3)
    P.resolve(U1)
    P.resolve(U2)
    P.resolve(U3)
    P.resolve(U4)
    P.resolve(U5)

    strangstrom = P.data[
        [
            "I1",
            "I2",
            "I3",
        ]
    ].values[1]
    cmax = abs(max(strangstrom))
    # smax = abs(max(compositespannung))
    smax = 230
    ocmax = 10 ** orderOfMagnitude(cmax)
    osmax = 10 ** orderOfMagnitude(smax)
    if cmax / ocmax > smax / osmax:
        currentaxes.set_rlim([0, cmax])
        ax.set_rlim([0, cmax / ocmax * osmax])
        # currentaxes.set_rgrids([0, cmax], angle=22)
        # ax.set_rgrids([0, cmax / ocmax * osmax], angle=80)
    else:
        # currentaxes.set_rgrids([0, smax / osmax * ocmax], angle=22)
        # ax.set_rgrids([0, smax], angle=22)
        currentaxes.set_rlim([0, smax / osmax * ocmax])
        ax.set_rlim([0, smax])
    currentaxes.quiver(
        np.zeros_like(strangstrom, dtype=float),
        np.zeros_like(strangstrom, dtype=float),
        strangstrom.real,
        strangstrom.imag,
        scale_units="y",
        scale=2,
        facecolor="#FFAA66",
        width=5e-3,
        label="Strangströme",
    )
    plotComplexChain(
        P.data[
            [
                "U2",
                "U3",
            ]
        ].values[1],
        axes=ax,
        color="#35baf6",
        label="Strangspannungen",
    )
    plotComplexChain(
        P.data[
            [
                "U4",
                "U5",
            ]
        ].values[1],
        axes=ax,
        color="#35baf6",
    )
    plotComplexChain(
        P.data[
            [
                "U1",
            ]
        ].values[1],
        axes=ax,
        color="#35baf6",
    )
    plotComplexChain(
        P.data[
            [
                "I1",
                "I2",
                "I3",
            ]
        ].values[1],
        axes=currentaxes,
        color="#800000",
        label="Strangstromsumme",
    )
    ax.set_title(f"Realler Verbraucher Sternschaltung Vertauscht")
    P.ax_legend_all(loc=0)
    P.savefig(f"zeigerSternRealVertauscht.pdf", clear=False)


if __name__ == "__main__":
    test_leistung_protokoll()
