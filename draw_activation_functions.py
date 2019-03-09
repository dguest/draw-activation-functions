#!/usr/bin/env python3

import matplotlib as mpl
import numpy as np
from canvas import Canvas
import os

def elu(x, alpha=1.0):
    y = np.zeros_like(x)
    y[x < 0] = alpha * np.expm1(x[x<0])
    y[x >= 0] = x[x >= 0]
    return y

def draw(odir, name):
    x = np.linspace(-4.5, 4.5, 101)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    with Canvas(f'{odir}/{name}') as can:
        can.ax.plot(x, 1/(1 + np.exp(-x)), '-', label="Sigmoid")
        can.ax.plot(x, np.maximum(0, x), '-', label='ReLU')
        # can.ax.plot(x, elu(x), '-', label='ELU')
        # can.ax.plot(x, np.tanh(x), '-', label='tanh')
        # can.ax.plot(x, np.maximum(0, np.minimum(1, x*0.2 + 0.5)),
        #             label='Hard Sigmoid')
        can.ax.legend(framealpha=1)
        can.ax.set_ylim(-0.2, 1.2)
        # can.ax.set_xscale('log')
        # for num, pt in enumerate(jet_pt, 1):
        #     line(can.ax, pt, f'Jet {num}\n{pt} GeV')

def run():
    draw('figures', 'activation-functions.pdf')

if __name__ == '__main__':
    run()
