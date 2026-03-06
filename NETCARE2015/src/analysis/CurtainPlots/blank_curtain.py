# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:28:04 2025

@author: repooley
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_blank_curtain(x_edges, y_edges, vmin, vmax, title, cbar_label):
    fig, ax = plt.subplots(figsize=(6, 6))

    ##--Create colormap with white for "under" values--##
    new_cmap = plt.get_cmap('viridis')
    new_cmap.set_under('w')

    ##--Make a blank (all-NaN) array for pcolormesh--##
    blank_data = np.full((len(y_edges)-1, len(x_edges)-1), np.nan)

    ##--Plot the empty mesh--##
    mesh = ax.pcolormesh(
        x_edges, y_edges, blank_data,
        shading="auto",
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ##--Colorbar--##
    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', location='bottom', pad=0.15) 
    cb.minorticks_on()
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cbar_label, fontsize=18)

    ##--Polar dome boundaries--##
    ax.axhline(y=285, color='k', linestyle='--', linewidth=2)
    ax.axhline(y=299, color='k', linestyle='--', linewidth=2)

    ##--Axis labels--##
    ax.set_xlabel("Latitude (°)", fontsize=18)
    ax.set_ylabel("Potential Temperature Θ (K)", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(title, fontsize=20)

    ##--Polar Dome Text--##
    ax.text(72, 282, "Polar Dome", fontsize=18, color="k",
            verticalalignment="center", horizontalalignment="left")
    ax.text(72, 288, "Marginal Dome", fontsize=18, color="k",
            verticalalignment="center", horizontalalignment="left")

    plt.tight_layout()
    plt.show()
    
    
lat_edges = np.linspace(70, 85, 50)
theta_edges = np.linspace(238, 309, 80)

plot_blank_curtain(
    x_edges=lat_edges,
    y_edges=theta_edges,
    vmin=0, vmax=1,
    title=" ",
    cbar_label="Particle Count"
)