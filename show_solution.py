
import pyvista as pv
import numpy as np
import natsort
import glob
import time

for file in sorted(glob.glob("build/solutions/*.vtu")):
    mesh = pv.read(file)
    mesh.points[:, 2] = mesh["potential"]

    plotter = pv.Plotter()
    plotter.camera_position = [
        (0.0, 0.0, 5.0),
        (0.0, 0.0, 0.0),
        (0, 1, 0)
    ]

    plotter.add_mesh(mesh, scalars="potential", show_edges=True, cmap="viridis", clim=[0, 1])
    plotter.add_text(f"{file}", position='upper_left', font_size=12, color='black')
    plotter.show(auto_close=False)
    plotter.close()

