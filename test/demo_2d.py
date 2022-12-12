import os
import taichi as ti
import numpy as np
import math
from mpm_solver import MPMSolver
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    args = parser.parse_args()
    print(args)
    return args


args = parse_args()

write_to_disk = args.out_dir is not None
if write_to_disk:
    os.mkdir(f'{args.out_dir}')

ti.init(arch=ti.cuda)  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

res=(32, 32, 32)
mpm = MPMSolver(res=res)

for i in range(1):
    mpm.add_sphere(center=[0.5, 0.5, 0.5],
                 radius=0.025,
                 sample_density=4**len(res),
                 material=MPMSolver.material_elastic)

for frame in range(500):
    mpm.step(8e-3)
    if 10 < frame < 100:
        mpm.add_cube(lower_corner=[0.45, 0.01, 0.45],
                     cube_size=[0.1, 0.01, 0.1],
                     material=MPMSolver.material_water,
                     velocity=[math.sin(frame * 0.1), 3, math.cos(frame * 0.1)])
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()

    if len(res) == 3:
        np_x = particles['position']
        transform = 'f'
        if transform == 'frontal':
            # simple camera transform
            screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
            screen_y = (np_x[:, 1])

            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            gui.circles(screen_pos,
                    radius=1.5,
                    color=colors[particles['material']])
        else:
            r = (np_x[:, 1])
            screen_x = (np_x[:, 2]) + 0.5 * (np_x[:, 2] - 0.5) * (r - 0.5)
            screen_y = (np_x[:, 0]) + 0.5 * (np_x[:, 0] - 0.5) * (r - 0.5)
            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            idx = r.argsort()
            screen_pos = screen_pos[idx]
            r = r[idx]
            material = particles['material'][idx]
            for i in range(len(screen_pos)):
                gui.circle(screen_pos[i],
                        radius=1.5 + 3 * r[i],
                        color=colors[material[i]])

        
    else:
        gui.circles(particles['position'],
                    radius=1.5,
                    color=colors[particles['material']])
    gui.show(f'{args.out_dir}/{frame:06d}.png' if write_to_disk else None)
