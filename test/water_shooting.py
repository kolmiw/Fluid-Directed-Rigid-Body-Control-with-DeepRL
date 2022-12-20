import taichi as ti
import numpy as np
import math
from mpm_solver import MPMSolver

RESOLUTION = 512
GRID_RESOLUTION = 128
MATERIAL = MPMSolver.material_water # or material_snow
BG_COLOR = 0xBBBBBB
BALL_R = 0.025
JET_SPEED = 0.05
JET_R = BALL_R * 3
JET_ANG_SPEED = JET_SPEED * ti.math.pi
JET_POWER = 5
FLOOR_H = 0.03
SHOT_DURATION = 3
SHOT_H = JET_POWER/150.0/SHOT_DURATION
COLORS = [0x3399FF, 0xED553B, 0xEEEEF0, 0xFFFF00]

def render_jet(gui, jet_params):
    # ugly jet
        jet_h = 0.02
        jet_color = 0x444444
        origin = [jet_params[0], FLOOR_H]
        vec1 = [ti.math.cos(jet_params[1]), ti.math.sin(jet_params[1])]
        vec2 = [vec1[1], -vec1[0]]
        ref_A = [origin[0] + JET_R/2 * vec2[0], origin[1] + JET_R/2 * vec2[1]]
        ref_B = [origin[0] - JET_R/2 * vec2[0], origin[1] - JET_R/2 * vec2[1]]
        point_A = [ref_A[0] - vec1[0], ref_A[1] - vec1[1]]
        point_B = [ref_B[0] - vec1[0], ref_B[1] - vec1[1]]
        point_C = [ref_B[0] + jet_h * vec1[0], ref_B[1] + jet_h * vec1[1]]
        point_D = [ref_A[0] + jet_h * vec1[0], ref_A[1] + jet_h * vec1[1]]
        gui.triangle(point_A, point_B, point_C, color=jet_color)
        gui.triangle(point_A, point_C, point_D, color=jet_color)
        # upper part is wider
        point_A1 = [ref_A[0] + JET_R/8 * vec2[0], ref_A[1] + JET_R/8 * vec2[1]]
        point_B1 = [ref_B[0] - JET_R/8 * vec2[0], ref_B[1] - JET_R/8 * vec2[1]]
        point_C1 = [point_C[0] - JET_R/8 * vec2[0], point_C[1] - JET_R/8 * vec2[1]]
        point_D1 = [point_D[0] + JET_R/8 * vec2[0], point_D[1] + JET_R/8 * vec2[1]]
        gui.triangle(point_A1, point_B1, point_C1, color=jet_color)
        gui.triangle(point_A1, point_C1, point_D1, color=jet_color)

def play(ac):
    ti.init(arch=ti.cuda)  # Try to run on GPU
    gui = ti.GUI("Robot is playing rn", res=RESOLUTION, background_color=BG_COLOR)
    mpm = MPMSolver(res=(GRID_RESOLUTION, GRID_RESOLUTION))
    mpm.add_sphere(center=[0.5, 0.5],
             radius=BALL_R, 
             sample_density=4,
             material=MPMSolver.material_elastic)
    jet_params = [0.5, ti.math.pi/2]
    particle_shot_counter = 0
    colors = np.array(COLORS, dtype=np.uint32)

    for frame in range(20000):
        states = None
        action = ac.step(states)
        """
        Managing gui inputs
        r - Reset to initial state
        esc - quits the program
        
        action
            - change jet location
            - change jet direction
            - Shoot dihydrogen monoxide from the Particle accelerator
        """
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        # action = [float x.acc, float z.acc, float theta acc, float phi acc, bool jet_on]

        # update jet location
        jet_params[0] = ti.math.min(ti.math.max(jet_params[0] + action[0], JET_R/2), 1 - JET_R/2)

        # update jet direction
        jet_params[1] = ti.math.min(ti.math.max(0, jet_params[1] + action[1]), ti.math.pi)
        
        # LMB is pressed, shoot water
        if action[2]:
            particle_shot_counter = SHOT_DURATION
        if particle_shot_counter > 0:
            particle_shot_counter -= 1        
            mpm.add_cube(lower_corner=[jet_params[0] - JET_R/2, FLOOR_H - SHOT_H],
                        cube_size=[JET_R, SHOT_H],
                        material=MATERIAL,
                        velocity=[JET_POWER * ti.math.cos(jet_params[1]), JET_POWER * ti.math.sin(jet_params[1])])

        mpm.step(2e-3)
        particles = mpm.particle_info()

        gui.circles(particles['position'],
                    radius=1.5,
                    color=colors[particles['material']])

        render_jet(gui, jet_params)
        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()

def main():
    ti.init(arch=ti.cuda)  # Try to run on GPU
    gui = ti.GUI("Taichi Elements", res=RESOLUTION, background_color=BG_COLOR)
    mpm = MPMSolver(res=(GRID_RESOLUTION, GRID_RESOLUTION))
    mpm.add_sphere(center=[0.5, 0.5],
             radius=BALL_R, 
             sample_density=4,
             material=MPMSolver.material_elastic)
    jet_params = [0.5, ti.math.pi/2]
    particle_shot_counter = 0
    colors = np.array(COLORS, dtype=np.uint32)

    for frame in range(20000):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
            elif gui.event.key == 'a':
                jet_params[0] = ti.math.max(jet_params[0] - JET_SPEED, JET_R/2)
            elif gui.event.key == 'd':
                jet_params[0] = ti.math.min(jet_params[0] + JET_SPEED, 1 - JET_R/2)
        
        mouse = gui.get_cursor_pos()
        if mouse[1] > 1e-4:
            # Compute the angle between the cursor and the jet's location
            dist = ti.math.max(1e-4, ti.math.sqrt((mouse[0] - jet_params[0])**2 + mouse[1]**2))
            mouse_jet_angle = ti.math.acos((mouse[0]- jet_params[0]) / dist)
            if mouse_jet_angle - 1.1*JET_ANG_SPEED > jet_params[1]:
                jet_params[1] = ti.math.min(ti.math.pi, jet_params[1] + JET_ANG_SPEED)
            elif mouse_jet_angle + 1.1*JET_ANG_SPEED < jet_params[1]:
                jet_params[1] = ti.math.max(0, jet_params[1] - JET_ANG_SPEED)
        
        if gui.is_pressed(ti.GUI.LMB):
            particle_shot_counter = SHOT_DURATION
        if particle_shot_counter > 0:
            particle_shot_counter -= 1
            mpm.add_cube(lower_corner=[jet_params[0] - JET_R/2, FLOOR_H - SHOT_H],
                        cube_size=[JET_R, SHOT_H],
                        material=MATERIAL,
                        velocity=[JET_POWER * ti.math.cos(jet_params[1]), JET_POWER * ti.math.sin(jet_params[1])])

        mpm.step(2e-3)
        particles = mpm.particle_info()

        gui.circles(particles['position'],
                    radius=1.5,
                    color=colors[particles['material']])

        render_jet(gui, jet_params)

        gui.show()

def render_jet_3D(gui, jet_params):
    # ugly jet
    jet_color = 0x444444

    screen_x = jet_params[0] + 0.5 * (jet_params[0] - 0.5) * (FLOOR_H - 0.5)
    screen_y = jet_params[1] + 0.5 * (jet_params[1] - 0.5) * (FLOOR_H - 0.5)
    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circle(screen_pos, radius=JET_R*200, color=jet_color)
    gui.circle(screen_pos, radius=JET_R*160, color=0xFFFFFF)

def play_3D(ac):
    ti.init(arch=ti.cuda)  # Try to run on GPU
    gui = ti.GUI("Robot is playing rn", res=RESOLUTION, background_color=BG_COLOR)
    mpm = MPMSolver(res=(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION))
    mpm.add_sphere(center=[0.5, 0.5, 0.5],
             radius=BALL_R, 
             sample_density=64,
             material=MPMSolver.material_elastic)
    jet_params = [0.5, 0.5, 0, ti.math.pi/2]
    particle_shot_counter = 0
    colors = np.array(COLORS, dtype=np.uint32)

    for frame in range(20000):
        states = None
        action = ac.step(states)
        """
        Managing gui inputs
        r - Reset to initial state
        esc - quits the program
        
        action
            - change jet location
            - change jet direction
            - Shoot dihydrogen monoxide from the Particle accelerator
        """
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        # action = [float x.acc, float angular acc, bool jet_on]

        # update jet location
        jet_params[0] = ti.math.min(ti.math.max(jet_params[0] + action[0], JET_R/2), 1 - JET_R/2)
        jet_params[1] = ti.math.min(ti.math.max(jet_params[1] + action[1], JET_R/2), 1 - JET_R/2)

        # update jet direction
        jet_params[2] = ti.math.min(ti.math.max(0, jet_params[1] + action[1]), 2 * ti.math.pi)
        jet_params[3] = ti.math.min(ti.math.max(0, jet_params[1] + action[1]), ti.math.pi/2)

        # LMB is pressed, shoot water
        if action[2]:
            particle_shot_counter = SHOT_DURATION
        if particle_shot_counter > 0:
            particle_shot_counter -= 1        
            mpm.add_cube(lower_corner=[jet_params[0] - JET_R/2, FLOOR_H - SHOT_H, jet_params[0] - JET_R/2],
                        cube_size=[JET_R, SHOT_H, JET_R],
                        material=MATERIAL,
                        velocity=[JET_POWER * ti.math.cos(jet_params[2]) * ti.math.sin(jet_params[3]), JET_POWER * ti.math.cos(jet_params[3]), JET_POWER * ti.math.sin(jet_params[2]) * ti.math.sin(jet_params[3])])

        mpm.step(2e-3)
        particles = mpm.particle_info()

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
            screen_x = (np_x[:, 0]) + 0.5 * (np_x[:, 0] - 0.5) * (r - 0.5)
            screen_y = (np_x[:, 2]) + 0.5 * (np_x[:, 2] - 0.5) * (r - 0.5)
            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            idx = r.argsort()
            screen_pos = screen_pos[idx]
            r = r[idx]
            material = particles['material'][idx]
            for i in range(len(screen_pos)):
                gui.circle(screen_pos[i],
                        radius=1.5 + 3 * r[i],
                        color=colors[material[i]])

        render_jet_3D(gui, jet_params)
        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()

def main_3D():
    ti.init(arch=ti.cuda)  # Try to run on GPU
    gui = ti.GUI("Taichi Elements", res=RESOLUTION, background_color=BG_COLOR)
    mpm = MPMSolver(res=(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION))
    mpm.add_sphere(center=[0.5, 0.5, 0.5],
             radius=BALL_R, 
             sample_density=64,
             material=MPMSolver.material_elastic)
    jet_params = [0.5, 0.5, 0, ti.math.pi/2]
    particle_shot_counter = 0
    colors = np.array(COLORS, dtype=np.uint32)

    for frame in range(20000):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
            elif gui.event.key == 'a':
                jet_params[0] = ti.math.max(jet_params[0] - JET_SPEED, JET_R/2)
            elif gui.event.key == 'd':
                jet_params[0] = ti.math.min(jet_params[0] + JET_SPEED, 1 - JET_R/2)
            elif gui.event.key == 's':
                jet_params[1] = ti.math.max(jet_params[1] - JET_SPEED, JET_R/2)
            elif gui.event.key == 'w':
                jet_params[1] = ti.math.min(jet_params[1] + JET_SPEED, 1 - JET_R/2)
        
        mouse = gui.get_cursor_pos()
        dist = ti.math.max(1e-4, ti.math.sqrt((mouse[0] - jet_params[0])**2 + (mouse[1] - jet_params[1])**2))
        mouse_jet_angle = 0
        if mouse[1] > jet_params[1]:
            # Compute the angle between the cursor and the jet's location
            mouse_jet_angle = ti.math.acos((mouse[0] - jet_params[0]) / dist)
        else:
            mouse_jet_angle = 2 * ti.math.pi - ti.math.acos((mouse[0] - jet_params[0]) / dist)
        if mouse_jet_angle - 1.1*JET_ANG_SPEED > jet_params[2]:
            jet_params[2] = math.fmod(jet_params[2] + JET_ANG_SPEED, 2*ti.math.pi)
        elif mouse_jet_angle + 1.1*JET_ANG_SPEED < jet_params[2]:
            jet_params[2] = math.fmod(jet_params[2] - JET_ANG_SPEED, 2*ti.math.pi)
        if jet_params[3] - 1.1*JET_ANG_SPEED < dist**(4.0/3):
            jet_params[3] = ti.math.max(jet_params[3] - JET_ANG_SPEED, 0)
        elif jet_params[3] + 1.1*JET_ANG_SPEED > dist**(4.0/3):
            jet_params[3] = ti.math.min(jet_params[3] - JET_ANG_SPEED, ti.math.pi/2)
        
        if gui.is_pressed(ti.GUI.LMB):
            particle_shot_counter = SHOT_DURATION
        if particle_shot_counter > 0:
            particle_shot_counter -= 1
            mpm.add_cube(lower_corner=[jet_params[0] - JET_R/2, FLOOR_H - SHOT_H, jet_params[1] - JET_R/2],
                        cube_size=[JET_R, SHOT_H, JET_R],
                        material=MATERIAL,
                        velocity=[JET_POWER * ti.math.cos(jet_params[2]) * ti.math.sin(jet_params[3]), JET_POWER * ti.math.cos(jet_params[3]), JET_POWER * ti.math.sin(jet_params[2]) * ti.math.sin(jet_params[3])])

        mpm.step(2e-3)
        particles = mpm.particle_info()

        render_jet_3D(gui, jet_params)

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
            screen_x = (np_x[:, 0]) + 0.5 * (np_x[:, 0] - 0.5) * (r - 0.5)
            screen_y = (np_x[:, 2]) + 0.5 * (np_x[:, 2] - 0.5) * (r - 0.5)
            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            idx = r.argsort()
            screen_pos = screen_pos[idx]
            r = r[idx]
            material = particles['material'][idx]
            for i in range(len(screen_pos)):
                gui.circle(screen_pos[i],
                        radius=1.5 + 3 * r[i],
                        color=colors[material[i]])
        gui.show()

if __name__ == "__main__":
    # execute only if run as a script
    main_3D()
