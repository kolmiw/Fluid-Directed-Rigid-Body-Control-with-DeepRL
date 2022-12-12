import taichi as ti

ti.init(arch=ti.gpu)  # Try to run on GPU

#______________Convenience parameters
RESOLUTION = 512
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 2000 * quality**2, 64 * quality
jet_speed = 0.05
jet_angular_speed =  jet_speed * ti.math.pi
ball_radius = 0.02
jet_r = ball_radius*5
jet_power = 20
shoot_length = 5 # Number of frames a "shot" takes
ratio = 0.1
floor_h = 0.025

#_______________
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
jet_attributes = ti.Vector.field(2, dtype=float, shape=())


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            if i < 3 and grid_v[i, j][0] < 0.5:
                grid_v[i, j][0] = 0.5  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection




@ti.kernel
def reset():
    """
    create random position, for the ball we randomize polar coords and recast them to normal ones 
    """
    # Init ball
    for i in range(ratio*n_particles):
        if i == 0: # center of ball to track ball location
            x[i] = [0.5, 0.5]
        elif i == 1:
            x[i] = [0.5 + ball_radius, 0.5] # to track ball rotation
        else:
            radius = ti.random() * ball_radius
            degree = ti.random() * 360
            new_x = ti.math.cos(degree) * radius + 0.5
            new_y = ti.math.sin(degree) * radius + 0.5
            x[i] = [new_x, new_y]
        material[i] = 1 # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)
    
    # Init water
    # TODO: think about paramaters
    for i in range(ratio * n_particles, n_particles):
        radius = ti.random() * ball_radius
        degree = ti.random() * 2*ti.math.pi
        new_x = ti.math.cos(degree)*radius + 0.5
        new_y = ti.math.sin(degree)*radius + 0.05
        x[i] = [new_x, new_y]
        material[i] = 0 # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)  
    
    # Init jet
    jet_attributes[None] = [0.5, ti.math.pi/2]

@ti.kernel
def random_pos() -> ti.f32:
    return ti.random()

def render_jet(gui, jet_attributes):
    # ugly jet
    jet_h = 0.04
    jet_color = 0x444444
    origin = [jet_attributes[None][0], floor_h]
    vec1 = [ti.math.cos(jet_attributes[None][1]), ti.math.sin(jet_attributes[None][1])]
    vec2 = [vec1[1], -vec1[0]]
    ref_A = [origin[0] + jet_r/2 * vec2[0], origin[1] + jet_r/2 * vec2[1]]
    ref_B = [origin[0] - jet_r/2 * vec2[0], origin[1] - jet_r/2 * vec2[1]]
    point_A = [ref_A[0] - vec1[0], ref_A[1] - vec1[1]]
    point_B = [ref_B[0] - vec1[0], ref_B[1] - vec1[1]]
    point_C = [ref_B[0] + jet_h * vec1[0], ref_B[1] + jet_h * vec1[1]]
    point_D = [ref_A[0] + jet_h * vec1[0], ref_A[1] + jet_h * vec1[1]]
    gui.triangle(point_A, point_B, point_C, color=jet_color)
    gui.triangle(point_A, point_C, point_D, color=jet_color)
    # upper part is wider
    point_A1 = [ref_A[0] + jet_r/8 * vec2[0], ref_A[1] + jet_r/8 * vec2[1]]
    point_B1 = [ref_B[0] - jet_r/8 * vec2[0], ref_B[1] - jet_r/8 * vec2[1]]
    point_C1 = [point_C[0] - jet_r/8 * vec2[0], point_C[1] - jet_r/8 * vec2[1]]
    point_D1 = [point_D[0] + jet_r/8 * vec2[0], point_D[1] + jet_r/8 * vec2[1]]
    gui.triangle(point_A1, point_B1, point_C1, color=jet_color)
    gui.triangle(point_A1, point_C1, point_D1, color=jet_color)

def play(ac):
    gui = ti.GUI("Robot is playing rn", res=RESOLUTION, background_color=0xABACAC)
    reset()
    gravity[None] = [0, -1]
    particle_shot_counter = 0
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
            if gui.event.key == 'r':
                reset()
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        # action = [float x.acc, float angular acc, bool jet_on]

        # update jet location
        new_jet_x = ti.math.min(ti.math.max(jet_attributes[None][0] + action[0], jet_r/2), 1 - jet_r/2)
        jet_attributes[None] = [new_jet_x, jet_attributes[None][1]]

        # update jet direction
        jet_attributes[None][1] = ti.math.min(ti.math.max(ti.math.pi, jet_attributes[None][1] + action[1]), 0)
        
        # LMB is pressed, shoot water
        if action[2]:
            particle_shot_counter = shoot_length
        if particle_shot_counter > 0:
            batch = 20
            start = int(random_pos() * (n_particles * (1 - ratio) - batch) + ratio * n_particles)
            for i in range(start, start + batch):
                new_x = random_pos() * jet_r
                new_y = random_pos() * 2 / n_grid
                x[i] = [new_x + jet_attributes[None][0] - jet_r/2, new_y + floor_h] 
                v[i] = [jet_power * ti.math.cos(jet_attributes[None][1]), jet_power * ti.math.sin(jet_attributes[None][1])]
            particle_shot_counter -= 1

        for s in range(int(2e-3 // dt)):
            substep()

        # display particles
        gui.circles(x.to_numpy(),
                    radius=1.5,
                    palette=[0x068587, 0xED553B, 0xEEEE00],
                    palette_indices=material)

        # display ball
        gui.circle(x[0], radius=600*ball_radius, color=0xED553B)
        gui.circle(x[1], radius=1.5, color=0xEEEE00)

        render_jet(gui, jet_attributes)

        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()

def main():
    gui = ti.GUI("Neil's and Robert's fun project uwu", res=RESOLUTION, background_color=0xABACAC)
    reset()
    gravity[None] = [0, -1]
    particle_shot_counter = 0
    for frame in range(20000):
        print(x[0])
        """
        Managing gui inputs
        r - Reset to initial state
        esc - quits the program
        any other button - sets the graviry to 0,1 xd
        LMB - Shoot dihydrogen monoxide from the Particle accelerator
        RMB - Left for debugging haha (secret)
        A/D - move the ~jet~ particle accelerator to the left/right
        """
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'r':
                reset()
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
            elif gui.event.key == 'a':
                tmp = ti.math.max(jet_attributes[None][0] - jet_speed, jet_r/2)
                jet_attributes[None] = [tmp, jet_attributes[None][1]]
            elif gui.event.key == 'd':
                tmp = ti.math.min(jet_attributes[None][0] + jet_speed, 1 - jet_r/2)
                jet_attributes[None] = [tmp, jet_attributes[None][1]]
        
        # update jet direction
        mouse = gui.get_cursor_pos()
        if mouse[1] > 1e-4:
            jet_x = jet_attributes[None][0]
            # Compute the angle between the cursor and the jet's location
            mouse_jet_angle = ti.math.acos((mouse[0]- jet_x)/ti.math.sqrt(ti.math.max(1e-4,(mouse[0] - jet_x)**2 + mouse[1]**2)))
            if mouse_jet_angle - 1.1*jet_angular_speed > jet_attributes[None][1]:
                jet_attributes[None][1] = ti.math.min(ti.math.pi, jet_attributes[None][1] + jet_angular_speed)
            elif mouse_jet_angle + 1.1*jet_angular_speed < jet_attributes[None][1]:
                jet_attributes[None][1] = ti.math.max(0, jet_attributes[None][1] - jet_angular_speed)
        
        # LMB is pressed, shoot water
        if gui.is_pressed(ti.GUI.LMB):
            particle_shot_counter = shoot_length
        if particle_shot_counter > 0:
            batch = 20
            start = int(random_pos() * (n_particles * (1 - ratio) - batch) + ratio * n_particles)
            for i in range(start, start + batch):
                new_x = random_pos() * jet_r
                new_y = random_pos() * 2 / n_grid
                x[i] = [new_x + jet_attributes[None][0] - jet_r/2, new_y + floor_h] 
                v[i] = [jet_power * ti.math.cos(jet_attributes[None][1]), jet_power * ti.math.sin(jet_attributes[None][1])]
            particle_shot_counter -= 1
        
        gui.circle((mouse[0], mouse[1]), color=0x336699, radius=10)

        for s in range(int(2e-3 // dt)):
            substep()

        # display particles
        gui.circles(x.to_numpy(),
                    radius=1.5,
                    palette=[0x068587, 0xED553B, 0xEEEE00],
                    palette_indices=material)

        # display ball
        gui.circle(x[0], radius=600*ball_radius, color=0xED553B)
        gui.circle(x[1], radius=1.5, color=0xEEEE00)

        render_jet(gui, jet_attributes)

        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()