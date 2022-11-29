import taichi as ti
import json
import math
from rl_functions import *

ti.init(arch=ti.gpu)  # Try to run on GPU

@ti.data_oriented
class Environment:
    
    def __init__(self, args:dict):
        for attr, value in args.items():
            setattr(self, attr, value)
        self.n_particles, self.n_grid = self.particle_multiplier * self.quality**2, 64 * self.quality
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.p_vol, self.p_rho = (self.dx * 0.5)**2, 1
        self.p_mass = self.p_vol * self.p_rho
        self.E, self.nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
            (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

        self.x = ti.Vector.field(2, dtype=float, shape=self.n_particles)  # position
        self.v = ti.Vector.field(2, dtype=float, shape=self.n_particles)  # velocity
        self.C = ti.Matrix.field(2, 2, dtype=float,
                            shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float,
                            shape=self.n_particles)  # deformation gradient
        self.material = ti.field(dtype=int, shape=self.n_particles)  # material id
        self.Jp = ti.field(dtype=float, shape=self.n_particles)  # plastic deformation
        self.grid_v = ti.Vector.field(2, dtype=float,
                                shape=(self.n_grid, self.n_grid))  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))  # grid node mass
        self.gravity = ti.Vector.field(2, dtype=float, shape=())
        self.attractor_strength = ti.field(dtype=float, shape=())
        self.attractor_pos = ti.Vector.field(2, dtype=float, shape=())
        self.jet_attributes = ti.Vector.field(2, dtype=float, shape=())
        print("\n_________________________________\nThe environment settings are:\n" + str(self.__dict__) + "\n_________________________________")

    @ti.kernel
    def substep(self):
        for i, j in self.grid_m:
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0
        for p in self.x:  # Particle state update and scatter to grid (P2G)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            # deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Hardening coefficient: snow gets harder when compressed
            h = max(0.1, min(5, ti.exp(10 * (1.0 - self.Jp[p]))))
            if self.material[p] == 1:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if self.material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                1 + 4.5e-3)  # Plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.material[p] == 0:
                # Reset deformation gradient to avoid numerical instability
                self.F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif self.material[p] == 2:
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose(
            ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx**2) * stress
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:  # No need for epsilon here
                # Momentum to velocity
                self.grid_v[i, j] = (1 / self.grid_m[i, j]) * self.grid_v[i, j]
                self.grid_v[i, j] += self.dt * self.gravity[None] * 30  # gravity
                dist = self.attractor_pos[None] - self.dx * ti.Vector([i, j])
                self.grid_v[i, j] += \
                    dist / (0.01 + dist.norm()) * self.attractor_strength[None] * self.dt * 100
                if i < 3 and self.grid_v[i, j][0] < 0:
                    self.grid_v[i, j][0] = 0  # Boundary conditions
                if i > self.n_grid - 3 and self.grid_v[i, j][0] > 0:
                    self.grid_v[i, j][0] = 0
                if j < 3 and self.grid_v[i, j][1] < 0:
                    self.grid_v[i, j][1] = 0
                if j > self.n_grid - 3 and self.grid_v[i, j][1] > 0:
                    self.grid_v[i, j][1] = 0
        for p in self.x:  # grid to particle (G2P)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]  # advection


    @ti.kernel
    def reset(self):
        """
        create random position, for the ball we randomize polar coords and recast them to normal ones 
        """
        group_size = self.n_particles // 2
        #Init ball
        ratio = 0.4
        for i in range(ratio*group_size):
            radius = ti.random() * self.ball_radius
            degree = ti.random() * 360 
            spread = 0.1
            new_x = ti.math.cos(degree)*radius*spread + 0.5
            new_y = ti.math.sin(degree)*radius*spread + 0.5
            self.x[i] = [new_x, new_y]
            self.material[i] = 1 # 0: fluid 1: jelly 2: snow
            self.v[i] = [0, 0]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1
            self.C[i] = ti.Matrix.zero(float, 2, 2)
        #Init water
        #TODO: think about paramaters
        for i in range(ratio*group_size, self.n_particles):
            radius = ti.random() * self.ball_radius
            degree = ti.random() * 2*ti.math.pi
            spread = 0.1
            new_x = ti.math.cos(degree)*radius*spread + 0.5
            new_y = ti.math.sin(degree)*radius*spread + 1e-4
            self.x[i] = [new_x, new_y]
            self.material[i] = 0 # 0: fluid 1: jelly 2: snow
            self.v[i] = [0, 0]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1
            self.C[i] = ti.Matrix.zero(float, 2, 2)  
        #Init jet
        self.jet_attributes[None] = [0.5, ti.math.pi/2]

    @ti.kernel
    def random_pos(self) -> ti.f32:
        return ti.random()

    def play(self,ac, display=True):
        print(type(ac))
        gui = ti.GUI("Señor robot is playing rn", res=512, background_color=0xABACAC)
        self.reset()
        self.gravity[None] = [0, -1]
        particle_shot_counter = 0
        for frame in range(20000):
            """
            Managing gui inputs
            r - Reset to initial state
            esc - quits the program
            any other button - sets the graviry to 0,1 xd
            LMB - Shoot dihydrogen monoxide from the Particle accelerator
            RMB - Left for debugging haha (secret)
            A/D - move the ~jet~ particle accelerator to the left/right
            """
            state = None #TODO create MDP
            action = ac.step(state) #TODO figure out params
            print("The chosen action was: " + action)
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == 'r':
                    self.reset()
                elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    break
                elif action == 'a':
                    tmp = ti.math.max(self.jet_attributes[None][0]-self.jet_speed, 0)
                    self.jet_attributes[None] = [tmp, self.jet_attributes[None][1]]
                elif action == 'd':
                    tmp = ti.math.min(self.jet_attributes[None][0]+self.jet_speed, 1)
                    self.jet_attributes[None] = [tmp, self.jet_attributes[None][1]]
            mouse = gui.get_cursor_pos()
            jet_x = self.jet_attributes[None][0]
            #Compute the angle between the cursor and the jet's location
            mouse_jet_angle = ti.math.acos((mouse[0]- jet_x)/ti.math.sqrt(ti.math.max(1e-4,(mouse[0]- jet_x)**2 + mouse[1]**2)))
            if mouse_jet_angle > self.jet_attributes[None][1]:
                self.jet_attributes[None][1] = ti.math.min(ti.math.pi, self.jet_attributes[None][1]+self.jet_angular_speed)
            elif mouse_jet_angle < self.jet_attributes[None][1]:
                self.jet_attributes[None][1] = ti.math.max(self.jet_attributes[None][1]-self.jet_angular_speed, 0)
            #LMB is pressed, shoot water
            if action == 'LMB':
                particle_shot_counter = self.shoot_length
            if particle_shot_counter > 0 :
                self.batch = 20
                start = self.random_pos() * (self.n_particles - 400 - self.batch) + 400 #TODO Parametrize number of ball particles
                for i in range(int(start), int(start + self.batch)):
                    new_x = self.random_pos()*self.jet_r
                    self.x[i] = [new_x + self.jet_attributes[None][0],1e-4] 
                    self.v[i] = [self.jet_power*ti.math.cos(self.jet_attributes[None][1]), self.jet_power*ti.math.sin(self.jet_attributes[None][1])]
                particle_shot_counter -= 1
            gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
            self.attractor_pos[None] = [mouse[0], mouse[1]]
            self.attractor_strength[None] = 0
            if gui.is_pressed(ti.GUI.RMB):
                self.attractor_strength[None] = -1 #shoot air
            for s in range(int(2e-3 // self.dt)):
                #pass
                self.substep()
            gui.circles(self.x.to_numpy(),
                        radius=1.5,
                        palette=[0x068587, 0xED553B, 0xEEEE00],
                        palette_indices=self.material)

            # Change to gui.show(f'{frame:06d}.png') to write images to disk
            if display:
                gui.show()





    def main(self):
        gui = ti.GUI("Neil's and Robert's fun project uwu", res=512, background_color=0xABACAC)
        self.reset()
        self.gravity[None] = [0, -1]
        particle_shot_counter = 0
        for frame in range(20000):
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
                    self.reset()
                elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    break
                elif gui.event.key == 'a':
                    tmp = ti.math.max(self.jet_attributes[None][0]-self.jet_speed, 0)
                    self.jet_attributes[None] = [tmp, self.jet_attributes[None][1]]
                elif gui.event.key == 'd':
                    tmp = ti.math.min(self.jet_attributes[None][0]+self.jet_speed, 1)
                    self.jet_attributes[None] = [tmp, self.jet_attributes[None][1]]
            mouse = gui.get_cursor_pos()
            jet_x = self.jet_attributes[None][0]
            #Compute the angle between the cursor and the jet's location
            mouse_jet_angle = ti.math.acos((mouse[0]- jet_x)/ti.math.sqrt(ti.math.max(1e-4,(mouse[0]- jet_x)**2 + mouse[1]**2)))
            if mouse_jet_angle > self.jet_attributes[None][1]:
                self.jet_attributes[None][1] = ti.math.min(ti.math.pi, self.jet_attributes[None][1]+self.jet_angular_speed)
            elif mouse_jet_angle < self.jet_attributes[None][1]:
                self.jet_attributes[None][1] = ti.math.max(self.jet_attributes[None][1]-self.jet_angular_speed, 0)
            #LMB is pressed, shoot water
            if gui.is_pressed(ti.GUI.LMB):
                particle_shot_counter = self.shoot_length
            if particle_shot_counter > 0 :
                self.batch = 20
                start = self.random_pos() * (self.n_particles - 400 - self.batch) + 400 #TODO Parametrize number of ball particles
                for i in range(int(start), int(start + self.batch)):
                    new_x = self.random_pos()*self.jet_r
                    self.x[i] = [new_x + self.jet_attributes[None][0],1e-4] 
                    self.v[i] = [self.jet_power*ti.math.cos(self.jet_attributes[None][1]), self.jet_power*ti.math.sin(self.jet_attributes[None][1])]
                particle_shot_counter -= 1
            gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
            self.attractor_pos[None] = [mouse[0], mouse[1]]
            self.attractor_strength[None] = 0
            if gui.is_pressed(ti.GUI.RMB):
                self.attractor_strength[None] = -1 #shoot air
            for s in range(int(2e-3 // self.dt)):
                #pass
                self.substep()
            gui.circles(self.x.to_numpy(),
                        radius=1.5,
                        palette=[0x068587, 0xED553B, 0xEEEE00],
                        palette_indices=self.material)

            # Change to gui.show(f'{frame:06d}.png') to write images to disk
            gui.show()

if __name__ == "__main__":
    with open('env_setting.json', 'r') as file:
        setting = json.load(file)
    setting.update({'jet_angular_speed': (setting['jet_speed'] * math.pi)})
    env = Environment(setting)
    env.main()
