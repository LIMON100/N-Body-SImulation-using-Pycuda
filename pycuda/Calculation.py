import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
import pygame
from MakeTree import qTree, findChild
import pycuda.gpuarray as gpuarray
import pandas as pd


X = 0
Y = 1
velx = 2
vely = 3
mass = 4
G = 4.*np.pi**2

dT = 0.01
BLOCK_SIZE = 64


WHITE = (255, 255, 255)
RED = (255,   0,   0)





CalculateVelocities = SourceModule("""

#include<math.h>                                   

#define G 4 * pow((3.1416) , 2)
#define dT 0.01

#define BLOCK_SIZE 128
                       
         

__device__ float calculate_velocity_change_planet(float* p, float* q , float* acc3)
{

    
    acc3[0] = (dT * G) * ((q[4] / (pow(sqrt(((q[0] - p[0])) * ((q[0] - p[0])) + (q[1] - p[1]) * (q[1] - p[1])) , 3))) * (q[0] - p[0]));
    acc3[1] = (dT * G) * ((q[4] / (pow(sqrt(((q[0] - p[0])) * ((q[0] - p[0])) + (q[1] - p[1]) * (q[1] - p[1])) , 3))) * (q[1] - p[1]));
    
    return *acc3;
}



__device__ float calculate_velocity_change_block(float* my_planet, float* shared_planets , float* acc2 , float* acc3)
{

    float *tempv = acc2;
    
    
    for(int i = 0; i < blockDim.x; i++) {
    
        *acc2 = calculate_velocity_change_planet(my_planet , (&shared_planets[i]) , acc3);
        
        tempv[0] += acc2[0];
        tempv[1] += acc2[1];
    }

    return *tempv;
}




__global__ void update_velocities(float* planets, float* velocities , float planet_size , float* acc1 , float* acc2 , float* acc3)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    float my_planet = planets[thread_id];

    __shared__ float shared_planets[BLOCK_SIZE];

    for(int i = 0; i < planet_size; i += blockDim.x) {
     
        shared_planets[threadIdx.x] = planets[i + threadIdx.x];
        
        __syncthreads();
        
      
        *acc1 = calculate_velocity_change_block(&my_planet, shared_planets , acc2 , acc3);

        
        velocities[thread_id] += acc1[0];
        velocities[thread_id] += acc1[1];
        
        __syncthreads();
    }
    
    
}

""")



#eval_mod_ac = SourceModule(CalculateVelocities)
eval_ker_cal = CalculateVelocities.get_function("update_velocities")



DensePosition = SourceModule("""
                             
#define dT 0.1
                             
__global__ void update_positions(float* planets, float* velocities)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    planets[0] += (velocities[2]) * (dT);
    planets[1] += (velocities[3]) * (dT);
    
    
}

""")


eval_ker_pos = DensePosition.get_function("update_positions")



class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



class body:
    """loc in units of AU
        mass in solar mass units
        velociy in AU/yr
    """

    
    def __init__(self, position , velocity):
        
        #self.loc = loc
        #self.mass = mass
        #self.vel = vel
        #self.name = name
        #self.colour = colour
        #self.dispMass = dispMass
        
        #self.mass = mass
        self.position = position
        self.velocity = velocity
        



class euler_integrator:

    def __init__(self, timestep, bodies, screen, size, mag):
        
        
        self.timestep = timestep
        self.bodies = bodies
        self.screen = screen
        self.size = size
        self.mag = mag
        
        self.bodies = np.float32(bodies)
        self.bodies_gpu = gpuarray.to_gpu(bodies)
        self.bodies_output_gpu = gpuarray.empty_like(self.bodies_gpu)
        
        
        self.bodies_len = len(self.bodies)
        self.bodies_len = np.float32(self.bodies_len)
        
        
        self.velocities = np.float32(bodies)
        self.velocities_gpu = gpuarray.to_gpu(self.velocities)
        self.velocities_output_gpu = gpuarray.empty_like(self.velocities_gpu)
        
        
        self.acc1 = [0 , 0]
        self.acc1 = np.float32(self.acc1)
        self.acc1_gpu = gpuarray.to_gpu(self.acc1)
        
        
        self.acc2 = [0 , 0]
        self.acc2 = np.float32(self.acc2)
        self.acc2_gpu = gpuarray.to_gpu(self.acc2)
        
        
        self.acc3 = [0 , 0]
        self.acc3 = np.float32(self.acc3)
        self.acc3_gpu = gpuarray.to_gpu(self.acc3)
        
        
        self.block = (70 , 1 , 1)
        self.grid = (int(np.ceil(len(self.bodies) / 70)), 1,1)
        #self.grid = (1 , 1 , 1)
        
    

    def calcVelocity(self):
        
        for idx, targetBody in enumerate(self.bodies):

            eval_ker_cal(self.bodies_gpu , self.velocities_gpu  , self.bodies_len , self.acc1_gpu , self.acc2_gpu , self.acc3_gpu ,  block = self.block , grid = self.grid)
            
        
        x_test = self.velocities_gpu.get()
        #print(x_test)
        return x_test
        

    def calcPosition(self):
       
        qt = qTree(1, self.size[1], self.size[0])
        
        for body in self.bodies:
            
            
            
            #body[0] += body[2] * self.timestep
            #body[1] += body[3] * self.timestep
            
            
            body_out = np.float32(body)
            body_out_gpu = gpuarray.to_gpu(body_out)
            
            
            vel_out = self.calcVelocity()
            vel_out = np.float32(vel_out)
            vel_out_gpu = gpuarray.to_gpu(vel_out)
            
            
            eval_ker_pos(body_out_gpu , vel_out_gpu  , block = self.block , grid = self.grid)
            
            body = body_out_gpu.get()
            body = np.float32(body)
            print(body[0])
            
            #body_out = self.calcVelocity()
            #body_out = np.float32(body_out)
            #print(body_out)
            
            #body = body_out
            
            
            #body_new = pd.DataFrame(body)
            #body_new = body_new.fillna(1.0)
            
            
            #body_new[0][0] += body_new[2][2] * self.timestep
            #body_new[1][1] += body_new[3][3] * self.timestep
            
            #body = np.float32(body)
            #body_gpu = gpuarray.to_gpu(body)
            
            #print(abs(body))
            #eval_ker_pos(body_gpu , self.velocities_output_gpu  , block = self.block , grid = self.grid)
            #body = abs(body)
            
            #body = body_gpu.get()
            #print(body[2])
            
            #body = np.int32(body)
            
            #qt.addPoint(body_new[X][X], body_new[Y][Y], body_new[mass][mass])
            qt.addPoint(abs(body[X]), abs(body[Y]), abs(body[mass]))
            
            pygame.draw.circle(self.screen,  WHITE , (self.size[2] + int(abs(body[X]) * self.mag), self.size[3] + int(abs(body[Y]) * self.mag)) , 2)
            
            
            
        qt.subDivide()
        c = findChild(qt.root)
        del qt
        return c
    

    def doStep(self):
        self.calcVelocity()
        c = self.calcPosition()
        return c


def runSim(integrator, steps, bodies, Histo):

    c = integrator.doStep()
    
    if steps % 200 == 0:
        for idx, bodyLoc in enumerate(Histo):
            
            try:
                bodyLoc["x"].append(bodies[idx].bodyLoc[0])
                bodyLoc["y"].append(bodies[idx].bodyLoc[1])
            except IndexError:
                continue
    return Histo, c