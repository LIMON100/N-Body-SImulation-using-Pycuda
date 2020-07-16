#import os
#if (os.system("cl.exe")):
#    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.11.25503\bin\HostX64\x64"
#if (os.system("cl.exe")):
#    raise RuntimeError("cl.exe still not found, path probably incorrect")






import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
import pygame
from MakeTree import qTree, findChild
import pycuda.gpuarray as gpuarray


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



__device__ float calculate_velocity_change_block(float my_planet, float* shared_planets , float* velocities , float* acc2 , float* acc3)
{

    //float velocity [2] = {0.0 , 0.0};
    //float *tempv[2];
    
    
    for(int i = 0; i < blockDim.x; i++) {
    
        *acc2 = calculate_velocity_change_planet(my_planet , shared_planets[i] , acc3);
        
        velocities[0] += acc2[0];
        velocities[1] += acc3[1];
    }

    return *velocities;
}




__global__ void update_velocities(float* planets, float* velocities , float planet_size , float* acc1 , float* acc2 , float* acc3){

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    float my_planet = planets[thread_id];

    __shared__ float shared_planets[BLOCK_SIZE];

    for(int i = 0; i < planet_size; i += blockDim.x) {
     
        shared_planets[threadIdx.x] = planets[i + threadIdx.x];
        
        __syncthreads();
        
        //float *tempv[2];
        *acc1 = calculate_velocity_change_block(my_planet, shared_planets , velocities , acc2 , acc3);
        
        //planets[thread_id]->velocities[0] += acc[0];
        //planets[thread_id]->velocities[1] += acc[1];
        
        velocities[0] += acc1[0];
        velocities[1] += acc1[1];
        
        __syncthreads();
    }
}

""")



#eval_mod_ac = SourceModule(CalculateVelocities)
eval_ker_cal = CalculateVelocities.get_function("update_velocities")




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
        self.bodies_len = len(self.bodies)
        self.bodies_gpu = gpuarray.to_gpu(bodies)
        self.bodies_output_gpu = gpuarray.empty_like(self.bodies_gpu)
        
    
        self.velocities = np.float32(bodies)
        self.velocities_gpu = gpuarray.to_gpu(self.velocities)
        self.velocities_output_gpu = gpuarray.empty_like(self.velocities_gpu)
        
        
        self.acc1 = [0 , 0]
        self.acc1 = np.float32(self.acc1)
        self.acc1_gpu = gpuarray.to_gpu(acc1)
        
        
        self.acc2 = [0 , 0]
        self.acc2 = np.float32(self.acc2)
        self.acc2_gpu = gpuarray.to_gpu(acc2)
        
        self.acc3 = [0 , 0]
        self.acc3 = np.float32(self.acc3_gpu)
        self.acc3_gpu = gpuarray.to_gpu(acc3)
        
        
        self.block = (128 , 1 , 1)
        self.grid = (8 , 1 , 1)
        
    

    def calcVelocity(self):
        
        for idx, targetBody in enumerate(self.bodies):

            
            eval_ker_cal(self.bodies_gpu , self.velocities_output_gpu  , self.bodies_len , self.acc1_gpu , self.acc2_gpu , self.acc3_gpu ,  block = self.block , grid = self.grid)
            
            

    def calcPosition(self):
        """
            updates postions and draws body on pygame screen
            also calculates quadtree for the collection of bodies
            returns list of children in quadtree for drawing
            
        """
        
        body = []
        qt = qTree(1, self.size[1], self.size[0])
        for body in self.bodies:
             
            #eval_ker_pos(self.bodies_gpu , self.velocities_gpu  , block = self.block , grid = self.grid)
            
            body[X] += body[velx] * self.timestep
            body[Y] += body[vely] * self.timestep
            
            
            qt.addPoint(body[X], body[Y], body[mass])
            
            pygame.draw.circle(self.screen,  RED , (self.size[2] + int(body[X] * self.mag), self.size[3] + int(body[Y] * self.mag)) , 2)
            
            
            
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
                bodyLoc["x"].append(bodies[idx].loc.x)
                bodyLoc["y"].append(bodies[idx].loc.y)
            except IndexError:
                continue
    return Histo, c


def addTrojanBody(bodies, count):
    """
    adds a small body where the trojan bodies are
    """
    for tmp in bodies:
        
        vel = [0 , 0]
        
        x1 = tmp[X]
        y1 = tmp[Y]
        
        jupPhi = np.arctan2(y1, x1)
        jupPhi -= np.pi / 6.
        
        vel[X] = tmp[velx] * (np.random.rand() * .1 + .9)
        vel[Y] = tmp[vely] * (np.random.rand() * .1 + .9)
        

        
    r = np.random.rand() * .2 + 5.1
    phi = np.random.rand() * 6 + (jupPhi * (180 / np.pi))
    phi *= (np.pi / 180.)
    pos = point(r * np.cos(phi), r * np.sin(phi))
    bodies.append(body(pos , vel , 1.652e-13))

    return bodies