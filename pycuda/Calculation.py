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


WHITE = (255, 255, 255)
RED = (255,   0,   0)
LAWN_GREEN = (124 , 252 , 0)




DenseEvalAcelretion = SourceModule("""
                    
                       
#include<math.h>                     

__global__ void dense_aceleration(float *targetBody , float *body , float *acc , float mag)
{
     
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    mag = 1. / pow(sqrt((pow((targetBody[0] - body[0]) , 2)) + (pow((targetBody[1] - body[1]) , 2))) , 3);
    
    acc[0] += body[4] * ((body[1] - targetBody[0]) * (mag));
    acc[1] += body[4] * ((body[1] - targetBody[0]) * (mag));
    
    
    //return acc;

}
""")



#eval_mod_ac = SourceModule(DenseEvalAcelretion)
eval_ker_ac = DenseEvalAcelretion.get_function('dense_aceleration')





DenseEvalVeclocity = SourceModule("""
                    

#define G 4 * pow((3.1416) , 2)
#define timestep 0.1

                                         

__global__ void dense_velocity(float *acc , float *targetBody)
{
    
 
    targetBody[2] += (G * timestep) * (acc[0]);
    targetBody[3] += (G * timestep) * (acc[1]);
    
    
}
""")



#eval_mod_ac = SourceModule(DenseEvalAcelretion)
eval_ker_vt = DenseEvalVeclocity.get_function('dense_velocity')





DenseEvalPosition = SourceModule("""
                    
#define timestep 0.1

                                         

__global__ void dense_pos(float *body)
{
     
     body[0] += (body[2]) * timestep;
     body[1] += (body[3]) * timestep;
    
    
}
""")



#eval_mod_ac = SourceModule(DenseEvalAcelretion)
eval_ker_pos = DenseEvalPosition.get_function('dense_pos')







class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class body:
    """loc in units of AU
        mass in solar mass units
        velociy in AU/yr
    """

    def __init__(self,  position, velocity):

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
        
        self.mag = np.float32(mag)
        
        self.block = (128 , 1 , 1)
        self.grid = (8 , 1 , 1)
        
     
        self.bodies = np.float32(bodies)
        self.bodies_gpu = gpuarray.to_gpu(bodies)
        self.bodies_output_gpu = gpuarray.empty_like(self.bodies_gpu)
        
        self.acc = [0 , 0]
        self.acc = np.float32(self.acc)
        self.acc_gpu = gpuarray.to_gpu(self.acc)
        self.acc_output_gpu = gpuarray.empty_like(self.acc_gpu)
        
        
        

    def calcAccleration(self, bodyIdx):    
        
        targetBody = self.bodies[bodyIdx]
        
        targetBody = np.float32(targetBody)
        targetBody_gpu = gpuarray.to_gpu(targetBody)
        targetBody_output_gpu = gpuarray.empty_like(targetBody_gpu)
   
        
        for idx, body in enumerate(self.bodies):
            
            body = np.float32(body)
            body_gpu = gpuarray.to_gpu(body)
            body_output_gpu = gpuarray.empty_like(body_gpu)
            
            if idx != bodyIdx:
                               
                
                eval_ker_ac(targetBody_gpu , body_gpu , self.acc_gpu , self.mag , block = self.block , grid = self.grid)
        
        
        aclarte_out = self.acc_gpu.get()
        #print(aclarte_out)
        return aclarte_out
        
        

    def calcVelocity(self):
        
        for idx, targetBody in enumerate(self.bodies):
            
            acc = self.calcAccleration(idx)
            acc = np.float32(acc)
            acc_gpu = gpuarray.to_gpu(acc)
            
            
            targetBody = np.float32(targetBody)
            targetBody_gpu = gpuarray.to_gpu(targetBody)
            #body_output_gpu = gpuarray.empty_like(targetBody_gpu)
            
            eval_ker_vt(acc_gpu , targetBody_gpu , block = self.block , grid = self.grid)
            
        
        
        ac_out = acc_gpu.get()
        #print(ac_out)
            
            
            
            

    def calcPosition(self):
        """
            updates postions and draws body on pygame screen
            also calculates quadtree for the collection of bodies
            returns list of children in quadtree for drawing
        """
        qt = qTree(1, self.size[1], self.size[0])
        
        for body in self.bodies:
            
            #body[0] += (body[2]) * self.timestep;
            #body[1] += (body[3]) * self.timestep;
            
            #print(body)
            
            body = np.float32(body)
            body_gpu = gpuarray.to_gpu(body)
            
            body_gpu_output = gpuarray.empty_like(body_gpu)
            
            eval_ker_pos(body_gpu , block = self.block , grid = self.grid)
            
            body_out = body_gpu.get()
            
            #body = np.float32(body)
            
            print(body_out)
            
            qt.addPoint(body_out[X], body_out[Y], body_out[mass])
            
            pygame.draw.circle(self.screen,  LAWN_GREEN , (self.size[2] + int(
                body_out[X] * self.mag), self.size[3] + int(body_out[Y] * self.mag)), 2)
            
            

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