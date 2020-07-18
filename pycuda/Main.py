# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 22:33:01 2020

@author: limon
"""


import pygame as pg
import numpy as np
import sys
import pycuda.gpuarray as gpuarray

from Calculation import point, body
#from Calculation import euler_integrator, addTrojanBody, runSim
from Calculation import euler_integrator, runSim
#from Calculation import addTrojanBody, runSim
#from Calculation import DenseEvalCode


X = 0
Y = 1
velx = 2
vely = 3
mass = 4

WHITE = (255, 255, 255)


def setupSolarSystem():
    particles = []
    filename = ('test2.txt')

    with open(filename) as f:
        for line in f:

            if line[0] == '#':
                continue

            else:
                x, y, vx, vy, m = line.split(' ')
                particles.append(
                    [float(x), float(y), float(vx), float(vy), float(m)])


    
    #particles = np.float32(particles)
    #particles_gpu = gpuarray.to_gpu(particles)
    
    bodies = setComFrame(particles)

    return bodies


def setComFrame(bodies):

    totMass = 0
    for body in bodies:
        totMass += body[mass]
        
    rx = 0
    ry = 0
    vx = 0
    vy = 0
    
    for body in bodies:
        rx += (body[mass] / totMass) * body[X]
        ry += (body[mass] / totMass) * body[Y]
        vx += (body[mass] / totMass) * body[X]
        vy += (body[mass] / totMass) * body[Y]


    for body in bodies:
        body[X] -= rx
        body[Y] -= ry
        body[X] -= vx
        body[Y] -= vy

    return bodies




if __name__ == '__main__':

    WIDTH = 1920
    HEIGHT = 1080
    
    hWidth = int(WIDTH / 2)
    hHeight = int(HEIGHT / 2)

    # dt = 1./(10.*365.)
    dt = 0.01
    debug = False
    drawQaud = False
    steps = 0
    trail = False
    mag = 50

    pg.init()
    font = pg.font.Font(None, 30)
    clock = pg.time.Clock()
    size = WIDTH, HEIGHT
    screen = pg.display.set_mode(size, pg.RESIZABLE)
    pg.display.set_caption("N-body simulator using PYCUDA")

    bodies = setupSolarSystem()

    systemHist = []


    integrator = euler_integrator(timestep = dt, bodies = np.float32(bodies), screen = screen, size = [
                                  WIDTH, HEIGHT, hWidth, hHeight], mag = mag)
    
    count = len(bodies)

    while True:
        
        screen.fill((0, 0, 0))
        
        systemHist, children = runSim(integrator, steps, bodies, systemHist)
        
        clock.tick()
        
        steps += 1