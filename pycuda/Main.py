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
                
    particles2 = []
    
    f = open('test3.txt', 'r').readlines() 
    
    N = len(f)
    for i in range(0 , N): 
        w = f[i].split()
        particles2.append(w)
    
    particles2 = np.float32(particles2)
    
    bodies = setComFrame(particles2)

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


def debugPrintOut(children, quad):
    if quad:
        for n in children:
            pg.draw.rect(screen, (255, 255, 255), [
                         hWidth + int(n.x0 * mag), hHeight + int(n.y0 * mag), n.width * mag, n.height * mag], 1)

    fps = font.render("FPS:" + str(int(clock.get_fps())),
                      True, pg.Color('white'))
    
    screen.blit(fps, (10, 10))

    Ocount = font.render("Bodies:" + str(count), True, pg.Color('white'))
    screen.blit(Ocount, (10, 30))

    masstxt = font.render("Total Mass:" + str(massTot),
                          True, pg.Color('white'))
    screen.blit(masstxt, (10, 50))

    stepsTxt = font.render("Steps done:" + str(steps), True, pg.Color('white'))
    screen.blit(stepsTxt, (10, 70))



def removeBodyOutofBounds(bodies, count):
    massTot = 0.
    
    for cbody in bodies:
        massTot += cbody[mass]
        if (cbody[X] * mag + 960) > 1920 or (cbody[X] * mag + 960) < 0:
            bodies.remove(cbody)
            count -= 1
            
        elif (cbody[Y] * mag + 540) > 1080 or (cbody[Y] * mag + 540) < 0:
            bodies.remove(cbody)
            count -= 1
            
    return bodies, massTot


def writeOut(paths):
    for path in paths:
        file = open(path["name"] + ".dat", "w")
        for i, j in zip(path["x"], path["y"]):
            file.write(str(i) + " " + str(j) + " " + "\n")
        file.close()


if __name__ == '__main__':

    WIDTH = 1920
    HEIGHT = 1080
    hWidth = int(WIDTH / 2)
    hHeight = int(HEIGHT / 2)

    # dt = 1./(10.*365.)
    dt = 0.1
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
    pg.display.set_caption("N-body simulator PYCUDA")

    bodies = setupSolarSystem()

    systemHist = []

    # for tmpBody in bodies:
    #    systemHist.append({"x": [], "y": [], "z": [], "name": tmpBody.name})

    #eval_mod = SourceModule(DenseEvalCode)
    #eval_ker = eval_mod.get_function('dense_eval')
    
    # Before change body type or change it inside the euler_integrator call function.
    
    #integrator = eval_ker(euler_integrator(timestep = dt, bodies=bodies, screen = screen, size = [WIDTH, HEIGHT, hWidth, hHeight], mag = mag) ,  block=(64,1,1), grid=(1,1,1))
    integrator = euler_integrator(timestep = dt, bodies = np.float32(bodies), screen = screen, size = [
                                  WIDTH, HEIGHT, hWidth, hHeight], mag = mag)
    
    count = len(bodies)

    while True:
        screen.fill((0, 0, 0))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                writeOut(systemHist)
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_d:
                    debug = not debug
                elif event.key == pg.K_t:
                    trail = not trail
                elif event.key == pg.K_q:
                    drawQaud = not drawQaud
                elif event.key == pg.K_p:
                    mag += 1
                    integrator.mag = mag
                    mag = max(0, mag)
                elif event.key == pg.K_o:
                    mag -= 1
                    integrator.mag = mag
                    mag = max(0, mag)
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    bodies = addTrojanBody(bodies, count)
                    count += 1
                    systemHist.append(
                        {"x": [], "y": [], "z": [], "name": f"added{count}"})
                    integrator.bodies = bodies
                elif event.button == 4:
                    mag -= 1
                    mag = max(0, mag)
                    integrator.mag = mag
                elif event.button == 5:
                    mag += 1
                    mag = max(0, mag)
                    integrator.mag = mag

            if event.type == pg.VIDEORESIZE:
                oldSurface = screen
                WIDTH = event.w
                hWidth = int(WIDTH / 2)
                HEIGHT = event.h
                hHeight = int(HEIGHT / 2)
                screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
                integrator.screen = screen
                integrator.size = [WIDTH, HEIGHT, hWidth, hHeight]
                del oldSurface

        systemHist, children = runSim(integrator, steps, bodies, systemHist)
        if trail:
            for i in systemHist:
                xtmp = np.array(i["x"][-100:])*mag + hWidth
                ytmp = np.array(i["y"][-100:])*mag + hHeight
                xtmp = xtmp.astype(int)
                ytmp = ytmp.astype(int)
                pts = list(zip(xtmp, ytmp))

                if len(pts) > 1:
                    pg.draw.lines(screen, (255, 255, 255), True, pts, 1)

        clock.tick()
        #bodies, massTot = removeBodyOutofBounds(bodies, count)
        if debug:
            debugPrintOut(children, drawQaud)

        pg.display.flip()
        steps += 1
