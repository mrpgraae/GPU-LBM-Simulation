import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as num
import pygame as pg
import math
import sys

# Load the cuda code
headerFile = open('./cudaCode.cuh')
functionsFile = open('./cudaCode.cu')

mod = SourceModule(headerFile.read() + functionsFile.read())

# Get the functions.
GPUcalcRho = mod.get_function("calcRhoGPU")
GPUcalcVel = mod.get_function("calcVelGPU")
GPUcalcEquilibrium = mod.get_function("calcEquilibriumGPU")
GPUBGKCollide = mod.get_function("BGKCollideGPU")
GPUbounceBack = mod.get_function("bounceBackGPU")
GPUstream = mod.get_function("streamGPU")
GPUstreamIn = mod.get_function("streamInGPU")
GPUstreamOut = mod.get_function("streamOutGPU")

# Velocity weights
w = num.array(num.ones(9), dtype=num.float32)
w[0] *= 4./9.
w[1:5] *= 1./9.
w[5:9] *= 1./36.

q = 9

blitSize = 2

# Node velocities
vel = num.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
hostVelCol0 = num.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=num.float32)
hostVelCol1 = num.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=num.float32)
hostVelColInt0 = num.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=num.int32)
hostVelColInt1 = num.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=num.int32)

# Array for mapping bounceback velocities.
bounceBackVel = num.array([0,3,4,1,2,7,8,5,6], dtype=num.int32)

def calcEquilibrium(velDir, rho, ux, uy, uSqr):
  velDot = vel[velDir, 0]*ux + vel[velDir, 1]*uy
  return rho*w[velDir]*(1. + 3.*velDot + 4.5*velDot*velDot - 1.5*uSqr)


def calcEquilibriumVector(rho, ux, uy, uSqr):
  velDot = hostVelCol0*ux + hostVelCol1*uy
  return rho * w * (1. + 3.*velDot + 4.5*velDot*velDot - 1.5*uSqr)

def calcRho(node):
  rho = 0
  for i in xrange(q):
    rho += node.densities[i]
  return rho

def calcVelocity(node, rho):
  ux = (node.densities[1] + node.densities[5] + node.densities[8]
       - (node.densities[3] + node.densities[6] + node.densities[7]))/rho
  uy = (node.densities[6] + node.densities[2] + node.densities[5] 
       - (node.densities[7] + node.densities[4] + node.densities[8]))/rho
  return num.array([ux,uy], dtype=float)

class Simulation():

  def __init__(self, hostLx, hostLy, obst_x, obst_y, obst_r, umax):
    self.umax = umax
    self.lx = hostLx
    self.ly = hostLy
    self.lyAll = hostLy + 2
    self.lyB = hostLy + 1
    self.nNodes = hostLx*hostLy+2*hostLx
    self.obst_x = obst_x
    self.obst_y = obst_y
    self.obst_r = obst_r
    self.grid_surf = pg.display.set_mode((self.lx*blitSize,(self.lyB-1)*blitSize))
    self.node_surf = pg.Surface((blitSize,blitSize))
    self.data = num.zeros(self.nNodes*q, dtype=num.float32)
    self.nodeMap = num.zeros(self.nNodes*q, dtype=num.int32)
    self.inFlow = num.zeros(self.lyAll*q, dtype=num.float32)
    self.usqrsBuffer = num.zeros(self.nNodes, dtype=num.float32)

  def getXYN(self, x, y, n):
    return n*self.nNodes + y*self.lx + x

  def calcIniVelocity(self, y):
    y = float(y)
    l = float(self.ly-1)
    vel = (4.*self.umax * (l*y-y*y)) / (l*l)
    return vel

  def initializeLattice(self):

    # Set nodes within cylinder to solid. And color them Red.
    for i in xrange(self.lx):
      for j in xrange(self.lyAll):
        if (i - self.obst_x)*(i - self.obst_x) + (j - self.obst_y)*(j - self.obst_y) <= self.obst_r*self.obst_r:
          self.nodeMap[self.getXYN(i,j,0)] = 1
          self.grid_surf.blit(self.node_surf, (i*blitSize,(j-1)*blitSize))

    # Set nodes on north and south boundaries to solid.
    for i in xrange(self.lx):
      self.nodeMap[self.getXYN(i,1,0)] = 1
      self.nodeMap[self.getXYN(i,self.ly,0)] = 1

    # Set node velocities
    for i in xrange(self.lx):
      for j in xrange(1,self.lyB):
        for c in xrange(q):
          vel = self.calcIniVelocity(j)
          uSqr = vel * vel
          self.data[self.getXYN(i,j,c)] = calcEquilibrium(c, 1., vel, 0., uSqr)

    for i in xrange(self.lyAll):
      for c in xrange(q):
        self.inFlow[self.lyAll*c+i] = self.data[self.getXYN(0,i,c)]

  def updateSurface(self, colParam=0):
    for i in xrange(self.lx):
      for j in xrange(1,self.lyB):
        node = self.getXYN(i,j,0)
        uSqr = self.usqrsBuffer[node]
        if self.nodeMap[node] == 0:
          if colParam == 0:
            if math.isnan(uSqr):
              col = (255,0,0)
            else:
              col = (0,math.sqrt(uSqr)*800+2,0)
              if col[1] > 255: col = (0,0,255)
          if colParam == 1:
            rho = calcRho(self.lattice[i,j])
            rho -= 0.5
            rho *= 2
            if rho > 1:
              col = (255,)*3
            elif rho < 0:
              col = (255,0,0)
            else:
              col = (rho*255,)*3
        else:
          continue 
        self.node_surf.fill(col)
        self.grid_surf.blit(self.node_surf, (i*blitSize,(j-1)*blitSize))
    pg.display.flip()  

def main(lx, ly, numIters):

  # Simulation dimensions
  hostLx = int(lx)
  hostLy = int(ly)
  lyB = hostLy+1
  lyAll = hostLy+2
  hostnNodes = hostLx*hostLy+2*hostLx
  hostGridIter = hostLx*hostLy+hostLx

  # Buffer for displaying the simulation
  usqrsBuffer = num.zeros(hostnNodes, dtype=num.float32)
  testData = num.zeros(hostnNodes*q, dtype=num.float32)

  # Obstacle location
  obst_x = hostLx/6
  obst_y = hostLy/2
  obst_r = hostLy/9

  # Fluid parameters
  Re = 130
  V     = 0.066
  umax = (3./2.)*V                      
  nu    = (V*obst_r)/Re
  hostOmega = 1.0 / (3.*nu+0.5);
  print hostOmega

  # Make 1-element numPy arrays to please pyCudas memcpy.
  hostOmegaArr = num.array([hostOmega], dtype=num.float32)
  hostLxArr = num.array([hostLx], dtype=num.int32)
  hostLyArr = num.array([hostLy], dtype=num.int32)
  hostGridIterArr = num.array([hostGridIter], dtype=num.int32)
  hostnNodesArr = num.array([hostnNodes], dtype=num.int32)

  print hostOmegaArr
  print hostGridIterArr
  print hostnNodesArr

  sim = Simulation(hostLx, hostLy, obst_x, obst_y, obst_r, umax)

  print "Setting up the lattice."
  sim.initializeLattice()
  print "Lattice setup complete."

  # Make PyCuda set up the memory space on the gpu.
  inData = cuda.mem_alloc(sim.data.nbytes)
  outData = cuda.mem_alloc(sim.data.nbytes)
  GPUinFlow = cuda.mem_alloc(sim.inFlow.nbytes)
  GPUnodeMap = cuda.mem_alloc(sim.nodeMap.nbytes)
  usqrs = cuda.mem_alloc(hostnNodes*4)
  vels = cuda.mem_alloc(hostnNodes*4*2)
  equis = cuda.mem_alloc(q*hostnNodes*4)
  rhos = cuda.mem_alloc(hostnNodes*4)

  cuda.memcpy_htod(inData, sim.data)
  cuda.memcpy_htod(GPUinFlow, sim.inFlow)
  cuda.memcpy_htod(GPUnodeMap, sim.nodeMap)

  # Initialize constant memory on the GPU
  velCol0Addr = mod.module.get_global('velCol0')
  velCol1Addr = mod.get_global('velCol1')
  velColInt0Addr = mod.get_global('velColInt0')
  velColInt1Addr = mod.get_global('velColInt1')
  weightsAddr = mod.get_global('weights')
  bounceBackMapAddr = mod.get_global('bounceBackMap')
  omegaAddr = mod.get_global('omega')
  lxAddr = mod.get_global('lx')
  lyAddr = mod.get_global('ly')
  yOffsetAddr = mod.get_global('yOffset')
  gridIterAddr = mod.get_global('gridIter')
  nNodesAddr = mod.get_global('nNodes')

  cuda.memcpy_htod(velCol0Addr[0], hostVelCol0)
  cuda.memcpy_htod(velCol1Addr[0], hostVelCol1)
  cuda.memcpy_htod(velColInt0Addr[0], hostVelColInt0)
  cuda.memcpy_htod(velColInt1Addr[0], hostVelColInt1)
  cuda.memcpy_htod(weightsAddr[0], w)
  cuda.memcpy_htod(bounceBackMapAddr[0], bounceBackVel)
  cuda.memcpy_htod(omegaAddr[0], hostOmegaArr)
  cuda.memcpy_htod(lxAddr[0], hostLxArr)
  cuda.memcpy_htod(lyAddr[0], hostLyArr)
  cuda.memcpy_htod(yOffsetAddr[0], hostLxArr)
  cuda.memcpy_htod(gridIterAddr[0], hostGridIterArr)
  cuda.memcpy_htod(nNodesAddr[0], hostnNodesArr)

  # Prepare the functions for faster execution.
  GPUcalcRho.prepare("PP")
  GPUcalcVel.prepare("PPPP")
  GPUcalcEquilibrium.prepare("PPPP")
  GPUBGKCollide.prepare("PPP")
  GPUbounceBack.prepare("PP")
  GPUstream.prepare("PP")
  GPUstreamIn.prepare("PPP")
  GPUstreamOut.prepare("PPP")

  # Get device specific attributes.
  device=cuda.Device(0)

  minNrBlocks = device.multiprocessor_count
  maxNrThreads = device.max_threads_per_block

  print "Number of streaming processors:", minNrBlocks
  print "Maximum number of threads per block:", maxNrThreads

  # Compute optimal kernel launch parameters.
  totalThreads = int(hostnNodes-2*lx)
  totalBlocks = int(math.ceil(totalThreads / maxNrThreads))
  threadsPerBlock = int(math.ceil(totalThreads / totalBlocks))

  if totalBlocks < minNrBlocks:
    totalBlocks = int(minNrBlocks)
    threadsPerBlock = int(math.ceil(hostnNodes / totalBlocks))

  lyThreads = int(hostLy)
  lyBlocks = int(math.ceil(lyThreads / maxNrThreads))

  if lyBlocks < minNrBlocks:
    lyBlocks = int(minNrBlocks)
    lyThreads = int(math.ceil(hostLy / lyBlocks))

  print "Running with:", totalBlocks, "blocks."
  print "Running with:", threadsPerBlock, "threads per block."

  # Main simulation loop
  usqrsOneIter = num.zeros(hostnNodes, dtype=num.float32)
  velsBuffer = num.zeros(hostnNodes*2, dtype=num.float32)
  velsOneIter = num.zeros(hostnNodes*2, dtype=num.float32)
  inDataBuffer = num.zeros(hostnNodes*q, dtype=num.float32)
  outDataBuffer = num.zeros(hostnNodes*q, dtype=num.float32)

  i = 0

  while True:

    GPUcalcRho.prepared_call((totalBlocks,1), (threadsPerBlock,1,1), inData, rhos)

    GPUcalcVel.prepared_call((totalBlocks,1), (threadsPerBlock,1,1), inData, vels, usqrs, rhos)

    GPUcalcEquilibrium.prepared_call((totalBlocks,1), (threadsPerBlock,1,1), equis, rhos, vels, usqrs)

    GPUBGKCollide.prepared_call((totalBlocks,1), (threadsPerBlock,1,1), inData, equis, GPUnodeMap)

    GPUbounceBack.prepared_call((totalBlocks,1), (threadsPerBlock,1,1), inData, GPUnodeMap)

    GPUstream.prepared_call((totalBlocks,1), (threadsPerBlock,1,1), outData, inData)

    GPUstreamIn.prepared_call((lyBlocks,1), (lyThreads,1,1), outData, inData, GPUinFlow)

    GPUstreamOut.prepared_call((lyBlocks,1), (lyThreads,1,1), outData, inData, GPUinFlow)

    pycuda.autoinit.context.synchronize()

    inData, outData = outData, inData

    if i%100 == 0:
      cuda.memcpy_dtoh(sim.usqrsBuffer, usqrs)
      sim.updateSurface()
      print "Iteration nr. ", i

    i += 1

    if i == numIters:
      break

if __name__ == "__main__":

  if len(sys.argv) < 4:
    print "Needs at least 3 arguments (x dimension, y dimension, number of iterations)"
    sys.exit()
  else:
    for e in sys.argv[1:]:
      if not e.isdigit():
        print "All arguments must be integers."
        sys.exit()

  lx = int(sys.argv[1])
  ly = int(sys.argv[2])
  numIters = int(sys.argv[3])

  main(lx, ly, numIters)