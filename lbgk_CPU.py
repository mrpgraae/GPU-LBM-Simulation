import numpy as num
import copy, math
import sys

#  Velocity scheme (D2Q9):
#
#       6   2   5
#        \  |  /
#         \ | /
#          \|/
#   3-------0-------1
#          /|\
#         / | \
#        /  |  \
#       7   4   8
#

q = 9

# Velocity weights
w = num.array(num.ones(9), dtype=float)
w[0] *= 4./9.
w[1:5] *= 1./9.
w[5:9] *= 1./36.

# Node velocities
vel = num.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
velCol0 = num.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=float)
velCol1 = num.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=float)

# Array for mapping bounceback velocities.
bounceBackVel = num.array([0,3,4,1,2,7,8,5,6], dtype=int)

# For initializing to motionless fluid.

noMotion = num.array([1.0,0.,0.,0.,0.,0.,0.,0.,0.], dtype=float)

# Functions for computing values

def calcEquilibrium(velDir, rho, ux, uy, uSqr):
  velDot = vel[velDir, 0]*ux + vel[velDir, 1]*uy
  return rho*w[velDir]*(1. + 3.*velDot + 4.5*velDot*velDot - 1.5*uSqr)

def calcEquilibriumVector(rho, ux, uy, uSqr):
  velDot = velCol0*ux + velCol1*uy
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

# A class for the simulation.
class simulation():

  def __init__(self, lx, ly):
    self.lx = lx
    self.ly = ly
    self.lyAll = ly+2
    self.lyB = ly+1
    self.obst_x = lx/6
    self.obst_y = ly/2
    self.obst_r = ly/9
    # Fluid parameters
    self.Re = 220
    self.V = 0.066
    self.umax = (3./2.)*self.V
    self.nu     = (self.V*self.obst_r)/self.Re                    # Velocity in lattice units.
    self.omega = 1.0 / (3.*self.nu+0.5);
    omega = self.omega
    print "Using omega = " + str(self.omega)
    # Array for setting inflow densities
    self.inflow = num.empty((self.lyAll,q),dtype=float)
    # Keep a copy of the lattice for "ping-pong buffering".
    self.lattice = num.empty([lx, ly + 2], dtype=object)          
    self.latticeCopy = num.empty([lx, ly + 2], dtype=object)

  def calcIniVelocity(self, y):
    y = float(y)
    l = float(self.ly-1)
    vel = 4.*self.umax / (l*l) * (l*y-y*y)
    return vel

  def initializeLattice(self):
    # Set nodes within cylinder to solid.
    for i in xrange(self.lx):
      for j in xrange(self.lyAll):
        self.lattice[i,j] = node(self.omega)
        if (i - self.obst_x)*(i - self.obst_x) + (j - self.obst_y)*(j - self.obst_y) <= self.obst_r*self.obst_r:
          self.lattice[i,j].setSolid()

    # Set nodes on north and south boundaries to solid.
    for i in xrange(self.lx):
      self.lattice[i,1].setSolid()
      self.lattice[i,self.ly].setSolid()

    # Set node velocities
    for i in xrange(self.lx):
      for j in xrange(1,self.lyB):
        vel = self.calcIniVelocity(j)
        uSqr = vel * vel
        self.lattice[i, j].setVelocityVector(calcEquilibriumVector(1., vel, 0., uSqr))  

    for i in xrange(self.lx):
      for j in xrange(2,self.ly):
        if (i - self.obst_x)*(i - self.obst_x) + (j - self.obst_y)*(j - self.obst_y) > self.obst_r*self.obst_r:             # DEBUGGING!
          Sum = num.sum(self.lattice[i,j].densities)

    for i in xrange(self.lyAll):
      for c in xrange(q):
        self.inflow[i,c] = self.lattice[0,i].densities[c]

    for i in xrange(self.lx):
      for j in xrange(self.lyAll):
        self.latticeCopy[i,j] = copy.deepcopy(self.lattice[i,j])

  def swapLattice(self):
    tmpLattice = self.latticeCopy
    self.lattice = self.latticeCopy
    self.latticeCopy = tmpLattice

  # This function treats the special cases of the first and last column of nodes.
  def periodicStream(self):
    for i in xrange(1,self.lyB):
      self.latticeCopy[self.lx-1,i].densities[6] = self.lattice[0,i-1].densities[6]
      self.latticeCopy[self.lx-1,i].densities[3] = self.lattice[0,i].densities[3]
      self.latticeCopy[self.lx-1,i].densities[7] = self.lattice[0,i+1].densities[7]

      self.latticeCopy[0,i].densities[5] = self.lattice[self.lx-1,i-1].densities[5]
      self.latticeCopy[0,i].densities[1] = self.lattice[self.lx-1,i].densities[1]
      self.latticeCopy[0,i].densities[8] = self.lattice[self.lx-1,i+1].densities[8]

      self.latticeCopy[self.lx-1,i].densities[2] = self.lattice[self.lx-1,i-1].densities[2]
      self.latticeCopy[self.lx-1,i].densities[4] = self.lattice[self.lx-1,i+1].densities[4]

      self.latticeCopy[0,i].densities[2] = self.lattice[0,i-1].densities[2]
      self.latticeCopy[0,i].densities[4] = self.lattice[0,i+1].densities[4]

      self.latticeCopy[self.lx-2,i].densities[6] = self.lattice[self.lx-1,i-1].densities[6]
      self.latticeCopy[self.lx-2,i].densities[3] = self.lattice[self.lx-1,i].densities[3]
      self.latticeCopy[self.lx-2,i].densities[7] = self.lattice[self.lx-1,i+1].densities[7]

      self.latticeCopy[1,i].densities[5] = self.lattice[0,i-1].densities[5]
      self.latticeCopy[1,i].densities[1] = self.lattice[0,i].densities[1]
      self.latticeCopy[1,i].densities[8] = self.lattice[0,i+1].densities[8]

  def flowStream(self):
    for i in xrange(1,self.lyB):
      self.latticeCopy[0,i].densities[5] = self.inflow[i,5]
      self.latticeCopy[0,i].densities[1] = self.inflow[i,1]
      self.latticeCopy[0,i].densities[8] = self.inflow[i,8]

      self.latticeCopy[0,i].densities[2] = self.lattice[0,i-1].densities[2]
      self.latticeCopy[0,i].densities[4] = self.lattice[0,i+1].densities[4]

      self.latticeCopy[1,i].densities[5] = self.lattice[0,i-1].densities[5]
      self.latticeCopy[1,i].densities[1] = self.lattice[0,i].densities[1]
      self.latticeCopy[1,i].densities[8] = self.lattice[0,i+1].densities[8]

      self.latticeCopy[self.lx-1,i].densities[6] = self.inflow[i,6]
      self.latticeCopy[self.lx-1,i].densities[3] = self.inflow[i,3]
      self.latticeCopy[self.lx-1,i].densities[7] = self.inflow[i,7]

      self.latticeCopy[self.lx-1,i].densities[2] = self.lattice[self.lx-1,i-1].densities[2]
      self.latticeCopy[self.lx-1,i].densities[4] = self.lattice[self.lx-1,i+1].densities[4]

      self.latticeCopy[self.lx-2,i].densities[6] = self.lattice[self.lx-1,i-1].densities[6]
      self.latticeCopy[self.lx-2,i].densities[3] = self.lattice[self.lx-1,i].densities[3]
      self.latticeCopy[self.lx-2,i].densities[7] = self.lattice[self.lx-1,i+1].densities[7]


  def stream(self, boundaryCond = 'periodic'):
    for i in xrange(1,self.lx-1):
      for j in xrange(1,self.lyB):
        for n in xrange(q):
          self.latticeCopy[i+vel[n,0],j+vel[n,1]].densities[n] = self.lattice[i,j].densities[n]
    if boundaryCond == 'periodic':
      self.periodicStream()
    if boundaryCond == 'flow':
      self.flowStream()
    self.lattice, self.latticeCopy = self.latticeCopy, self.lattice

  def collide(self):
    for i in xrange(self.lx):
      for j in xrange(1,self.lyB):
        self.lattice[i,j].collide()

# Node class definition.
# Contains all data local to a node, as well as all methods that manipulate this data.
class node():
  def __init__(self, omega, solid=False):
    self.densities = num.array(num.zeros(9))
    self.solid = solid
    self.omega = omega

  def setSolid(self):
    self.solid = True

  def isSolid(self):
    return self.solid

  def setVelocity(self, direction, value):
    self.densities[direction] = value

  def setVelocityVector(self, value):
    self.densities = value

  def collide(self):
    if self.isSolid():
      self.bounceBack()
    else:
      self.BGKCollide()

  def bounceBack(self):
    oldDensities = num.copy(self.densities)
    for i in xrange (q):
      self.densities[i] = oldDensities[bounceBackVel[i]]

  # Uses BGK to compute collision term.
  def BGKCollide(self):
    rho = calcRho(self)
    assert rho > 0.
    vel = calcVelocity(self, rho)
    uSqr = vel[0] * vel[0] + vel[1] * vel[1]
    if math.isnan(uSqr): raise Exception("uSqr is NaN.")
    self.densities *= (1 - self.omega)
    self.densities += self.omega * calcEquilibriumVector(rho, vel[0], vel[1], uSqr)

# Main simulation loop

def main(lxIn, lyIn, numIters):

  # Simulation dimensions
  lx = lxIn
  ly = lyIn
  q = 9

  sim = simulation(lx, ly)

  print "Setting up the lattice."
  sim.initializeLattice()
  print "Lattice setup complete."

  i = 0

  while True:

    sim.stream('flow')

    sim.collide()

    i += 1

    print i, " iterations complete."

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