from fenics import *
import random

T = 3000            # Final time
num_steps = 1000    # Number of time steps
dt = T / num_steps  # Time step size

# Create mesh and define function space
nx = ny = 100
mesh = RectangleMesh(Point(0, 0), Point(0.02, 0.02), nx, ny)

# Define function space for concentrations
P2 = FiniteElement('CG', triangle, 2)
element = MixedElement([P2, P2])
V = FunctionSpace(mesh, element)

# Define the coefficients
a = Constant(1.89 * pow(10,-2))
b = Constant(-3.98 * pow(10,-3))
c = Constant(2.25 * pow(10,-4))
Do = Constant(2*pow(10,-9))
kc = Constant(7.3*pow(10,-10))
f = Constant(0.65)
Oam = Constant(0.21) # The ambient oxygen (0.01-0.21)

# Define initial condition for the system
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        W_unif = 5*pow(10,7) # Initial worm density at the uniform steady state
        # Perturb the uniform steady state:
        values[0] = W_unif + pow(10,7)*(0.5 - random.random()) 
        # Equilibrium for oxygen:
        values[1] = Oam - kc*values[0]/f 
    def value_shape(self):
        return (2,)

# Class for interfacing with the solver
class KS_Nematodes(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

# Define test functions
v_1, v_2 = TestFunctions(V)

# Define functions for concentrations
du = TrialFunction(V)   # For getting the Jacobian
u = Function(V)         # The unknown functions
u_n = Function(V)       # Values of the function at previous time step

# Define initial value
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u_n.interpolate(u_init)

# Split functions to access components
dW, dO = split(du)
u1, u2 = split(u)
u_n1, u_n2 = split(u_n)

# Define functions in the variational problem
def Dw(u1,u2):
    return (a * u2**2 + b*u2 + c)**2 

def beta(u1,u2):
    return (a*u2**2 + b*u2 + c)*(2*a*u2 + b)


# Define variational problem
L = (u1 - u_n1) / dt * v_1 * dx \
+ Dw(u1,u2) * dot(grad(u1), grad(v_1)) * dx \
+ beta(u1,u2) * u1 * dot(grad(u2), grad(v_1)) * dx \
+ (u2 - u_n2) / dt * v_2 * dx \
+ Do * dot(grad(u2), grad(v_2)) * dx \
- f * (Oam - u2) * v_2 * dx + kc * u1 * v_2 * dx

a = derivative(L, u, du)
problem = KS_Nematodes(a, L)


# Create VTK files for visualization output
vtkfile_u1 = File('u1/u1.pvd')
vtkfile_u2 = File('u2/u2.pvd')

solver = PETScSNESSolver()
solver.parameters['line_search'] = 'basic'
solver.parameters['linear_solver']= 'lu'

# Time stepping
t = 0
for n in range(num_steps):
    
    # Update current time
    t += dt
    # Solve variational problem for time step
    solver.solve(problem, u.vector())
    
    # Save solution to file 
    _u1, _u2 = u.split()
    vtkfile_u1 << (_u1, t)
    vtkfile_u2 << (_u2, t)
    
    # Update previous solution
    u_n.assign(u)
    print('t=',t)