# -*- coding: utf-8 -*-
"""

@author: http://helloworld922.blogspot.com/2013/04/transmission-line-simulation.html

Updated Erwan Pannier 30/11/15:
- physical line parameters (length / permittivity)
- animation of voltage
- oscilloscope view of one single point
- jit to improve performances

"""

import matplotlib
#matplotlib.use("TkAgg")

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
#from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure

import numpy as np
from numpy import zeros, ceil, sqrt, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from numba import jit


#    @jit
class Pulser():
    '''
    Pulser: force voltage
    
    Input
    -----------
    
    - Z                   # pulser impedance
    - Voff
    - Von
    - tRise
    - tFall
    - tPeriod             # between two pulses (s)
    - tOn                 # full amplitude (s)
    
    '''
    
    def __init__(self,Z,Von, Voff, tRise, tOn, tFall, tPeriod):
        
        self.Z = Z
        self.Voff = Voff
        self.Von = Von
        self.tRise = tRise
        self.tFall = tFall
        self.tPeriod = tPeriod
        self.tOn = tOn
        
    def V(self,t):
        '''
        Calculates the output voltage of the pulse source at instant t
        '''
        Voff, Von, tRise, tFall, tPeriod, tOn = (
                self.Voff, self.Von, self.tRise, self.tFall, self.tPeriod, self.tOn)
                
        t = t % tPeriod
        Vs = self.Voff
        if (t < tRise):
            Vs = (Von - Voff) * t / tRise + Voff
        elif (t < tRise + tOn):
            Vs = Von
        elif (t < tRise + tOn + tFall):
            Vs = (Voff - Von) * (t-tRise-tOn) / tFall + Von
        return Vs
 
class Load():
    ''' Generic class. Use one of the subclasses'''
    
    def __init__(self,Z):
        self.Z = None

class ResistiveLoad(Load):
    
    def __init__(self,Z):
        self.Z = lambda t: Z

class PlasmaLoad(Load):
    
    def __init__(self,Z,D=1e-3,l=6e-3):
        ''' Input
        
        D: plasma diameter
        l: gap distance 
        '''
        self.Z = lambda t: Z
        self.D = D
        self.l = l

    def sigma(self,t):
        ''' Electrical conductivity'''
        e = 1.60217657e-19

        return e**2*ne(t)/

    def ne(self):
        ''' Electron density'''        
    
    def Rp(self,t):
        2*pi*R


class Line():
 
    '''
    A model for a transmission line. Original code from:
    http://helloworld922.blogspot.com/2013/04/transmission-line-simulation.html
    
    Model parameters
    - elems = 256         # number of elements to model the line
    - Z = 50              # equivalent line impedance (Ohm)
    - d = 2               # line length (m)
    - eps = 2.25          # dielectric permittivity (2.25 typical for polyethylene)
     
    - inp:                # Input (ex: pulser)
    - out:                # Output (ex: load)
    
    - dt                  # time resolution. lower = better but more time consuming
     
    # Scope
    - xscope              # % of total cable length where to place the scope [0-1]
    
    imulation:        
    - maxSteps
    - interval            # time before refreshing screen
    
    '''    
       
    def __init__(self,Z,Inp,Out,d,eps=2.25,
                 xscope=1,elems=256,interval=20):
        
        # Model parameters    
        v = 2.998e8/sqrt(eps)   # signal speed (m/s)
        tDelay = d/v        # Propagation time (ns)
        C = tDelay / Z / elems
        L = Z**2 * C
        
        # Scope    
        self.iscope = int(ceil(xscope*elems))
        
        
        self.elems = elems
        self.interval = interval        
        
        # Input / output objects
        self.Inp = Inp
        self.Out = Out
        R1 = Inp.Z
        R2 = Out.Z        # function of time
        
        # %% build simulation
        G1 = 1 / R1
        G2 = lambda t: 1 / R2(t)
        # trapz
        GC = 2*C / dt        # capacitor conductance in Norton equivalent circuit
        GL = dt / (2*L)      # inductor conductance in Norton equivalent circuit
        #    # backward euler
        #    GC = C / dt
        #    GL = dt / L
         
        # diagonal divisors
        diags = zeros(elems)
        diags[0] = G1 + GL
        for i in range(1, elems):
            diags[i] = 2 * GL + GC - GL * GL / diags[i - 1]
        diags_out = lambda t: GC + GL + G2(t) - GL * GL / diags[elems - 1]
        
        # storage dictionary for voltage / intensity
        self.V = {}
        self.I = {}
        
        # %% Start
        self.x = x = np.linspace(0,d,elems+1)  # line
        self.xindex = np.arange(len(x))
   
        # Adjust interval to change the animation speed
#        params = [G1, G2, GC, GL,tRise,tFall,tPeriod,tOn,eps,Von,Voff,xscope,
#                 dt,elems,diags,x]
        params = [G1, G2, GC, GL, elems, diags,diags_out,x]
        self.params = params
        
    def solve(self,tmax=20e-9,dt=1e-11):
        ''' do all calculations '''
        
        tarr = np.arange(0,tmax,dt)
        self.t = tarr
        
        self.V[0] = V = zeros(self.elems + 1)
        self.I[0] = I = zeros(2 * self.elems)
        
        for t in tarr:
            # source voltage
            Vs = self.Inp.V(t)
            
            # line voltage
            V, I = self.simStep(V, I, Vs, t, dt)
            
            self.V[t], self.I[t] = V, I
            
    def get_V(self,t,xindex):
        ''' Assumes already calculated '''
        
        return self.V[t][xindex]
    
    def plot(self):
        
        # Plot everything

        x = self.x
        
        plt.close()
        self.fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1],hspace=0.5)
        ax = plt.subplot(gs[0])
        ax.set_xlim((0,d))
        ax.set_ylim((-2,2))
        ax.set_ylabel('Voltage (V)')
        ax.set_xlabel('Line (m)')
        ax.set_title('Line ${0:.0f}\Omega$, Load ${1:.0f}\Omega$, Pulse ${2:.0f}ns$ with ${3:.0f}ns$ rise time'.format(
                                Z,self.Inp.Z,self.Inp.tOn*1e9,self.Inp.tRise*1e9))
        ax.plot(x,np.zeros_like(x),':k')
        ax.plot([xscope*d]*2,ax.get_ylim(),'-r',linewidth=2,alpha=0.3)
        self.line, = ax.plot([], [], lw=2)
        self.ttl = ax.text(.05, 0.95, '', transform = ax.transAxes, va='center')
        
        # %% Oscilloscope
        ax2 = plt.subplot(gs[1])    
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_title('Oscilloscope at x={0:.1f}m'.format(xscope*d))
        ax2.set_xlim((-100,0))
        ax2.set_ylim((-2,2))
        ax2.grid(True)
        self.lscope, = ax2.plot([], [], lw=2)
        
    def start(self,nframes=1000):
        ''' note: animation has to create a self.anim ?'''

        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
               frames=nframes, interval=self.interval)
               #, blit=True,repeat=True,               repeat_delay=500)
#                           
#        self.anim = animation.FuncAnimation(self.fig, self.animate,
#            frames=nframes, interval=50, blit=True,repeat=True,
#            repeat_delay=500)
            
#    @jit
    def simStep(self, V, I, Vs, t, dt):
        '''
        Simulates one time step starting from V, I with timestep dt 
        
        Note: performance test:
        - without @jit: 100 loops, best of 3: 1.97 ms per loop
        - with @jit: 10000 loops, best of 3: 16.1 µs per loop
        '''

        (G1, G2, GC, GL, elems, diags,diags_out,x) = self.params
                
        # calculate norton currents for caps/inductors
        INort = zeros(2 * elems)
        for i in range(0, elems):
            # trapz
            # inductor
            INort[2*i] = -((V[i] - V[i+1]) * GL + I[2*i])
            # capacitor
            INort[2*i+1] = V[i+1] * GC + I[2*i+1]
             
    #        # backwards euler
    #        # inductor
    #        INort[2 * i] = -I[2 * i]
    #        # capacitor
    #        INort[2 * i + 1] = V[i+1] * GC
         
        # build B vector
        Bvec = zeros(elems + 1)
        Bvec[0] = INort[0] + G1 * Vs
        for i in range(1, elems):
            Bvec[i] = INort[2*i-1] + INort[2*i] - INort[2*i-2] + GL * Bvec[i-1] / diags[i-1]
        Bvec[elems] = INort[2*elems-1] - INort[2*elems-2] + GL * Bvec[elems-1] / diags[elems-1]
         
        # back-sub for voltages
        Vnew = zeros(elems + 1)
        Vnew[elems] = Bvec[elems] / diags_out(t)
        for i in range(elems - 1, -1, -1):
            Vnew[i] = (Bvec[i] + GL * Vnew[i+1]) / diags[i]
         
        # calculate currents through inductors/caps
        #Inew = zeros(2 * elems)
        for i in range(0, elems):
            # trapz
            # inductor
            #Inew[2*i] = GL * (Vnew[i] - Vnew[i+1] + V[i] - V[i+1]) + I[2*i]
            I[2*i] = GL * (Vnew[i] - Vnew[i+1] + V[i] - V[i+1]) + I[2*i]
            # capacitor
            #Inew[2*i + 1] = GC * (Vnew[i + 1] - V[i + 1]) - I[2*i + 1]
            I[2*i+1] = GC * (Vnew[i+1] - V[i+1]) - I[2*i+1]
             
    #         # backwards euler
    #         # inductor
    #         Inew[2 * i] = GL * (Vnew[i] - Vnew[i+1]) + I[2*i]
    #         # capacitor
    #         Inew[2 * i + 1] = GC * (Vnew[i + 1] + V[i + 1])
        return Vnew, I
 

    def init(self):    

        self.ttl.set_text('')
        self.line.set_data([], [])
        self.lscope.set_data([], [])
        self.tscope, self.Vscope = ([], [])
        return self.line, self.lscope, self.ttl
                    
    # animation function.  This is called sequentially
    def animate(self,i):
    #    global G1, G2, GC, GL, dt, step, elems, Voff, Von, tRisetFall, tPeriod
    #    global tOn, diags, V, I,x, ax, tscope, Vscope, iscope, lscope
    
        t = self.t[i]
        
        self.line.set_data(self.x,self.get_V(t,self.xindex))    
        self.ttl.set_text('{0:.1f}ns'.format(t*1e9))
        
        # scope
        self.tscope.append(-t*1e9)
        self.Vscope = [self.V[t][self.iscope]] + self.Vscope
        self.lscope.set_data(self.tscope,self.Vscope)
        
        
        return self.line, self.lscope, self.ttl
    

if __name__ == '__main__':
    
    #%% Model parameters
    Z = 50              # equivalent line impedance (Ohm)
    d = 2               # line length (m)
     
    Zin = 300             # pulser impedance
    Zout = 1e-9            # Load impedance
    tmax = 100e-9
    dt = 1e-10          # time resolution. lower = better but more time consuming
    tRise = 0.1e-9 #1e-9
    tFall = tRise
    tPeriod = 5000e-9 #1000e-9
    tOn = 5e-9 # tPeriod / 2 - tRise
    Von = 4
    Voff = 0
     
    # Scope
    xscope = 0.59   # % of total cable length where to place the scope
    
    fid = Pulser(Zin,Von,Voff,tRise,tOn,tFall,tPeriod)
    load = PlasmaLoad(Zout)
    tl=Line(Z,fid,load,d,eps=2.25,xscope=xscope,interval=1)
    
    tl.solve(tmax,dt)
    
    tl.plot()
    nframes = int(tmax//dt)
    tl.start(nframes=nframes)
    
    plt.show() 