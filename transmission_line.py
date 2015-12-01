# -*- coding: utf-8 -*-
"""

@author: http://helloworld922.blogspot.com/2013/04/transmission-line-simulation.html

Updated Erwan Pannier 30/11/15:
- physical line parameters (length / permittivity)
- animation of voltage
- oscilloscope view of one single point
- jit to improve performances

"""

import numpy as np
from numpy import zeros, ceil, sqrt
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from numba import jit

class line():
 
    '''
    A model for a transmission line. Original code from:
    http://helloworld922.blogspot.com/2013/04/transmission-line-simulation.html
    
    Model parameters
    - elems = 256         # number of elements to model the line
    - Z = 50              # equivalent line impedance (Ohm)
    - d = 2               # line length (m)
    - eps = 2.25          # dielectric permittivity (2.25 typical for polyethylene)
     
    - R1                  # pulser impedance
    - R2                  # Load impedance
    - dt                  # time resolution. lower = better but more time consuming
    - Voff
    - Von
    - tRise
    - tFall
    - tPeriod             # between two pulses (s)
    - tOn                 # full amplitude (s)
     
    # Scope
    - xscope              # % of total cable length where to place the scope [0-1]
    
    imulation:        
    - maxSteps
    - interval            # time before refreshing screen
    
    '''    
       
    def __init__(self,Z,R1,R2,d,tRise,tFall,tPeriod,tOn,eps=2.25,Von=1,Voff=0,
                 xscope=1,dt=1e-11,elems=256,maxSteps=1000,interval=20):
        
        # Model parameters    
        v = 2.998e8/sqrt(eps)   # signal speed (m/s)
        tDelay = d/v        # Propagation time (ns)
        C = tDelay / Z / elems
        L = Z**2 * C
        
        # Scope    
        self.iscope = int(ceil(xscope*elems))
        self.tscope, self.Vscope = ([], [])
        
        # %% build simulation
        G1 = 1 / R1
        G2 = 1 / R2
        # trapz
        GC = 2*C / dt        # capacitor conductance in Norton equivalent circuit
        GL = dt / (2*L)      # inductor conductance in Norton equivalent circuit
        #    # backward euler
        #    GC = C / dt
        #    GL = dt / L
         
        # diagonal divisors
        diags = zeros(elems + 1)
        diags[0] = G1 + GL
        for i in range(1, elems):
            diags[i] = 2 * GL + GC - GL * GL / diags[i - 1]
        diags[elems] = GC + GL + G2 - GL * GL / diags[elems - 1]
         
        self.V = zeros(elems + 1)
        self.I = zeros(2 * elems)
        
        # Plot everything
        x = np.linspace(0,d,elems+1)  # line
        plt.close()
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1],hspace=0.5)
        ax = plt.subplot(gs[0])
        ax.set_xlim((0,d))
        ax.set_ylim((-2,2))
        ax.set_ylabel('Voltage (V)')
        ax.set_xlabel('Line (m)')
        ax.set_title('Line ${0:.0f}\Omega$, Load ${1:.0f}\Omega$, Pulse ${2:.0f}ns$ with ${3:.0f}ns$ rise time'.format(
                                Z,R1,tOn*1e9,tRise*1e9))
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
        
        # %% Start
        self.step = 0
        
        # Adjust interval to change the animation speed
        args = [G1, G2, GC, GL,tRise,tFall,tPeriod,tOn,eps,Von,Voff,xscope,
                 dt,elems,diags,x]
        self.args = args
        animation.FuncAnimation(fig, self.animate, fargs=[args,],init_func=self.init,
               frames=maxSteps, interval=interval, blit=True)
        
        plt.show()                    
            
    @jit
    def pulse(self,t, Voff, Von, tRise, tFall, tPeriod, tOn):
        '''
        Calculates the output voltage of a pulse source
        '''
        t = t % tPeriod
        Vs = Voff
        if (t < tRise):
            Vs = (Von - Voff) * t / tRise + Voff
        elif (t < tRise + tOn):
            Vs = Von
        elif (t < tRise + tOn + tFall):
            Vs = (Voff - Von) * (t-tRise-tOn) / tFall + Von
        return Vs
 
    @jit
    def simStep(self,G1, G2, GC, GL, dt, elems, Voff, Von, tRise, tFall, 
                tPeriod, tOn, diags, V, I):
        '''
        Simulates one time step
        
        Note: performance test:
        - without @jit: 100 loops, best of 3: 1.97 ms per loop
        - with @jit: 10000 loops, best of 3: 16.1 µs per loop
        '''
        # source voltage
        Vs = self.pulse(self.step*dt, Voff, Von, tRise, tFall, tPeriod, tOn)
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
        Vnew[elems] = Bvec[elems] / diags[elems]
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
        return self.line, self.ttl, self.lscope
                    
    # animation function.  This is called sequentially
    def animate(self,i,args):
    #    global G1, G2, GC, GL, dt, step, elems, Voff, Von, tRisetFall, tPeriod
    #    global tOn, diags, V, I,x, ax, tscope, Vscope, iscope, lscope
    
        [G1, G2, GC, GL,tRise,tFall,tPeriod,tOn,eps,Von,Voff,xscope,
                 dt,elems,diags,x] = args    
        
        self.V, self.I = self.simStep(G1, G2, GC, GL, dt, elems, Voff, Von, tRise, 
                           tFall, tPeriod, tOn, diags,self.V,self.I)
        
#        print('V',self.V)
#        print('step',self.step)
        self.line.set_data(x,self.V)    
        self.ttl.set_text('{0:.1f}ns'.format(self.step*dt*1e9))
        
        # scope
        self.tscope.append(-self.step*dt*1e9)
        self.Vscope = [self.V[self.iscope]] + self.Vscope
        self.lscope.set_data(self.tscope,self.Vscope)
        
        self.step += 1
        
        return self.line, self.ttl, self.lscope
    

if __name__ == '__main__':
    
    #%% Model parameters
    Z = 50              # equivalent line impedance (Ohm)
    d = 2               # line length (m)
     
    R1 = 50             # pulser impedance
    R2 = 1e9            # Load impedance
    dt = 5e-11          # time resolution. lower = better but more time consuming
    tRise = 0.1e-9 #1e-9
    tFall = tRise
    tPeriod = 50e-9 #1000e-9
    tOn = 5e-9 # tPeriod / 2 - tRise
     
    # Scope
    xscope = 0.9    # % of total cable length where to place the scope
    
    tl=line(Z,R1,R2,d,tRise,tFall,tPeriod,tOn,dt=dt,interval=0,xscope=xscope)