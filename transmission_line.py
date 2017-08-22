# -*- coding: utf-8 -*-
"""

@author: http://helloworld922.blogspot.com/2013/04/transmission-line-simulation.html

Updated Erwan Pannier 30/11/15:
- physical line parameters (length / permittivity)
- animation of voltage
- oscilloscope view of one single point
- jit to improve performances

"""

#import matplotlib
#matplotlib.use("TkAgg")

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
#from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure

import numpy as np
from numpy import zeros, ceil, sqrt
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from numba import jit

DEBUG = False
plotI = False

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

    def __init__(self,Z):
        self.Z = Z

class Oscilloscope():

    def __init__(self, line, xscope, tmax):

        self.line = line
        self.line = line
        self.xscope = xscope
        self.tmax = tmax

        elems = self.line.elems
        self.iscope = int(ceil(xscope*elems))

        self._compute()

    def _compute(self):
        try:
            assert(self.line.solved)
        except AssertionError:
            raise AssertionError('Solve transmission line before adding a scope')

        self.t = self.line.t
        self.V = self.line.Varray[:,self.iscope]
        self.I = self.line.Iarray[:,self.iscope]

    def plot(self,nfig=None):
        ''' Plot scope trace on whole range '''

        fig = plt.figure(num=nfig)
        fig.clear()
        axV = fig.gca()
        axV.plot(self.t*1e9,self.V)
        axV.set_xlabel('Time (ns)')
        axV.set_ylabel('Voltage (V)')
        axV.set_title('Oscilloscope at x={0:.1f}m'.format(self.xscope*self.line.d))
#        axV.set_ylim((-2,2))
#        axV.set_xlim((0,self.tmax*1e9))
        axV.grid(True)

        if plotI:
            axI = axV.twinx()
            axI.plot(self.t*1e9,self.I, 'r')
            axI.set_ylabel('Current (A)')
            axI.yaxis.label.set_color('r')
            axI.tick_params(axis='y', colors='r')
#            axI.set_xlim((0,self.tmax*1e9))
        axV.set_xlim((0,self.tmax*1e9))
    #        axI.set_ylim((-1,1))

        fig.tight_layout()
        fig.show()

    def get_V(self,t):
        return self.V(self.t==t)

    def get_I(self,t):
        return self.I(self.t==t)


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

    def __init__(self,Z,d,Inp,Out,eps=2.25,
                 xscope=1,elems=256,interval=20,dt=1e-11):

        # Model parameters
        v = 2.998e8/sqrt(eps)   # signal speed (m/s)
        tDelay = d/v        # Propagation time (ns)
        C = tDelay / Z / elems
        L = Z**2 * C
        self.dt = dt
        self.d = d
        self.Z = Z

        self.solved = False

        # Scope
        self.scope = None

        self.elems = elems
        self.interval = interval

        # Input / output objects
        self.Inp = Inp
        self.Out = Out
        R1 = Inp.Z
        R2 = Out.Z

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

        # storage dictionary for voltage / intensity
        self.V = {}
        self.I = {}

        # %% Start
        self.x = x = np.linspace(0,d,elems+1)  # line
        self.xindex = np.arange(len(x))

        # Adjust interval to change the animation speed
#        params = [G1, G2, GC, GL,tRise,tFall,tPeriod,tOn,eps,Von,Voff,xscope,
#                 dt,elems,diags,x]
        params = [G1, G2, GC, GL, elems, diags,x]
        self.params = params


        # %% Define code variables
        self.t = None
        self.Vscope = None
        self.Iscope = None
        self.lscopeV = None
        self.lscopeI = None
        self.tscope = None


    def solve(self,tmax=20e-9):
        ''' do all calculations '''

        dt = self.dt
        tarr = np.arange(0,tmax,dt)
        self.t = tarr

        self.V[0] = V = zeros(self.elems + 1)
        self.I[0] = I = zeros(2 * self.elems)

        Varray = []
        Iarray = []
        for t in tarr:
            # source voltage
            Vs = self.Inp.V(t)

            # line voltage
            V, I = self.simStep(V, I, Vs, dt)

            self.V[t], self.I[t] = V, I         # TODO: turn V[t] into a function that reads Varray [and maybe interpolate]
            Varray.append(V)
            Iarray.append(I)

        self.Varray = np.array(Varray)
        self.Iarray = np.array(Iarray)

        self.solved = True

#    @jit
    def simStep(self, V, I, Vs, dt):
        '''
        Simulates one time step starting from V, I with timestep dt

        Note: performance test:
        - without @jit: 100 loops, best of 3: 1.97 ms per loop
        - with @jit: 10000 loops, best of 3: 16.1 µs per loop
        '''

        (G1, G2, GC, GL, elems, diags,x) = self.params

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
        Inew = zeros(2 * elems)
        for i in range(0, elems):
            # trapz
            # inductor
            Inew[2*i] = GL * (Vnew[i] - Vnew[i+1] + V[i] - V[i+1]) + I[2*i]
#            I[2*i] = GL * (Vnew[i] - Vnew[i+1] + V[i] - V[i+1]) + I[2*i]
            # capacitor
            Inew[2*i + 1] = GC * (Vnew[i + 1] - V[i + 1]) - I[2*i + 1]
#            I[2*i+1] = GC * (Vnew[i+1] - V[i+1]) - I[2*i+1]

    #         # backwards euler
    #         # inductor
    #         Inew[2 * i] = GL * (Vnew[i] - Vnew[i+1]) + I[2*i]
    #         # capacitor
    #         Inew[2 * i + 1] = GC * (Vnew[i + 1] + V[i + 1])
        return Vnew, Inew

    def get_V(self,t,xindex):
        ''' Assumes already calculated '''

        return self.V[t][xindex]

    def get_I(self,t,xindex):
        ''' Assumes already calculated '''

        return self.I[t][xindex]

    def add_scope(self,xscope,plot_tmax):
        self.scope = Oscilloscope(self,xscope,plot_tmax)
        return self.scope

    def has_scope(self):
        return self.scope is not None

    def init_movie(self, nfig=None):

        # Plot voltage

        x = self.x

        self.fig = plt.figure(num=nfig)
        self.fig.clear()
        nscreens = 1
        if self.has_scope():
            nscreens = 2
        gs = gridspec.GridSpec(nscreens, 1)
        axV = plt.subplot(gs[0])
        axV.set_xlim((0,self.d))
        axV.set_ylim((-2,2))
        axV.set_ylabel('Voltage (V)')
        axV.set_xlabel('Line (m)')
        axV.set_title('Line ${0:.0f}\Omega$, Pulser ${1:.0f}\Omega$, Pulse ${2:.0f}ns$ with ${3:.0f}ns$ rise time'.format(
                                self.Z,self.Inp.Z,self.Inp.tOn*1e9,self.Inp.tRise*1e9))
        axV.plot(x,np.zeros_like(x),':k')
        if self.has_scope():
            axV.plot([self.scope.xscope*self.d]*2,axV.get_ylim(),'-r',linewidth=2,alpha=0.3)
        else:
            self.tscope = None
            self.lscopeV = None
            if plotI:
                self.lscopeI = None
        self.lineV, = axV.plot([], [], lw=2)
        self.ttl = axV.text(.05, 0.95, '', transform = axV.transAxes, va='center')

        # Plot current

        if plotI:
            axI = axV.twinx()
            axI.set_xlim((0,self.d))
            axI.set_ylabel('Current (A)')
            axI.yaxis.label.set_color('r')
            axI.tick_params(axis='y', colors='r')
            self.lineI, = axI.plot([], [], color='r', lw=2)

        # %% Oscilloscope
        if self.scope is not None:
            ax2V = plt.subplot(gs[1])
            ax2V.set_xlabel('Time (ns)')
            ax2V.set_ylabel('Voltage (V)')
            ax2V.set_title('Oscilloscope at x={0:.1f}m'.format(self.scope.xscope*self.d))
            ax2V.set_xlim((-self.scope.tmax*1e9,0))
            ax2V.set_ylim((-2,2))
            ax2V.grid(True)
            self.lscopeV, = ax2V.plot([], [], lw=2)

            if plotI:
                ax2I = ax2V.twinx()
                ax2I.set_ylabel('Current (A)')
                ax2I.yaxis.label.set_color('r')
                ax2I.tick_params(axis='y', colors='r')
                ax2I.set_xlim((-self.scope.tmax*1e9,0))
                ax2I.set_ylim((-0.1,0.1))
                self.lscopeI, = ax2I.plot([], [], c='r', lw=2)

            plt.tight_layout()

    def start_movie(self,nframes=1000, blit=False):
        ''' note: animation has to create a self.anim ?'''

        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
               frames=nframes, interval=self.interval, blit=blit)
               # repeat=True, repeat_delay=500)

    def init(self):

        self.ttl.set_text('')
        self.lineV.set_data([], [])
        if plotI:
            self.lineI.set_data([], [])
        if self.scope is not None:
            self.lscopeV.set_data([], [])
            self.tscope, self.Vscope = ([], [])
            if plotI:
                self.lscopeI.set_data([], [])
                self.tscope, self.Iscope = ([], [])

        if plotI:
            return self.lineV, self.lineI, self.lscopeV, self.lscopeI, self.ttl
        else:
            return self.lineV, self.lscopeV, self.ttl


    # animation function.  This is called sequentially
    def animate(self,i):
    #    global G1, G2, GC, GL, dt, step, elems, Voff, Von, tRisetFall, tPeriod
    #    global tOn, diags, V, I,x, ax, tscope, Vscope, iscope, lscope

        t = self.t[i]

        self.lineV.set_data(self.x,self.get_V(t,self.xindex))
        if plotI:
            self.lineI.set_data(self.x,self.get_I(t,self.xindex))
        self.ttl.set_text('{0:.1f}ns'.format(t*1e9))

        # draw scope view (against time)
        if self.has_scope():
            self.tscope.append(-t*1e9)
            self.Vscope = [self.V[t][self.scope.iscope]] + self.Vscope
            self.lscopeV.set_data(self.tscope,self.Vscope)
            if plotI:
                self.Iscope = [self.I[t][self.scope.iscope]] + self.Iscope
                self.lscopeI.set_data(self.tscope,self.Iscope)

        # DEBUG MODE: should print t here
        # Else an error happens in animate, but is not reported because animate
        # doesn't seem to stream back its errors
        if DEBUG:
            print(t)

        if plotI:
            return self.lineV, self.lineI, self.lscopeV, self.lscopeI, self.ttl
        else:
            return self.lineV, self.lscopeV, self.ttl


if __name__ == '__main__':


    dt=3e-10
    tmax = 100e-9


#    plt.close('all')

    # Scope
    xscope = 0.95   # % of total cable length where to place the scope

    fid = Pulser(Z=75,        # pulser impedance
                 Von=1*75/75,   # Zin/Z
                 Voff=0,
                 tRise=3e-9,   
                 tOn=8e-9,   
                 tFall=4e-9,   
                 tPeriod=100e-6, #1000e-9
                 )
    load = Load(Z=75,        # Load impedance
                )
    tl=Line(Z=75,               # equivalent line impedance (Ohm)
            d=3,                # line length (m)
            Inp=fid,
            Out=load,
            eps=2.25,interval=1,elems=250,
            dt=dt,          # time resolution. lower = better but more time consuming
            )

    tl.solve(tmax)


    scope = tl.add_scope(xscope,tmax)

    scope.plot(nfig=1)

    nframes = int(tmax//dt)
    tl.init_movie(nfig=2)
    tl.start_movie(nframes=nframes, blit=False)

    tl.fig.show()
