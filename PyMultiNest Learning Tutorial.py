
# coding: utf-8

# # PyMultiNest Learning Tutorial
# 
# CCT = Charge Carrier Trapping - This is a test of comparing the Zhou et al 2017 results with a data driven analysis using multinest

# In[22]:

get_ipython().magic('matplotlib inline')
from pylab import *;ion()
from pandas import read_csv


# In[23]:

import os
if not os.path.exists('chains/'):
    os.mkdir('chains')

print(os.path.exists('chains'))


# # PyMultiNest Solver Class

# **Initial Example**

# In[24]:

#!/usr/bin/env python
from pymultinest.solve import Solver
from numpy import pi, sin

class EggboxProblem(Solver):
    def Prior(self, cube):
        return cube * 10 * pi
    def LogLikelihood(self, cube):
        chi = (sin(cube)).prod()
        return (2. + chi)**5

solution = EggboxProblem(n_dims = 1)
print(solution)
solution = EggboxProblem(n_dims = 2)
print(solution)


# **My PyMultiNest Test**

# In[25]:

get_ipython().magic('matplotlib inline')
from pylab import *;ion()

from pymultinest.solve import Solver,solve
from numpy import pi, sin, cos, linspace

def straight_line(cube):
    offset = cube[0]
    slope  = cube[1]
    return lambda abscissa: offset + slope * abscissa

def sine_wave(cube):
    amp    = cube[0]
    period = cube[1]
    return lambda abscissa: amp*sin(2*pi / period * abscissa)

np.random.seed(0)

param0= 0.1#0.05
param1= 0.1#5*pi
yunc  = 0.025
nPts  = int(10)
nThPts= int(1e3)

xmin  = -0.5*pi
xmax  =  0.5*pi
dx    = 0.1*(xmax - xmin)

# model = sine_wave
model = straight_line

yuncs = np.random.normal(yunc, 1e-2 * yunc, nPts)
thdata= np.linspace(xmin-dx, xmax+dx, nThPts)

xdata = np.random.uniform(xmin, xmax, nPts)
xdata = sort(xdata)

ydata = model([param0,param1])(xdata)

yerr  = np.random.normal(0, yuncs, nPts)
zdata = ydata + yerr

figure(figsize=(20,5))
plot(thdata, model([param0, param1])(thdata))
errorbar(xdata, zdata, yuncs, fmt='o')


# In[26]:

class ChisqFit(Solver):
    def Prior(self, cube):
        return cube
    
    def LogLikelihood(self, cube):
        chisq = (-0.5*((model(cube)(xdata) - ydata)**2.) )#/ yuncs**2.
        return chisq.prod()

# solution = ChisqFit(n_dims = 2, resume=False, outputfiles_basename='./42-')

solution = ChisqFit(n_dims = 2, n_params=None, n_clustering_params=None, wrapped_params=None        ,                     importance_nested_sampling=True, multimodal=True, const_efficiency_mode=False   ,                     n_live_points=400, evidence_tolerance=0.5, sampling_efficiency=0.8              ,                     n_iter_before_update=100, null_log_evidence=-1e+90, max_modes=100               ,                     mode_tolerance=-1e+90, outputfiles_basename='chains/1-', seed=-1, verbose=False ,                     resume=False, context=0, write_output=True, log_zero=-1e+100, max_iter=0         ,                     init_MPI=False, dump_callback=None)
print(solution)


# **Simplest Example**

# In[27]:

import pymultinest

def prior(cube, ndim, nparams):
    cube[0] = cube[0] * 2

def loglikelihood(cube, ndim, nparams):
    return ((cube[0] - 0.2) / 0.1)**2

pymultinest.run(loglikelihood, prior, n_dims=1, max_iter=2)


# # PyMultiNest Solve Function

# In[28]:

#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import os

if not os.path.exists("chains"): os.mkdir("chains")

# probability function, taken from the eggbox problem.

def myprior(cube):
    return cube * 10 * pi

def myloglike(cube):
    chi = (cos(cube / 2.)).prod()
    return (2. + chi)**5


# In[29]:

# number of dimensions our problem has
parameters = ["x", "y"]
n_params = len(parameters)

# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
    n_dims=n_params, outputfiles_basename="chains/3-")

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
    print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))


# # PyMultiNest Run Function from Demo_Minimalist
from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math, os
if not os.path.exists("chains"): os.mkdir("chains")

# our probability functions
# Taken from the eggbox problem.

def myprior(cube, ndim, nparams):
    for i in range(ndim):
        cube[i] = cube[i] * 10 * math.pi

def myloglike(cube, ndim, nparams):
    chi = 1.
    for i in range(ndim):
        chi *= math.cos(cube[i] / 2.)
    return math.pow(2. + chi, 5)
#     return (2. + chi)**5# number of dimensions our problem has
parameters = ["x", "y"]
n_params = len(parameters)

# run MultiNest
pymultinest.run(myloglike, myprior, n_params, resume = True, verbose = True)

# run 
# $ multinest_marginals.py chains/1-
# which will produce pretty marginal pdf plots

# for code to analyse the results, and make plots see full demo
# # PyMultiNest Self-Generated Run Function: 'multinest_marginals.py'
#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
__doc__ = """
Script that does default visualizations (marginal plots, 1-d and 2-d).

Author: Johannes Buchner (C) 2013
"""
import numpy
from numpy import exp, log
import matplotlib.pyplot as plt
import sys, os
import json
import pymultinestargv = ['multinest_marginals.py','chains/1-']

if len(argv) != 2:
    sys.stderr.write("""SYNOPSIS: %s <output-root> 

    output-root:     Where the output of a MultiNest run has been written to. 
                    Example: chains/1-
%s""" % (argv[0], __doc__))
    sys.exit(1)prefix = argv[1]
print('model "%s"' % prefix)
parameters = json.load(open(prefix + 'params.json'))
n_params = len(parameters)a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
s = a.get_stats()

json.dump(s, open(prefix + 'stats.json', 'w'), indent=4)print('  marginal likelihood:')
print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
print('  parameters:')
for p, m in zip(parameters, s['marginals']):
    lo, hi = m['1sigma']
    med = m['median']
    sigma = (hi - lo) / 2
    i = max(0, int(-numpy.floor(numpy.log10(sigma))) + 1)
    fmt = '%%.%df' % i
    fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
    print(fmts % (p, med, sigma))print('creating marginal plot ...')
p = pymultinest.PlotMarginal(a)values = a.get_equal_weighted_posterior()
assert n_params == len(s['marginals'])
modes = s['modes']dim2 = os.environ.get('D', '1' if n_params > 20 else '2') == '2'
dim2%matplotlib inline
from pylab import *;ion()
nbins = 100 if n_params < 3 else 20
if dim2:
    print('dim2 is True and n_params is', n_params)
    plt.figure(figsize=(5*n_params, 5*n_params))
    for i in range(n_params):
        plt.subplot(n_params, n_params, i + 1)
        plt.xlabel(parameters[i])
    
        m = s['marginals'][i]
        plt.xlim(m['5sigma'])
    
        oldax = plt.gca()
        x,w,patches = oldax.hist(values[:,i], bins=nbins, edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
        oldax.set_ylim(0, x.max())
    
        newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
        p.plot_marginal(i, ls='-', color='blue', linewidth=3)
        newax.set_ylim(0, 1)
    
        ylim = newax.get_ylim()
        y = ylim[0] + 0.05*(ylim[1] - ylim[0])
        center = m['median']
        low1, high1 = m['1sigma']
        #print(center, low1, high1)
        newax.errorbar(x=center, y=y,
            xerr=numpy.transpose([[center - low1, high1 - center]]), 
            color='blue', linewidth=2, marker='s')
        oldax.set_yticks([])
        #newax.set_yticks([])
        newax.set_ylabel("Probability")
        ylim = oldax.get_ylim()
        newax.set_xlim(m['5sigma'])
        oldax.set_xlim(m['5sigma'])
        #plt.close()
    
        for j in range(i):
            plt.subplot(n_params, n_params, n_params * (j + 1) + i + 1)
            p.plot_conditional(i, j, bins=20, cmap = plt.cm.gray_r)
            for m in modes:
                plt.errorbar(x=m['mean'][i], y=m['mean'][j], xerr=m['sigma'][i], yerr=m['sigma'][j])
            plt.xlabel(parameters[i])
            plt.ylabel(parameters[j])
            #plt.savefig('cond_%s_%s.pdf' % (params[i], params[j]), bbox_tight=True)
            #plt.close()

    plt.savefig(prefix + 'marg.pdf')
    plt.savefig(prefix + 'marg.png')
    plt.close()
else:
    from matplotlib.backends.backend_pdf import PdfPages
    sys.stderr.write('1dimensional only. Set the D environment variable \n')
    sys.stderr.write('to D=2 to force 2d marginal plots.\n')
    pp = PdfPages(prefix + 'marg1d.pdf')
    
    for i in range(n_params):
        plt.figure(figsize=(3, 3))
        plt.xlabel(parameters[i])
        plt.locator_params(nbins=5)
        
        m = s['marginals'][i]
        iqr = m['q99%'] - m['q01%']
        xlim = m['q01%'] - 0.3 * iqr, m['q99%'] + 0.3 * iqr
        #xlim = m['5sigma']
        plt.xlim(xlim)
    
        oldax = plt.gca()
        x,w,patches = oldax.hist(values[:,i], bins=numpy.linspace(xlim[0], xlim[1], 20), edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
        oldax.set_ylim(0, x.max())
    
        newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
        p.plot_marginal(i, ls='-', color='blue', linewidth=3)
        newax.set_ylim(0, 1)
    
        ylim = newax.get_ylim()
        y = ylim[0] + 0.05*(ylim[1] - ylim[0])
        center = m['median']
        low1, high1 = m['1sigma']
        #print center, low1, high1
        newax.errorbar(x=center, y=y,
            xerr=numpy.transpose([[center - low1, high1 - center]]), 
            color='blue', linewidth=2, marker='s')
        oldax.set_yticks([])
        newax.set_ylabel("Probability")
        ylim = oldax.get_ylim()
        newax.set_xlim(xlim)
        oldax.set_xlim(xlim)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.close()
    pp.close()
# # PyMultiNest Demo Script -- Full

# In[39]:

from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math
import os
import threading, subprocess
from sys import platform

if not os.path.exists("chains"): os.mkdir("chains")
    


# In[58]:

get_ipython().magic('matplotlib inline')
from pylab import *;ion()

from pymultinest.solve import Solver,solve
from numpy import pi, sin, cos, linspace

def straight_line(cube):
    offset = cube[0]
    slope  = cube[1]
    return lambda abscissa: offset + slope * abscissa

def sine_wave(cube):
    amp    = cube[0]
    period = cube[1]
    return lambda abscissa: amp*sin(2*pi / period * abscissa)

np.random.seed(0)

param0= 0.1#0.05
param1= 0.7#5*pi
yunc  = 0.01
nPts  = int(50)
nThPts= int(1e3)

xmin  = -0.1#*pi
xmax  =  0.1#*pi
dx    = 0.1*(xmax - xmin)

model = sine_wave
# model = straight_line

yuncs = np.random.normal(yunc, 1e-2 * yunc, nPts)
thdata= np.linspace(xmin-dx, xmax+dx, nThPts)

xdata = np.random.uniform(xmin, xmax, nPts)
xdata = sort(xdata)

ydata = model([param0,param1])(xdata)

yerr  = np.random.normal(0, yuncs, nPts)
zdata = ydata + yerr

figure(figsize=(10,10))
plot(thdata, model([param0,param1])(thdata))
errorbar(xdata, zdata, yunc*ones(zdata.size), fmt='o')


# In[59]:

# our probability functions
# Taken from the eggbox problem.
model = sine_wave
parameters = ["amp", "period"]

# model = straight_line
# parameters = ["offset", "slope"]

def myprior(cube, ndim, nparams):
    #print "cube before", [cube[i] for i in range(ndim)]
    for i in range(ndim):
        cube[i] = cube[i]# U(0,1) -- default
    #print "python cube after", [cube[i] for i in range(ndim)]

def myloglike(cube, ndim, nparams):
    chi = 1.
    # print "cube", [cube[i] for i in range(ndim)], cube
    # for i in range(ndim):
    #     chi *= -0.5 * ((cube[i] - 0.2) / 0.1)**2#math.cos(cube[i] / 2.) * math.sin(cube[i] / 2.)
    # print "returning", math.pow(2. + chi, 5)
    modelNow = model(cube)(xdata)
    return -0.5*((modelNow - ydata)**2. / yuncs**2.).sum()


# In[60]:

# number of dimensions our problem has
# parameters = ["x", "y"]
n_params = len(parameters)

plt.figure(figsize=(5*n_params, 5*n_params))
# we want to see some output while it is running
progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/2-'); progress.start()
# threading.Timer(2, show, ["chains/2-phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = False, verbose = True,             sampling_efficiency = 'model', n_live_points = 1000, outputfiles_basename='chains/2-')

# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/2-')
s = a.get_stats()


# In[61]:

import json

# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
    json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
    json.dump(s, f, indent=2)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))


# In[62]:

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
    plt.subplot(n_params, n_params, n_params * i + i + 1)
    p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    
    for j in range(i):
        plt.subplot(n_params, n_params, n_params * j + i + 1)
        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
        plt.xlabel(parameters[i])
        plt.ylabel(parameters[j])

# plt.savefig("chains/marginals_multinest.pdf") #, bbox_inches='tight')
# show("chains/marginals_multinest.pdf")

plt.figure(figsize=(5*n_params, 5*n_params))
plt.subplot2grid((5*n_params, 5*n_params), loc=(0,0))
for i in range(n_params):
    #plt.subplot(n_params, n_params, i + 1)
    # outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()
    
    # outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    plt.ylabel("Cumulative probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()

print("Take a look at the pdf files in chains/") 


# In[63]:

p.analyser.get_best_fit()['parameters'], [param0, param1]


# In[64]:

p.analyser.get_stats()


# In[65]:

figure(figsize=(10,10))
plot(thdata, model([param0,param1])(thdata))
plot(thdata, model(p.analyser.get_best_fit()['parameters'])(thdata))
errorbar(xdata, zdata, yunc*ones(zdata.size), fmt='o')


# # 2D Gaussian Modeling

# # PyMultiNest Demo Script -- Full

# In[66]:

from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math
import os
import threading, subprocess
from sys import platform

if not os.path.exists("chains"): os.mkdir("chains")    


# In[58]:

get_ipython().magic('matplotlib inline')
from pylab import *;ion()

from pymultinest.solve import Solver,solve
from numpy import pi, sin, cos, linspace

def gaussian2D(center_y, center_x, width_y, width_x = None, height = None, offset = None):
    """
        Written by Nate Lust
        Edited  by Jonathan Fraine

        Returns a 2D gaussian function with the given parameters

        center_y, center_x  = center position of 2D gaussian profile

        width_y , width_x   = widths of 2D gaussian profile (if width_y != width_x, then gaussian crossection = ellipse)

        height  = height of gaussian profile
                    -- defaults to `1 / np.sqrt(2.*pi*sigma**2.)`

        offset  = background, lower limit value for gaussian
                    -- defaults to 0.0
    """

    if width_x == None:
        width_x = width_y

    if height == None:
        height = np.sqrt(2*np.pi*(width_x**2 + width_y**2))
        height = 1./height

    if offset == None:
        offset = 0.0

    width_x = float(width_x)
    width_y = float(width_y)

    return lambda y,x: height*np.exp(-(((center_x-x)/width_x)**2 + ( (center_y-y)/width_y)**2)/2)+offset

np.random.seed(0)

height  = 0.1#0.05
xcenter = 0.7#5*pi
ycenter = 0.7#5*pi
xwidth  = 0.1
ywidth  = 0.2

yunc  = 0.01
nPts  = int(50)
nThPts= int(1e3)

xmin  = -0.1#*pi
xmax  =  0.1#*pi
dx    = 0.1*(xmax - xmin)

model = sine_wave
# model = straight_line

yuncs = np.random.normal(yunc, 1e-2 * yunc, nPts)
thdata= np.linspace(xmin-dx, xmax+dx, nThPts)

xdata = np.random.uniform(xmin, xmax, nPts)
xdata = sort(xdata)

ydata = model([param0,param1])(xdata)

yerr  = np.random.normal(0, yuncs, nPts)
zdata = ydata + yerr

figure(figsize=(10,10))
plot(thdata, model([param0,param1])(thdata))
errorbar(xdata, zdata, yunc*ones(zdata.size), fmt='o')


# In[59]:

# our probability functions
# Taken from the eggbox problem.
model = sine_wave
parameters = ["amp", "period"]

# model = straight_line
# parameters = ["offset", "slope"]

def myprior(cube, ndim, nparams):
    #print "cube before", [cube[i] for i in range(ndim)]
    for i in range(ndim):
        cube[i] = cube[i]# U(0,1) -- default
    #print "python cube after", [cube[i] for i in range(ndim)]

def myloglike(cube, ndim, nparams):
    chi = 1.
    # print "cube", [cube[i] for i in range(ndim)], cube
    # for i in range(ndim):
    #     chi *= -0.5 * ((cube[i] - 0.2) / 0.1)**2#math.cos(cube[i] / 2.) * math.sin(cube[i] / 2.)
    # print "returning", math.pow(2. + chi, 5)
    modelNow = model(cube)(xdata)
    return -0.5*((modelNow - ydata)**2. / yuncs**2.).sum()


# In[60]:

# number of dimensions our problem has
# parameters = ["x", "y"]
n_params = len(parameters)

plt.figure(figsize=(5*n_params, 5*n_params))
# we want to see some output while it is running
progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/2-'); progress.start()
# threading.Timer(2, show, ["chains/2-phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = False, verbose = True,             sampling_efficiency = 'model', n_live_points = 1000, outputfiles_basename='chains/2-')

# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/2-')
s = a.get_stats()


# In[61]:

import json

# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
    json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
    json.dump(s, f, indent=2)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))


# In[62]:

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
    plt.subplot(n_params, n_params, n_params * i + i + 1)
    p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    
    for j in range(i):
        plt.subplot(n_params, n_params, n_params * j + i + 1)
        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
        plt.xlabel(parameters[i])
        plt.ylabel(parameters[j])

# plt.savefig("chains/marginals_multinest.pdf") #, bbox_inches='tight')
# show("chains/marginals_multinest.pdf")

plt.figure(figsize=(5*n_params, 5*n_params))
plt.subplot2grid((5*n_params, 5*n_params), loc=(0,0))
for i in range(n_params):
    #plt.subplot(n_params, n_params, i + 1)
    # outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()
    
    # outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    plt.ylabel("Cumulative probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()

print("Take a look at the pdf files in chains/") 


# In[63]:

p.analyser.get_best_fit()['parameters'], [param0, param1]


# In[64]:

p.analyser.get_stats()


# In[65]:

figure(figsize=(10,10))
plot(thdata, model([param0,param1])(thdata))
plot(thdata, model(p.analyser.get_best_fit()['parameters'])(thdata))
errorbar(xdata, zdata, yunc*ones(zdata.size), fmt='o')

