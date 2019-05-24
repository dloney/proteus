"""
seawall
"""
from __future__ import division
from past.utils import old_div
import numpy as np
from proteus import (Domain, Context)
from proteus.Profiling import logEvent
from proteus.mprans.SpatialTools import Tank2D
from proteus.mprans import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
from proteus.Gauges import PointGauges, LineIntegralGauges, LineGauges
from proteus import WaveTools as wt
from proteus.ctransportCoefficients import smoothedHeaviside
import math

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    ("final_time", 12.0, "Final time for simulation"),
    ("dt_output", 0.01, "Time interval to output solution"),
    ("he", 0.05, "he relative to Length of domain in x"),
    ("slope", (1 / 19.85), "Beta, slope of incline (y/x)"),
    ("slope_length", 50.0, "right extent of domain x(m)"),
    ("mwl", 0.2, "water level"),  # h0
    ("wave_height", 0.35, "Height of the waves in s"),
    ("wave_dir", (1., 0., 0.), "Direction of the waves (from left boundary)"),
    ("wave_type", 'solitaryWave', "type of wave"),
    ("g", [0, -9.8, 0], "Gravity vector in m/s^2")])

USE_WAVE_TOOLS = False
# To define the boundary of the domain
toe = 7.0  # where first slope starts
top = 0.5 # pick top of tank

# To define the initial condition
a = opts.wave_height
k = np.sqrt(3 * a / (4.0 * opts.mwl**3))
L = (2.0 / k) * np.arccosh(1.0 / np.sqrt(0.05))
x0 = 5.9
height = opts.wave_height
direction = opts.wave_dir
wave = wt.SolitaryWave(waveHeight=height,
                       mwl=opts.mwl,
                       depth=opts.mwl,
                       g=np.array(opts.g),
                       waveDir=direction,
                       trans=np.array([x0, 0., 0.]),
                       fast=False)
# ****************** #
# ***** GAUGES ***** #
# ****************** #
# None

# *************************** #
# ***** DOMAIN AND MESH ***** #
# ****************** #******* #
domain = Domain.PlanarStraightLineGraphDomain()
nLevels = 1
# parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0

boundaries = ['left', 'right', 'bottom', 'slope', 'top']
boundaryOrientations = {'bottom': np.array([0., -1., 0.]),
                        'right': np.array([+1., 0., 0.]),
                        'top': np.array([0., +1., 0.]),
                        'left': np.array([-1., 0., 0.]),
                        'slope': np.array([-1., 0., 0.]), }
boundaryTags = dict([(key, i + 1) for (i, key) in enumerate(boundaries)])
vertices = [[0.0, 0.0],  # 0
            [7.0, 0.0],  # 1
            [10.6, 0.18],  # 2
            [10.9, 0.256],  # 3
            [10.948, 0.256],  # 4
            [11.045, 0.202],  # 5
            [15.0, 0.4],  # 6
            [15.0, top],  # 7
            [0.0, top]]  # 8
vertexFlags=[boundaryTags['left'],
               boundaryTags['bottom'],
               boundaryTags['slope'],
               boundaryTags['slope'],
               boundaryTags['slope'],
               boundaryTags['slope'],
               boundaryTags['slope'],
               boundaryTags['right'],
               boundaryTags['top']]
segments= [[0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 0]]
segmentFlags= [boundaryTags['bottom'],
                boundaryTags['slope'],
                boundaryTags['slope'],
                boundaryTags['slope'],
                boundaryTags['slope'],
                boundaryTags['slope'],
                boundaryTags['right'],
                boundaryTags['top'],
                boundaryTags['left']]
regions= [[0.1, 0.01]]
regionFlags= [1]
tank= st.CustomShape(domain, vertices=vertices, vertexFlags=vertexFlags,
                      segments=segments, segmentFlags=segmentFlags,
                      regions=regions, regionFlags=regionFlags,
                      boundaryTags=boundaryTags, boundaryOrientations=boundaryOrientations)


##############################
# domain
#############################

# ****************************** #
# ***** INITIAL CONDITIONS ***** #
# ****************************** #
class zero(object):
    def uOfXT(self, x, t):
        return 0.0


class clsvof_init_cond(object):
    def uOfXT(self, x, t):
        if USE_WAVE_TOOLS:
            h= wave.eta(x, 0)
        else:
            r= np.sqrt(3.0 * a * opts.mwl / (4.0 * opts.mwl**3 * (1 + a)))
            aux1= (np.cosh(r * (x[0] - x0)))**2
            h= a * opts.mwl / aux1
        #
        return x[1] - (h + opts.mwl)


epsFact_consrv_heaviside= 1.5
wavec = np.sqrt(9.8 * opts.mwl * (1 + opts.wave_height))


def weight(x, t):
    return 1.0 - smoothedHeaviside(epsFact_consrv_heaviside * opts.he,
                                   # -ct.epsFact_consrv_heaviside*ct.opts.he+
                                   (x[1] - (max(wave.eta(x, t % (toe / wavec)),
                                                wave.eta(x + toe, t % (toe / wavec))) +
                                            opts.mwl)))


class vel_u(object):
    def uOfXT(self, x, t):
        if USE_WAVE_TOOLS:
            h= wave.eta(x, t)
            vel= wave.u(x, t)[0]
        else:
            r= np.sqrt(3.0 * a * opts.mwl / (4.0 * opts.mwl**3 * (1 + a)))
            aux1= (np.cosh(r * (x[0] - x0)))**2
            c= np.sqrt(9.8 * opts.mwl * (1 + a))
            h= a * opts.mwl / aux1
            vel= c * h / (h + opts.mwl)
        #
        if x[1] <= h + opts.mwl:
            return weight(x, t) * vel
        else:
            return 0.0


class vel_v(object):
    def uOfXT(self, x, t):
        if USE_WAVE_TOOLS:
            if x[1] <= wave.eta(x, t) + opts.mwl:
                return weight(x, t) * wave.u(x, t)[1]
            else:
                return 0.0
        else:
            return 0.0


#
# ****************************** #
# ***** Boundary CONDITIONS***** #
# ****************************** #
tank.BC['top'].setAtmosphere()
tank.BC['bottom'].setFreeSlip()
tank.BC['left'].setFreeSlip()
tank.BC['right'].setFreeSlip()
tank.BC['slope'].setFreeSlip()

domain.MeshOptions.he= opts.he
st.assembleDomain(domain)
domain.MeshOptions.triangleOptions= "VApq30Dena%8.8f" % (
    old_div((opts.he ** 2), 2.0),)

############################################
# ***** Create myTwoPhaseFlowProblem ***** #
############################################
outputStepping= TpFlow.OutputStepping(
    opts.final_time, dt_output=opts.dt_output)
initialConditions= {'pressure': zero(),
                     'pressure_increment': zero(),
                     'vel_u': vel_u(),
                     'vel_v': vel_v(),
                     'clsvof': clsvof_init_cond()}

boundaryConditions= {
    # DIRICHLET BCs #
    'pressure_DBC': lambda x, flag: domain.bc[flag].p_dirichlet.init_cython(),
    'pressure_increment_DBC': lambda x, flag: domain.bc[flag].pInc_dirichlet.init_cython(),
    'vel_u_DBC': lambda x, flag: domain.bc[flag].u_dirichlet.init_cython(),
    'vel_v_DBC': lambda x, flag: domain.bc[flag].v_dirichlet.init_cython(),
    'vel_w_DBC': lambda x, flag: domain.bc[flag].w_dirichlet.init_cython(),
    'clsvof_DBC': lambda x, flag: domain.bc[flag].vof_dirichlet.init_cython(),
    # ADVECTIVE FLUX BCs #
    'pressure_AFBC': lambda x, flag: domain.bc[flag].p_advective.init_cython(),
    'pressure_increment_AFBC': lambda x, flag: domain.bc[flag].pInc_advective.init_cython(),
    'vel_u_AFBC': lambda x, flag: domain.bc[flag].u_advective.init_cython(),
    'vel_v_AFBC': lambda x, flag: domain.bc[flag].v_advective.init_cython(),
    'vel_w_AFBC': lambda x, flag: domain.bc[flag].w_advective.init_cython(),
    'clsvof_AFBC': lambda x, flag: domain.bc[flag].vof_advective.init_cython(),
    # DIFFUSIVE FLUX BCs #
    'pressure_increment_DFBC': lambda x, flag: domain.bc[flag].pInc_diffusive.init_cython(),
    'vel_u_DFBC': lambda x, flag: domain.bc[flag].u_diffusive.init_cython(),
    'vel_v_DFBC': lambda x, flag: domain.bc[flag].v_diffusive.init_cython(),
    'vel_w_DFBC': lambda x, flag: domain.bc[flag].w_diffusive.init_cython(),
    'clsvof_DFBC': lambda x, flag: None}

myTpFlowProblem=TpFlow.TwoPhaseFlowProblem(ns_model=1,
                                             nd=2,
                                             cfl=0.25,
                                             outputStepping=outputStepping,
                                             structured=False,
                                             he=opts.he,
                                             nnx=None,
                                             nny=None,
                                             nnz=None,
                                             domain=domain,
                                             initialConditions=initialConditions,
                                             boundaryConditions=boundaryConditions)
