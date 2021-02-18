# FEMFormal: A Formal Methods Approach to Boundary Control Synthesis and Verification

![FEMFormal results](https://franpenedo.com/media/femformal.gif)

## What is FEMFormal

FEMFormal is a framework for boundary control synthesis and verification of systems
governed by Partial Differential Equations using a formal methods approach. With
FEMFormal you can write a formal specification of the behavior of a PDE system in a
user-friendly language and obtain (or verify) a control policy that produces a
satisfying trajectory with some guarantees.

In the latest version of FEMFormal, we support 1D and 2D first order and second order
linear PDEs (heat equation and wave equation respectively), as well as special forms of
nonlinear 1D equations.

## Requirements

You need Python2.7, [Gurobi 9.1](https://www.gurobi.com/) or newer, and the use of
virtualenv's or similar is encouraged.

## Quickstart

Clone the repository with:

    $ git clone https://github.com/franpenedo/femformal.git

Install with PIP:

    $ pip install femformal
    
## Spatial-Signal Temporal Logic

The formal language we use to describe the behavior of a PDE system a spatial extension
of Signal Temporal Logic (STL), called Spatial-STL (S-STL). You can see an in depth
description in our publications below, but an example should suffice here:

Consider a metallic rod of 100mm. The
temperature at one end of the rod is fixed at 300k, while a
heat source is applied to the other end. The temperature of the rod follows
a 1D linear heat equation. We want the temperature distribution of the
rod to be within 3k of the linear profile mu(x) = x/4
+ 300 *at all times* between 4 and 5 seconds in the section between 30 and 60
mm. Furthermore, the temperature should *never* exceed 345k
at the point where the heat source is applied (x=100).
We can formulate such a specification using the following S-STL formula:

```
f_ex = G_[4,5] ((∀x ∈ [30, 60] : u(x) - (x/4 + 303) < 0 ) ^
                (∀x ∈ [30, 60] : u(x) - (x/4 + 297) > 0)) ^
       G_[0,5] (∀x ∈ [100, 100] : u(x) - 345 < 0),
```

where *G* stands for *Globally* (also known as *Always*), with the complementary
operator *F* being *in the Future* (or *Eventually*). Note that S-STL admits
quantitative semantics following a similar definition to STL. As an example, a
trajectory that follows the pattern x/4 + 300 at all points in time would have a score
or robustness of 3 (positive and thus satisfying), while one that follows x/4 + 294
would have a robustness of -3 (negative and thus not satisfying).

In order to define the above formula in FEMFormal, you would write the following (this
running example can be found in `examples/heat_mix/`):

```python
from femformal.core import logic

# Define each predicate (∀x ∈ [minbound, maxbound] : u(x) - p(x) ~ 0
# with logic.APCont([minbound, maxbound], "~", p(x), dp(x)/dx))
apc1 = logic.APCont([30, 60], ">", lambda x: 0.25 * x + 297, lambda x: 0.25)
apc2 = logic.APCont([30, 60], "<", lambda x: 0.25 * x + 303, lambda x: 0.25)
apc3 = logic.APCont([100, 100], "<", lambda x: 345.0, lambda x: 0.0)

# Give each predicate a label
cregions = {"A": apc1, "B": apc2, "C": apc3}

# Define the spec as a string using predicate labels
cspec = "((G_[4.0, 5.0] ((A) & (B))) & (G_[0.0, 5.0] (C)))"
```

## S-STL Correction Terms

The way FEMFormal solves the synthesis or verification problem is through a
reformulation into a Mixed-Integer Linear Program (MILP). This reformulation involves a
series of approximations (using the Finite Element Method) and discretizations (both in
time and space). In order to give a guarantee that our solution is correct (i.e., after
obtaining a control input we can say the input is guaranteed to produce a satisfying
trajectory), we must introduce correction terms in the specification. Intuitively, this
amounts to writing a stricter specification so that satisfaction by the approximated,
discretized system guarantees satisfaction by the original PDE system.

So far we have not been able to derive theoretic a priori correction terms, so they must
be obtained through simulation. In particular, you must provide the following:

- The maximum pointwise error the FEM approximation can commit at each *element*.
- The maximum of the spatial derivative of the FEM approximation at each *element*.
- The maximum of the time derivative of the FEM approximation at each *node*.

You can see an example of how to obtain these bounds in
`examples/heat_mix/hm_maxdiff.py`.

## Verification and Synthesis
    
Finally, you must describe the system. In our example, we will define the temperature
evolution of a 1D rod of different materials. First, we define the parameters of the
system (`examples/heat_mix/hm_model.py`):

```python
import numpy as np

# Number of nodes
N = 30

# Rod length
L = 100.0

# Density of each material
rho_steel = 4e-6 * .466e9
rho_brass = 4.5e-6 * .38e9

# Young's modulus of each material
E_steel = 800.0e3
E_brass = 1500.0e3

# Distribution of the material
rho = lambda x: rho_steel if x < 30 or x > 60 else rho_brass
E = lambda x: E_steel if x < 30 or x > 60 else E_brass

# Partition of the domain
xpart = np.linspace(0, L, N + 1)

# Initial temperature at each end
g = [300.0, None]

# Nodal forces
f_nodal = np.zeros(N + 1)

# Time interval for discretization
dt = .05
```

Then, we can define the system itself (`examples/heat_mix/hm_synth_model.py`):

```python
import numpy as np

from femformal.core import system as sys
from femformal.core.fem import heatlinfem as heatlinfem
# The actual module has batch_hm_model here for batch execution
from examples.heat_mix.hm_model import *
# Import the correction terms
from examples.heat_mix.results import hm_maxdiff_results_N30 as mdiff

# Bounds for initial state and nodal force. This is only used in verification
# but must be set here anyway
d_par = 300.0
dset = np.array([[1, d_par], [-1, -d_par]])
fd = lambda x, p: p[0]

# Initial states
u0 = lambda x: fd(x, [d_par])
d0 = heatlinfem.state(u0, xpart, g)

# Maximum integration time
T = 5.0

# Define the template for the input (a piecewise affine function in this case,
# with interpolation points every `input_dt`, bounded to [0, 1e6] and applied
# at x=L
input_dt = 0.5
pwlf = sys.PWLFunction(
    np.linspace(0, T, round(T / input_dt) + 1), ybounds=[0.0, 1e6], x=L
)
fset = pwlf.pset()
fosys = heatlinfem.heatlinfem_mix(xpart, rho, E, g, f_nodal, dt)

# Error bounds for the correction terms (second element of each pair
# is for specifications involving the spatial derivative of the state,
# i.e., heat or strain)
error_bounds = [[mdiff.eps, None], [mdiff.eta, None], [mdiff.nu, None]]
```

Finally, build the casestudy to pass to FEMFormal
(`examples/heat_mix/hm_synth_simple2.py`):

```python
# Time discretization of the S-STL formula will be made with an interval of
# `fdt_mult * dt`
fdt_mult = 1

# Min and max bounds on the predicates used in the specification. Used in the
# MILP encoding, make sure they are as big as you can make them without introducing
# numerical issues in the MILP
bounds = [-1e4, 1e4]

cs = fem.build_cs(
    fosys,
    d0,
    g,
    cregions,
    cspec,
    discretize_system=False,
    pset=[dset, fset],
    f=[fd, pwlf],
    fdt_mult=fdt_mult,
    bounds=bounds,
    error_bounds=error_bounds,
    T=T,
)
```

You can run this synthesis example with:

```shell
$ femformal milp_synth examples/heat_mix/hm_synth_simple2.py
...
robustness = 0.650027305316
inputs = [205109.7044407373, 1000000.0, 1000000.0, 1000000.0, 796938.2494931725, 637445.3323546323, 567391.681809517, 98980.805871299, 0.0, 350110.0171634133, 0.0]
time = 7.51218795776
```

Save those results to a file, then run the following to plot the results:

    $ femformal draw -i examples/heat_mix/results/hm_synth_simple2_N30_results.py animated examples/heat_mix/hm_synth_simple2.py

![Heat example results](https://franpenedo.com/media/femformal_heat.gif)

    $ femformal draw -i examples/heat_mix/results/hm_synth_simple2_N30_results.py inputs examples/heat_mix/hm_synth_simple2.py

![Heat example results](https://franpenedo.com/media/femformal_heat_inputs.png)


## Other examples

In the `examples` folder you can find other examples used in my thesis:

- 1D wave equation with mixed linear materials: `examples/mech_mix`.
- 1D wave equation with nonlinear materials: `examples/mm_nl2`.
- 2D wave equation: `examples/column2d`. 

## Publications

A full description and analysis of our approach can be found in our peer reviewed
publication [Penedo, F., H. Park, and
C. Belta. “Control Synthesis for Partial Differential Equations from Spatio-Temporal
Specifications.” In 2018 IEEE Conference on Decision and Control (CDC), 4890–95, 2018.
https://doi.org/10.1109/CDC.2018.8619313.](https://franpenedo.com/publication/cdc2018/)

A more complete discussion of our approach, including 2D and nonlinear PDEs can be found
in chapter 2 of my PhD thesis [Penedo Alvarez, Francisco. “Formal Methods for Partial
Differential Equations,” 2020.
https://open.bu.edu/handle/2144/41039.](https://open.bu.edu/handle/2144/41039)

## Copyright and Warranty Information

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2016-2021, Francisco Penedo Alvarez (contact@franpenedo.com)
