import examples.heatlinfem as fem
import femformal.util as u
import femformal.logic as logic
import numpy as np


Ns = [50 for i in range(50)]
Nlen = len(Ns)
L = 10.0
T = [10.0, 100.0]
dt = .1
# ablist = [((np.random.rand() * 4 - 2) * abs(T[1] - T[0]) / L,
#           np.random.rand() * abs(T[1] - T[0]))
#           for N in Ns]
ablist = [
 (5.010633504873983, 23.302098979971348),
 (-14.694938385207744, 36.630383112605394),
 (-6.0445425127146795, 80.87008508630994),
 (11.4781804587297, 3.9649792723793564),
 (3.7082817514645745, 74.01390422706666),
 (12.18854066337395, 77.57403179686395),
 (17.565169482152502, 83.2308603844469),
 (-12.427285297642271, 75.07625114653213),
 (6.605682185876253, 1.7790603179978004),
 (8.985886770158197, 79.52034734663962),
 (5.942199182839988, 46.42631782978602),
 (-14.872947703697132, 33.8777444067672),
 (-14.37532325088869, 74.74636685886735),
 (12.651663805270362, 86.79467176096047),
 (7.629697191850619, 77.36048036365212),
 (-9.83012800091654, 20.459631176064793),
 (16.89275207938106, 22.305505346152835),
 (-0.8607617419343722, 46.637217101830494),
 (6.128627418998578, 3.624817436818325),
 (-16.502007871518344, 41.630997206368505),
 (-4.805359390574727, 27.115810023962233),
 (0.285755823934692, 6.316473152818956),
 (8.842388178227393, 59.66018745738115),
 (-4.088868943385749, 34.2848142985384),
 (-16.284653131132835, 85.57227263787048),
 (5.592455774855856, 9.672997250437563),
 (-17.30793904130026, 77.98167118979761),
 (-10.75253243298896, 34.75399171488918),
 (-10.157333012864422, 55.035278563865646),
 (-12.915432943758669, 1.505744147678335),
 (-13.478543418531356, 74.60501668553329),
 (2.801624814566894, 37.52083190949668),
 (3.0542025014243963, 54.27351075340623),
 (-5.603669608752398, 89.17714438894674),
 (-8.477113158660977, 8.642460272757498),
 (16.90883017103746, 73.60982864948586),
 (-12.148416313803146, 19.914596796240403),
 (-7.967543463056113, 34.97653194626451),
 (3.750991481879616, 56.67017015804738),
 (-3.4317286821770088, 23.259027053201116),
 (7.027840217221582, 34.0280349574753),
 (0.8708789142336091, 39.87380054910484),
 (7.941740185507108, 36.370897292678904),
 (11.833360587444146, 54.75395703940606),
 (2.8301464201320536, 34.69995637788964),
 (9.387021542577727, 14.413076397031638),
 (-12.994439362874573, 20.411901478372204),
 (-4.642497017828057, 38.773022060367275),
 (-16.259817033882577, 28.85387755164763),
 (6.151313862951533, 75.10748858227345)]

d0s = [[T[0]] + [min(max(a * x + b, T[0]), T[1])
                 for x in np.linspace(0, L, N + 1)[1:-1]] + [T[1]]
       for (a, b), N in zip(ablist, Ns)]

apc1 = logic.APCont([1, 9], -1, lambda x: 9.0 * x)
apc2 = logic.APCont([6, 7], 1, lambda x: 9.0 * x + 15.0)
cregions = {'A': apc1, 'B': apc2}

cspec = "((G_[1, 10] (A)) & (F_[4, 6] (B)))"
# t \in [1,10], T = [10, 100], x \in [1, 9], N = [10, 20, 30, 40, 50], L = 10
eps = [1.0 for N in Ns]

cstrues = [fem.build_cs(
    1000, L, T, 0.01, [T[0]] + [min(max(a * x + b, T[0]), T[1])
                 for x in np.linspace(0, L, 1000 + 1)[1:-1]] + [T[1]],
    cregions, cspec, discretize_system=False) for (a, b), N in zip(ablist, Ns)]

cslist = [fem.build_cs(N, L, T, dt, d0, cregions, cspec, eps=e)
          for N, d0, e in zip(Ns, d0s, eps)]

print "loaded cs"


