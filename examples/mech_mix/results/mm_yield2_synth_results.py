from examples.mech_mix.results.draw_opts import draw_opts


draw_opts["file_prefix"] = "temp_plots/mm_yield2"
ts = [0.0, 0.2, 0.48]

robustness = 1.25726939889e-06
inputs = [
    0.0,
    -5000.000000000002,
    -5000.000000000002,
    -5000.000000000002,
    167.0720688575813,
    5000.0,
]
time = 5.47220206261
# Thu 18 Jul 2019 12:54:47 PM EDT v0.1.1-2-g3486ae5-dirty Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz 16759MB RAM
# python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth ./examples/mech_mix/mm_yield2_synth.py
robustness = 8.63737426402e-06
inputs = [
    0.0,
    -5000.000000000002,
    75.32502554907862,
    1.6186655673193406,
    -5000.000000000002,
    5000.0,
]
time = 77.8449528217
#Thu 18 Jul 2019 12:59:12 PM EDT v0.1.1-2-g3486ae5-dirty Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz 16759MB RAM
#python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth ./examples/mech_mix/mm_yield2_synth.py
robustness = 8.87505140887e-06
inputs = [0.0, 220.24701304275447, 99.78553957945462, -2631.7352880283256, 240.75436315572753, 5000.0]
time = 87.569177866
