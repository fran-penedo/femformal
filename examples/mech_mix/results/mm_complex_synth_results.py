from examples.mech_mix.results.draw_opts import draw_opts


draw_opts["file_prefix"] = "temp_plots/mm_complex"
ts = [0.0, 0.1, 0.2, 0.33, 0.45, 0.5]

robustness = 0.000333304227126
inputs = [
    -384.34465924353054,
    5000.0,
    5000.0,
    -3725.6844346234657,
    4487.968529802179,
    5000.0,
]
time = 135.684077978
# Thu 18 Jul 2019 11:01:03 AM EDT v0.1.1-2-g3486ae5-dirty Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz 16759MB RAM
# python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth ./examples/mech_mix/mm_complex_synth
# Thu 18 Jul 2019 11:01:48 AM EDT v0.1.1-2-g3486ae5-dirty Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz 16759MB RAM
# python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth ./examples/mech_mix/mm_complex_synth.py
robustness = 1.22642909571
inputs = [
    0.0,
    4999.5467253867455,
    4999.999977499392,
    -4023.9002862700886,
    4966.364176377593,
    4999.067292854957,
]
time = 117.656996965
#Thu 18 Jul 2019 12:18:40 PM EDT v0.1.1-2-g3486ae5-dirty Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz 16759MB RAM
#python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth ./examples/mech_mix/mm_complex_synth.py
robustness = 1.22644962661
inputs = [0.0, 4987.91806043411, 5000.0, -4271.605007637, 4572.692507775115, 4852.873684880302]
time = 108.234906912
