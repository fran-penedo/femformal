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
#Tue 30 Jul 2019 03:02:49 PM EDT v0.1.1-7-g7703fe3-dirty Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz 16641MB RAM
#python run_benchmark.py --log-level INFO --gthreads 10 --goutputflag 0 milp_synth ./examples/mech_mix/mm_complex_synth.py
robustness = 1.22644962661
inputs = [0.0, 4987.918060429322, 5000.0, -4293.968719747719, 4300.127871602573, 4873.991405827966]
time = 151.207561016
#Wed 31 Jul 2019 05:06:20 PM EDT v0.1.1-11-gea169af-dirty Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz 16641MB RAM
#python run_benchmark.py --log-level INFO --gthreads 10 --goutputflag 0 milp_synth ./examples/mech_mix/mm_complex_synth.py
robustness = 1.22644962661
inputs = [0.0, 4987.918060439159, 5000.0, -4072.671748767706, 4340.405783826487, 4875.190222679593]
time = 165.517359972
