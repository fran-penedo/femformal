from examples.mech2d.results.draw_opts import draw_opts

draw_opts['file_prefix'] = 'temp_plots/c2d_buckling'
draw_opts['xaxis_scale'] = 1e-3
draw_opts['yaxis_scale'] = 1e-3
draw_opts['zoom_factors'] = [.1, .9]
draw_opts['xticklabels_pick'] = 2
ts = [0.0, 3.45, 4.05]
robustness = 13.5757491128
inputs = [-600.8848490011425, -3368.41075257536, -4000.000000000001, -2252.268110525205, -819.4919776636758, -4000.000000000001, 0.0, -4000.000000000001]
time = 5546.76539898
#Thu 18 Jul 2019 05:36:20 PM EDT v0.1.1-2-g3486ae5-dirty Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz 16759MB RAM
#python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth ./examples/column2d/c2d_synth_buckling.py
