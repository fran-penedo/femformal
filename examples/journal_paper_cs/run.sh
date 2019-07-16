RUN_CMD="python run_benchmark.py --log-level INFO --gthreads 10 --goutputflag 0 milp_synth %s"
DRAW1D_CMD="python run_benchmark.py --log-level INFO draw -i %s snapshots %s"
DRAW2D_CMD="python run_benchmark.py --log-level INFO draw -i %s snapshots_disp %s"
DRAW_INPUTS_CMD="python run_benchmark.py --log-level INFO draw -i %s inputs %s"

CS_DIR="./examples/"
CS_FILES_1D="heat_mix/hm_synth_simple2.py"
CS_FILES_2D=""

run_cs () {
    printf -v run -- $RUN_CMD $1
    base=${1##*/}
    dir=${1%/$base}
    res_file=$dir/results/${base%.py}_results.py
    echo "#$(date)" >> $res_file
    echo "#$run" >> $res_file
    echo "Running milp_synth:" > /dev/tty
    echo "$run" > /dev/tty
    eval $run >> $res_file
    echo $res_file
}

run_nd () {
    res_file="$(run_cs $1)"
    printf -v run -- $2 $res_file $1
    echo "Plotting..." > /dev/tty
    echo "$run" > /dev/tty
    eval $run
    printf -v run $DRAW_INPUTS_CMD $res_file $1
    echo "Plotting inputs..." > /dev/tty
    echo "$run" > /dev/tty
    eval $run
}

for f in $CS_FILES_1D; do
    run_nd $CS_DIR$f $DRAW1D_CMD
done

for f in $CS_FILES_2D; do
    run_nd $CS_DIR$f $DRAW2D_CMD
done
