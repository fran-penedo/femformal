#!/bin/zsh

THREADS=${THREADS:-10}

RUN_CMD="python run_benchmark.py --log-level INFO --gthreads $THREADS --goutputflag 0 milp_synth %s"
DRAW1D_CMD="python run_benchmark.py --log-level INFO draw -i %s snapshots %s"
DRAW2D_CMD="python run_benchmark.py --log-level INFO draw -i %s snapshots_disp %s"
DRAW_INPUTS_CMD="python run_benchmark.py --log-level INFO draw -i %s inputs %s"

run_cs () {
    printf -v run -- $RUN_CMD $1
    base=${1##*/}
    dir=${1%/$base}
    res_file=$dir/results/${base%.py}_results.py
    echo "#$(date) $(version) $(hardware)" >> $res_file
    echo "#$run" >> $res_file
    echo "Running milp_synth:" > /dev/tty
    echo "$run" > /dev/tty
    eval $run | sed -r -n '/\s*\S+ = .*/p' >> $res_file
    echo $res_file
}

hardware () {
    cpu=$(lscpu | grep 'Model name' | awk '{ print $2 }' FS='  +')
    mem="$(free -m --si | grep 'Mem' | awk '{ print $2 }')MB RAM"
    echo "$cpu $mem"
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

version () {
    v=$(git describe --tags --long --dirty --always)
    if [[ $? ]]; then
        echo $v
    else
        echo $(grep version setup.py | sed -r "s/.*'([0-9.]+)'.*/\1/")
    fi
}

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: running script out of virtualenv"
fi

if grep APCont2D $1; then
    run_nd $1 $DRAW2D_CMD
else
    run_nd $1 $DRAW1D_CMD
fi
