if [[ -n "$ZSH_VERSION" ]]; then
    setopt KSH_ARRAYS
elif [[ -n "$BASH_VERSION" ]]; then
    echo "Zsh is preferred, this script may or may not work correctly in bash"
else
    echo "Unsupported shell. Please use zsh. Running script in untested environment"
fi

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: running script out of virtualenv"
fi

RUN_CMD="python run_benchmark.py --log-level INFO --gthreads 4 --goutputflag 0 milp_synth %s"
DRAW1D_CMD="python run_benchmark.py --log-level INFO draw -i %s snapshots %s"
DRAW2D_CMD="python run_benchmark.py --log-level INFO draw -i %s snapshots_disp %s"
DRAW_INPUTS_CMD="python run_benchmark.py --log-level INFO draw -i %s inputs %s"

CS_DIR="./examples/"
CS_FILES_1D=("heat_mix/hm_synth_simple2.py" "mech_mix/mm_complex_synth.py" "mech_mix/mm_exists_synth.py" "mech_mix/mm_yield2_synth.py")
# CS_FILES_1D=""
CS_FILES_2D=("column2d/c2d_synth_buckling.py")
CS_FILES_2D=""

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

diff=$(git diff --shortstat | tail -n1)
ans="y"
if [[ $? && $diff != "" ]]; then
    echo -n "There are uncommitted changes. Do you want to run anyway? [Y/n]: "
    read ans
fi

if [[ $ans =~ ^[Yy] ]]; then
    for f in $CS_FILES_1D; do
        run_nd $CS_DIR$f $DRAW1D_CMD
    done

    for f in $CS_FILES_2D; do
        run_nd $CS_DIR$f $DRAW2D_CMD
    done
fi
