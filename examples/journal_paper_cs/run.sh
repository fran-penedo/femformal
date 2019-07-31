#!/bin/zsh

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

RUN_SCRIPT="./run.sh"
OTHER_SCRIPTS=("./examples/heat_mix/hm_batch.sh")

CS_DIR="./examples/"
CS_FILES=("heat_mix/hm_synth_simple2.py" "mech_mix/mm_complex_synth.py" "mech_mix/mm_exists_synth.py" "mech_mix/mm_yield2_synth.py")
# CS_FILES_1D=""
CS_FILES_2D=("column2d/c2d_synth_buckling.py")
CS_FILES_2D=""


diff=$(git diff --shortstat | tail -n1)
ans="y"
if [[ $? && $diff != "" ]]; then
    echo -n "There are uncommitted changes. Do you want to run anyway? [Y/n]: "
    read ans
fi

if [[ $ans =~ ^[Yy] ]]; then
    for script in $OTHER_SCRIPTS; do
        echo "Running script $script"
        eval $script
    done

    for f in ${CS_FILES[@]}; do
        eval $RUN_SCRIPT $CS_DIR$f
    done
fi
