#!/bin/zsh

if [[ -n "$ZSH_VERSION" ]]; then
    setopt KSH_ARRAYS
fi
DIR="examples/heat_mix"
PREFIX="batch_"
MODEL="hm_model.py"
VERIF_MODEL="hm_verif_model.py"
SYNTH_MODEL="hm_synth_model.py"
RUN_FILE="hm_verif_simple2.py"
SYNTH_FILE="hm_synth_simple2.py"
SYNTH_RUN_FILE_STR="hm_synth_simple2_N%d.py"
MDIFF="hm_maxdiff_results_N"
RESULTS_STR="$DIR/results/hm_verif_simple2_%d_results_N%d.py"
INPUTS="$DIR/results/hm_synth_simple2_results_N30.py"

BATCH_MODEL="$DIR/$PREFIX$MODEL"
BATCH_VERIF_MODEL="$DIR/$PREFIX$VERIF_MODEL"
BATCH_SYNTH_MODEL="$DIR/$PREFIX$SYNTH_MODEL"

VERIFY_CMD="python run_benchmark.py --log-level INFO verify -i %s %s > %s"
VERIFY_RESULTS_CMD="python examples/heat_mix/results/hm_verif_simple2_rob_table.py"
SYNTH_CMD="./run.sh %s"
SYNTH_RESULTS_CMD="python examples/heat_mix/results/hm_synth_simple2_rob_vs_time.py"

ERROR_BOUNDS=(
    "[[None, None], [None, None], [None, None]]"
    "[[mdiff.eps, None], [None, None], [None, None]]"
    "[[mdiff.eps, None], [mdiff.eta, None], [None, None]]"
    "[[mdiff.eps, None], [mdiff.eta, None], [mdiff.nu, None]]"
)
N_ERROR_BOUNDS=${#ERROR_BOUNDS[@]}

for n in {10..100..10}; do
    sed -r "s/N = [0-9]+/N = $n/" < "$DIR/$MODEL" > "$BATCH_MODEL"
    sed -r "s/${MDIFF}[0-9]+/${MDIFF}$n/" < "$DIR/$SYNTH_MODEL" > "$BATCH_SYNTH_MODEL"
    printf -v synth_run_file -- $SYNTH_RUN_FILE_STR "$n"
    cp "$DIR/$SYNTH_FILE" "$DIR/$synth_run_file"
    printf -v run -- $SYNTH_CMD "$DIR/$synth_run_file"
    echo "Running synth:"
    echo "$run"
    # eval $run
    rm "$DIR/$synth_run_file"
done

echo "Aggregating synth results:"
echo $SYNTH_RESULTS_CMD
eval $SYNTH_RESULTS_CMD

for n in {10..100..10}; do
    sed -r "s/N = [0-9]+/N = $n/" < "$DIR/$MODEL" > "$BATCH_MODEL"
    sed -r "s/${MDIFF}[0-9]+/${MDIFF}$n/" < "$DIR/$VERIF_MODEL" > "$BATCH_VERIF_MODEL"
    for ((i = 0; i < $N_ERROR_BOUNDS; i++)); do
        sed -i -r "s/error_bounds = .*/error_bounds = ${ERROR_BOUNDS[$i]}/" "$BATCH_VERIF_MODEL"
        printf -v results -- $RESULTS_STR "$i" "$n"
        printf -v run -- $VERIFY_CMD "$INPUTS" "$DIR/$RUN_FILE" "$results"
        echo "Running verify:"
        echo "$run"
        # eval $run
    done
done

echo "Aggregating verify results:"
echo $VERIFY_RESULTS_CMD
eval $VERIFY_RESULTS_CMD
