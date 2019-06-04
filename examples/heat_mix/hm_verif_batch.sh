set -x

DIR="examples/heat_mix"
PREFIX="batch_"
MODEL="hm_model.py"
VERIF_MODEL="hm_verif_model.py"
RUN_FILE="hm_verif_simple2.py"
MDIFF="hm_maxdiff_results_N"
RESULTS="results/hm_verif_simple2_0_results_N"
INPUTS="$DIR/results/hm_synth_simple2_results_N30.py"

BATCH_MODEL="$DIR/$PREFIX$MODEL"
BATCH_VERIF_MODEL="$DIR/$PREFIX$VERIF_MODEL"

for n in {10..100..10}; do
    sed -r "s/N = [0-9]+/N = $n/" < "$DIR/$MODEL" > "$BATCH_MODEL"
    sed -r "s/${MDIFF}[0-9]+/${MDIFF}$n/" < "$DIR/$VERIF_MODEL" > "$BATCH_VERIF_MODEL"
    python run_benchmark.py --log-level INFO verify -i "$INPUTS" "$DIR/$RUN_FILE" > "$DIR/$RESULTS${n}.py"
done

set +x
