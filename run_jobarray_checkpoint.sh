#!/bin/zsh

#SBATCH --mail-user=t.shamoyan@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=nuclear_receptor_with_indications
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --error=%x.%j.err.log

# Compared with normal run_jobarray.sh script this has no --mem-per-gpu restriction and does not train (no --do_train, but --init_checkpoint/-init). This is for doing different eval modes.

# ---------- Parameters ----------
DATASET_ROOT="data_k_fold/yamanishi"
DATASET_TYPE="nuclear_receptor"
DATASET_NAME="with_indications"
FOLD=$SLURM_ARRAY_TASK_ID
DATA_PATH="${DATASET_ROOT}/${DATASET_TYPE}/${DATASET_NAME}/${FOLD}"

MODEL_KGE="DistMult"                # TransE | RotatE | ComplEx | DistMult

HP=`apptainer exec /nfs/data/env/jq.sif jq ".${DATASET_TYPE}.${MODEL_KGE}" hyperparameters.json`

NEGATIVES=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".NEGATIVES" -r)
BATCHSIZE=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".BATCHSIZE" -r)
DIM=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".DIM" -r)
GAMMA=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".GAMMA" -r)
LR=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".LR" -r)
MAX_STEPS=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".MAX_STEPS" -r)
VAL_STEPS=$(expr $MAX_STEPS / 20)
DE=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".DE" -r)
DR=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".DR" -r)
REG=$(echo $HP | apptainer exec /nfs/data/env/jq.sif jq ".REG" -r)

NEG_SAMPLING_METHOD="uniform"               # [ uniform ] | jaccard | count
EVAL_NEG_SAMPLING_METHOD="uniform"          # [ jaccard ] | uniform | count | type

WANDB_ENTITY="l3s-future-lab"
WANDB_PROJECT="${DATASET_TYPE}_${DATASET_NAME}"
WANDB_RUN_NAME="${MODEL_KGE}_fold-${FOLD}_eval-${EVAL_NEG_SAMPLING_METHOD}"

# --------- Load Modules ---------
source ~/.zshrc
conda activate kge && "$@"

# ----- Export Env Variables -----
source .env
export WANDB_API_KEY=$wandb_api_key

# ------------- Run --------------
python -u codes/run.py \
 --wandb_project $WANDB_PROJECT \
 --wandb_entity $WANDB_ENTITY \
 --wandb_run_name $WANDB_RUN_NAME \
 --neg_sampling_method $NEG_SAMPLING_METHOD \
 --eval_neg_sampling_method $EVAL_NEG_SAMPLING_METHOD \
 --cuda \
 --seed 42 \
 --do_valid \
 --do_test \
 --data_path $DATA_PATH \
 --model $MODEL_KGE \
 --valid_steps $VAL_STEPS \
 --log_steps $VAL_STEPS \
 -n $NEGATIVES -b $BATCHSIZE -d $DIM \
 -g $GAMMA -a 1.0 -adv \
 -lr $LR --max_steps $MAX_STEPS \
 -save models/"${DATASET_TYPE}_${DATASET_NAME}_${MODEL_KGE}" \
 -init models/"${DATASET_TYPE}_${DATASET_NAME}_${MODEL_KGE}" \
 --test_batch_size 8 $DE $DR $REG \
