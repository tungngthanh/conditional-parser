export DATA_PATH=data/Treebank/cleaned_spmrl
export RUN_LANGUAGE="${RUN_LANGUAGE:-Basque}"
export DATA_PATH=$DATA_PATH/$RUN_LANGUAGE
export MODEL_DIR=/directory/to/model/folder
export MODEL_PATH=$MODEL_DIR/check/point
export PREDICT_PATH=$MODEL_DIR/output/file
#export MODE='predict'
export MODE="${MODE:-train}"
export FEAT="${FEAT:-char}"
export BEAM_SIZE=20
export BINARIZE_DIRECTION="${BINARIZE_DIRECTION:-right}"
export BERT_REQUIRE_GRADIENT="${BERT_REQUIRE_GRADIENT:-True}"
export DUMMY_LABEL_MANIPULATING="${DUMMY_LABEL_MANIPULATING:-parent}"
export LEARNING_RATE_SCHEDULE="${LEARNING_RATE_SCHEDULE:-Exponential}"
#export LEARNING_RATE_SCHEDULE='Exponential'
export BERT_MODEL='bert-base-multilingual-cased'
export BATCH_SIZE=5000
if [ $MODE == 'train' ]
then
python -m src.cmds.pointing_constituency_spmrl train -b -d 0 -p exp/spmrl.pointing.constituency.$RUN_LANGUAGE.$FEAT \
--data_path $DATA_PATH --bert bert-base-cased --beam-size $BEAM_SIZE \
-f $FEAT --learning_rate_schedule $LEARNING_RATE_SCHEDULE --bert $BERT_MODEL \
 --batch-size $BATCH_SIZE --conf 'config_spmrl.ini' --binarize_direction $BINARIZE_DIRECTION \
 --dummy_label_manipulating $DUMMY_LABEL_MANIPULATING --bert_requires_grad $BERT_REQUIRE_GRADIENT
#--embed $PRETRAINED_EMBEDDING
elif [ $MODE == 'predict' ]
then
python -m src.cmds.pointing_constituency_spmrl predict -b -d 0 -p $MODEL_PATH \
--data $DATA_PATH --pred $PREDICT_PATH
fi