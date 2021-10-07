export TRAIN_DIR='data/WSJ_parsing_clean/02-21.10way.clean'
export DEV_DIR='data/WSJ_parsing_clean/22.auto.clean'
export TEST_DIR='data/WSJ_parsing_clean/23.auto.clean'
export BEAM_SIZE=1
export FEAT="${FEAT:-char}"
export PRETRAINED_EMBEDDING='data/glove.6B.100d.txt'
export LEARNING_RATE_SCHEDULE='Exponential'
export BERT_MODEL='bert-large-cased'
export BATCH_SIZE=5000
export MODE='train'
export MODEL_PATH=/path/to/model
export PREDICT_PATH=/path/to/predict/file
if [ $MODE == 'train' ]
then
python -m src.cmds.pointing_constituency train -b -d 0 -p exp/ptb.pointing.constituency.$FEAT -f $FEAT \
--train $TRAIN_DIR --dev $DEV_DIR --test $TEST_DIR --beam-size $BEAM_SIZE \
--learning_rate_schedule $LEARNING_RATE_SCHEDULE --bert $BERT_MODEL --batch-size $BATCH_SIZE --conf 'config_bert.ini'
#--embed $PRETRAINED_EMBEDDING
elif [ $MODE == 'predict' ]
then
echo $MODE
python -m src.cmds.pointing_constituency predict -b -d 0 -p $MODEL_PATH \
--data $TEST_DIR --pred $PREDICT_PATH
fi