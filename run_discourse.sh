export DATA_PATH=data/pickle_data
export MODE='train'

export FEAT='bert'
export PRETRAINED_EMBEDDING='data/glove.6B.100d.txt'
export LEARNING_RATE_SCHEDULE='Exponential'
export BERT_MODEL='bert-large-cased'
export BATCH_SIZE=5000
python -m src.cmds.pointing_discourse train -b -d 0 -p exp/ptb.pointing.discourse.bert \
--data_path $DATA_PATH \
-f $FEAT --learning_rate_schedule $LEARNING_RATE_SCHEDULE --bert $BERT_MODEL \
 --batch-size $BATCH_SIZE --conf 'discourse_config.ini'
#--embed $PRETRAINED_EMBEDDING

#python -m src.cmds.pointing_discourse predict -b -d 0 -p exp/ptb.pointing.discourse.char \
#--data $DATA_PATH --predict_path $PREDICT_PATH
#-f $FEAT --learning_rate_schedule $LEARNING_RATE_SCHEDULE --bert $BERT_MODEL \
# --batch-size $BATCH_SIZE --conf 'discourse_config.ini'