model=Greed_Hinge
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DATASET_NAME=$2
data_mode=$3

FILTERS_3=10
GMN_NPROPLAYERS=5

use_cost_features=False


CUDA_VISIBLE_DEVICES=$1 python -m baselines.src.greed_train model.name=$model dataset.name=$DATASET_NAME\
                                  training.batch_size=256 training.overwrite=True dataset.data_type=pyg\
                                  model.n_layers=5 model.hidden_dim=10 model.out_dim=10 model.output_mode=L1\
                                  mode=no_attr training.resume=False data_mode=$data_mode\
                                  dataset.use_cost_features=$use_cost_features model.norm_mode=asymm\

