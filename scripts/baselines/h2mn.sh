model=H2MN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DATASET_NAME=$2
data_mode=$3

if [[ "$data_mode" == "unequal" ]]
then
    use_cost_features=True
else
    use_cost_features=False
fi

# ALL 10
FILTERS_3=10

CUDA_VISIBLE_DEVICES=$1 python -m baselines.src.simgnn_train model.name=$model dataset.name=$DATASET_NAME\
                                  training.batch_size=256 training.overwrite=True dataset.data_type=pyg\
                                  mode=no_attr training.resume=False data_mode=$data_mode\
                                  use_cost_features=$use_cost_features model.norm_mode=symm