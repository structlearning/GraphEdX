model=SimGNNTensorized
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

CUDA_VISIBLE_DEVICES=$1 python -m baselines.src.simgnn_train model.name=$model dataset.name=$DATASET_NAME task.wandb_project='GED-Base'\
                                  training.batch_size=256 task.name="${FILTERS_3}${NORM_MODE}_cost=${use_cost_features}" training.overwrite=True\
                                  task.wandb_group=${DATASET_NAME}_${model}_${FILTERS_3} dataset.data_type=pyg\
                                  model.filters_1=$FILTERS_3 model.filters_2=$FILTERS_3 model.filters_3=$FILTERS_3\
                                  model.tensor_neurons=$FILTERS_3 model.bins=$FILTERS_3 model.bottleneck_neurons=$FILTERS_3\
                                  mode=no_attr training.resume=False data_mode=$data_mode use_cost_features=$use_cost_features model.norm_mode=symm