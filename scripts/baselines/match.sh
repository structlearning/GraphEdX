model=Match
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DATASET_NAME=$2
data_mode=$3

if [[ "$data_mode" == "unequal" ]]
then
    use_cost_features=True
else
    use_cost_features=False
fi

FILTERS_3=10
GMN_NPROPLAYERS=5


CUDA_VISIBLE_DEVICES=$1 python -m baselines.src.train model.name=$model dataset.name=$DATASET_NAME training.batch_size=256\
                                  gmn.variant=deep model.output_mode=L2 mode=no_attr model.norm_mode=symm training.resume=False\
                                  data_mode=$data_mode use_cost_features=$use_cost_features model.norm_mode=$NORM_MODE