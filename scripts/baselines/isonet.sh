
model=ISONET
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
GMN_NPROPLAYERS=5

CUDA_VISIBLE_DEVICES=$1 python -m baselines.src.ISONET_train model.name=$model dataset.name=$DATASET_NAME data_mode=$data_mode\
                                  training.batch_size=256 training.overwrite=True gmn.variant=deep\
                                  mode=no_attr training.sinkhorn_temp=0.01 dataset.return_adj=True\
                                  training.resume=False use_cost_features=$use_cost_features model.norm_mode=symm