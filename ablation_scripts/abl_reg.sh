model=ABL_REG
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DATASET_NAME=$2
data_mode=$3
output_mode=L1
SCALE_FACTOR=1
sinkhorn_temp=0.01
use_max=False
use_h_hp_node=$4
use_m_ms_edge=$5
REG=0.001
info_str="Correct"
if [ $use_max == True ]; then
    info_str+="Smax"
fi
if [ $use_h_hp_node == True ]; then
    info_str+="H-HP"
fi
if [ $use_m_ms_edge == True ]; then
    info_str+="M-MS"
fi
info_str+="REG${REG}"


# ALL 10
FILTERS_3=10
GMN_NPROPLAYERS=5


CUDA_VISIBLE_DEVICES=$1 python -m src.train_abl_reg model.name=$model dataset.name=$DATASET_NAME \
                                  training.batch_size=256 task.name="GS=MSS_0.01${output_mode}_${FILTERS_3}_${GMN_NPROPLAYERS}_s=${SCALE_FACTOR}_GN=0_LAM0.1_${info_str}" training.overwrite=True\
                                  gmn.variant=deep\
                                  gmn.filters_3=$FILTERS_3 gmn.GMN_NPROPLAYERS=$GMN_NPROPLAYERS model.output_mode=$output_mode model.LAMBDA=1\
                                  mode=no_attr model.edge_scale=$SCALE_FACTOR training.sinkhorn_temp=0.01 training.sinkhorn_noise=0 dataset.return_adj=True\
                                  model.use_max=$use_max model.REG=$REG\
                                  model.use_h_hp_node=$use_h_hp_node model.use_m_ms_edge=$use_m_ms_edge  training.resume=False data_mode=$data_mode dataset.use_cost_features=False
