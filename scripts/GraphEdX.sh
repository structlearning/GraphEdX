export CUDA_DEVICE_ORDER=PCI_BUS_ID
run_mode=$2
DATASET_NAME=$3
data_mode=$4
edge_variant=$5 # AD, DA, XOR
node_variant=$6 # AD, DA, XOR

SCALE_FACTOR=1
sinkhorn_temp=0.01
use_max=False
use_second_sinkhorn=False
use_second_sinkhorn_log=False
info_str="Correct"

# ALL 10
FILTERS_3=10
GMN_NPROPLAYERS=5

if [ "$edge_variant" = "DA" ]; then
        xor_on_edges=False
        use_m_ms_edge=True
elif [ "$edge_variant" = "AD" ]; then
        xor_on_edges=False
        use_m_ms_edge=False
elif [ "$edge_variant" = "XOR" ]; then
        xor_on_edges=True
        use_m_ms_edge=True
else
        echo "Invalid edge variant"
        exit 1
fi

if [ "$node_variant" = "DA" ]; then
        xor_on_nodes=False
        use_h_hp_node=True
elif [ "$node_variant" = "AD" ]; then
        xor_on_nodes=False
        use_h_hp_node=False
elif [ "$node_variant" = "XOR" ]; then
        xor_on_nodes=True
        use_h_hp_node=True
else
        echo "Invalid node variant"
        exit 1
fi


if [ "${xor_on_edges}" == "False" -a "${xor_on_nodes}" == "False" ]; then
        model=GRAPHEDX_no_xor
elif [ "${xor_on_edges}" == "True" -a  "${xor_on_nodes}" == "False" ]; then
        model=GRAPHEDX_xor_on_edge
elif [ "${xor_on_edges}" = "False" -a "${xor_on_nodes}" == "True" ]; then
        model=GRAPHEDX_xor_on_node
elif [ "${xor_on_edges}" == "True" -a "${xor_on_nodes}" == "True" ]; then
        model=GRAPHEDX_Dual_xor
else
        echo "Invalid xor_on_nodes and xor_on_edges"
        exit 1
fi

if [ "${run_mode}" == "train" ]; then
        script=train_graphedx
else
        script=test_graphedx
fi

if [ "${data_mode}" == "label" ]; then
        model=${model}_label
        script=${script}_label
        if [ "${xor_on_edges}" == "False" ]; then
                echo "ERROR:: ONLY XOR on edge variant is supported for label data"
                exit
        fi
fi

CUDA_VISIBLE_DEVICES=$1 python -m src.$script method_name=${edge_variant}_${node_variant} model.name=$model \
                                    dataset.name=$DATASET_NAME data_mode=$data_mode training.batch_size=256 gmn.variant=deep \
                                    model.output_mode=L1 model.LAMBDA=0.1 model.edge_scale=1\
                                    training.sinkhorn_temp=0.01 training.sinkhorn_noise=0\
                                    dataset.return_adj=True model.use_max=False model.use_second_sinkhorn=False\
                                    model.use_second_sinkhorn_log=False model.use_h_hp_node=$use_h_hp_node model.use_m_ms_edge=$use_m_ms_edge

CUDA_VISIBLE_DEVICES=$1 python -m src.$script method_name=${edge_variant}_${node_variant} model.name=$model \
                                    dataset.name=$DATASET_NAME data_mode=$data_mode training.batch_size=256 gmn.variant=deep \
                                    model.output_mode=L1 model.LAMBDA=1 model.edge_scale=1\
                                    training.sinkhorn_temp=0.01 training.sinkhorn_noise=0\
                                    dataset.return_adj=True model.use_max=False model.use_second_sinkhorn=False\
                                    model.use_second_sinkhorn_log=False model.use_h_hp_node=$use_h_hp_node model.use_m_ms_edge=$use_m_ms_edge

