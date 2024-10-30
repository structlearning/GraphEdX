import torch
import time
import datetime
from utils.utils import *
from utils.training_utils import *
from utils.model_utils import *
from utils.data_utils import *
from omegaconf import OmegaConf
from utils.common import get_logger
import pyfiglet
import glob

def test(authors_ckpt=True):
    test_data = GraphEditDatasetCombinationSeparateLabel(conf, "test")
    test_gmn_config = gmn_func(get_default_gmn_config(conf), conf, file_logger)
    model = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, test_gmn_config).to(conf.training.device)
    if authors_ckpt == True:
        weight_files = glob.glob(f'checkpoints/{conf.data_mode}/{conf.method_name}/weights/*{conf.dataset.name}*LAM{conf.model.LAMBDA}*_best.pt')
        weight_file = list(filter(lambda x: conf.dataset.name in x, weight_files))
        file_logger.info(weight_file)
        assert len(weight_file) == 1
        weight_file = weight_file[0]
        checkpoint = torch.load(weight_file)
    else:
        checkpoint = torch.load(f'{conf.base_dir}/{conf.training.weights_dir}/{conf.task.name}_best.pt')

    file_logger.info(f'Loading validation file saved at epoch: {checkpoint["epoch"]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    mse, rmse, mae = evaluate(model, test_data)
    num_params = sum([p.numel() for p in model.parameters()])
    file_logger.info(f"TEST mse: {mse:.6f}\t rmse: {rmse:.6f}\t mae: {mae:.6f}\t num_params: {num_params}")
    file_logger.info(f'==================== END OF {conf.task.name} ====================')



def evaluate(model, sampler):
    model.eval()

    pred = []
    ubs = []
    lbs = []

    n_batches = sampler.create_batches(shuffle=False)
    for i in range(n_batches):
        (
            batch_data,
            batch_data_sizes,
            upper_bounds,
            lower_bounds,
            one_hot_labels,
            query_adj,
            target_adj
        ) = sampler.fetch_batched_data_by_id(i)
        pred.append(model(batch_data, batch_data_sizes, one_hot_labels, query_adj, target_adj).data)
        ubs.append(upper_bounds)
        lbs.append(lower_bounds)
    all_pred = torch.cat(pred, dim=0)
    all_ubs = torch.cat(ubs, dim=0)
    all_lbs = torch.cat(lbs, dim=0)

    common = (torch.nn.functional.relu(all_lbs-all_pred) + torch.nn.functional.relu(all_pred-all_ubs))
    mse = (common**2).mean()
    rmse = torch.sqrt(mse)
    mae = common.mean()

    return mse, rmse, mae


if __name__ == "__main__":
    cli_conf = OmegaConf.from_cli()

    if cli_conf.data_mode == 'equal':
        main_conf = OmegaConf.load("configs/symm_config.yaml")
        banner = pyfiglet.figlet_format('Mode = equal', font="slant", justify="center")
        print(banner)
        data_config_dir = 'no_attr_data_configs'
        main_conf.log.dir = 'no_attr_logs'
        main_conf.training.weights_dir = 'no_attr_weights'
        
    elif cli_conf.data_mode == 'unequal':
        main_conf = OmegaConf.load("configs/asymm_config.yaml")
        banner = pyfiglet.figlet_format('Mode = unequal', font="slant", justify="center")
        print(banner)
        data_config_dir = 'no_attr_asymm_data_configs'
        main_conf.log.dir = 'no_attr_asymm_logs'
        main_conf.training.weights_dir = 'no_attr_asymm_weights'
    elif cli_conf.data_mode == 'label':
        main_conf = OmegaConf.load("configs/label_config.yaml")
        banner = pyfiglet.figlet_format('Mode = label', font="slant", justify="center")
        print(banner)
        data_config_dir = 'label_symm_configs'
        main_conf.log.dir = 'label_symm_logs'
        main_conf.training.weights_dir = 'label_symm_weights'
    else:
        raise NotImplementedError(f"Invalid data mode: {cli_conf.data_mode}")

    
    os.makedirs(main_conf.log.dir, exist_ok=True)
    os.makedirs(main_conf.training.weights_dir, exist_ok=True)
    
    data_conf = OmegaConf.load(f"configs/{data_config_dir}/{cli_conf.dataset.name}.yaml")
    model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, model_conf, cli_conf)
    run_time = "{date:%Y-%m-%d||%H:%M:%S}".format(date=datetime.datetime.now())
    conf.task.name = f"{conf.model.name}_{conf.dataset.name}_{conf.task.name}_{run_time}"
    if conf.dataset.use_labels_as_features == False:
        conf.dataset.one_hot_dim = 1
        
    file_logger = get_logger(conf.task.name, f"{conf.log.dir}/{conf.task.name}.log")
    file_logger.info(OmegaConf.to_yaml(conf))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    set_seed(conf.training.seed)
    if conf.gmn.variant == 'shallow':
        gmn_func = modify_gmn_main_config_shallow
    else:
        gmn_func = modify_gmn_main_config
    gmn_config = gmn_func(get_default_gmn_config(conf), conf, file_logger)
    
    test()
    