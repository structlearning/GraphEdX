import torch
from loguru import logger
import time
import datetime
from utils.utils import *
from utils.training_utils import *
from utils.model_utils import *
from utils.data_utils import *
from omegaconf import OmegaConf
import os
import pyfiglet

def test():

    test_data = GraphEditDatasetCombination(conf, "test")
    test_gmn_config = gmn_func(get_default_gmn_config(conf), conf)
    model = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, test_gmn_config).to(conf.training.device)
    checkpoint = torch.load(f'{conf.base_dir}/{conf.training.weights_dir}/{conf.task.name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loading model from {conf.base_dir}/{conf.training.weights_dir}/{conf.task.name}_best.pt")
    start = time.perf_counter()
    mse, rmse, mae = evaluate(model, test_data) 
    end = time.perf_counter()
    logger.info(f"TEST mse: {mse:.6f}\t rmse: {rmse:.6f}\t mae: {mae:.6f}")
    num_params = sum([p.numel() for p in model.parameters()])
    logger.info(f'==================== END OF {conf.task.name} | Took {end-start}s ====================')
    os.makedirs(f"test_results/{conf.dataset.name}", exist_ok=True)
    with open(f"test_results/{conf.dataset.name}/{conf.task.name}.txt", "w") as f:
        f.write(f"mse: {mse:.6f}\t rmse: {rmse:.6f}\t mae: {mae:.6f}\n")
        f.write(f"Number of parameters: {num_params}\n")

    




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
        ) = sampler.fetch_batched_data_by_id(i)
        pred.append(model(batch_data, batch_data_sizes).data)
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
    main_conf = OmegaConf.load("configs/config.yaml")
    cli_conf = OmegaConf.from_cli()

    if cli_conf.mode == 'no_attr':
        banner = pyfiglet.figlet_format('Mode = no attr', font="slant", justify="center")
        print(banner)
        data_config_dir = 'no_attr_data_configs'
        main_conf.log.dir = 'no_attr_logs/test'
        main_conf.training.weights_dir = 'no_attr_weights'
    else:
        data_config_dir = 'data_configs'
        main_conf.log.dir = 'logs/test'
        main_conf.training.weights_dir = 'weights'

    data_conf = OmegaConf.load(f"configs/{data_config_dir}/{cli_conf.dataset.name}.yaml")
    model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, model_conf, cli_conf)
    run_time = "{date:%Y-%m-%d||%H:%M:%S}".format(date=datetime.datetime.now())
    conf.task.name = f"{conf.model.name}_{conf.dataset.name}_{conf.task.name}"
    if conf.training.overwrite:
        open(f"{conf.log.dir}/{conf.task.name}.log", "w").close()  # Clear log file
    logger.add(f"{conf.log.dir}/{conf.task.name}.log")
    logger.info(OmegaConf.to_yaml(conf))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False



    set_seed(conf.training.seed)
    if conf.gmn.variant == 'shallow':
        gmn_func = modify_gmn_main_config_shallow
    else:
        gmn_func = modify_gmn_main_config
    gmn_config = gmn_func(get_default_gmn_config(conf), conf, logger)
    test()
