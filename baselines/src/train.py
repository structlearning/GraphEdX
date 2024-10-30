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

def test():
    test_data = data_gen(conf, "test")
    test_gmn_config = gmn_func(get_default_gmn_config(conf), conf)
    model = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, test_gmn_config).to(conf.training.device)
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



def train():
    train_data = data_gen(conf, "train") # * 
    val_data = data_gen(conf, "val") # *  (ADDED CUDA2)
    file_logger.info(f"This uses the {conf.model.classPath}.{conf.model.name} model")
    model = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, gmn_config).to(conf.training.device)
    file_logger.info(model)
    file_logger.info(f"no. of params in model: {sum([p.numel() for p in model.parameters()])}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.training.learning_rate,
        weight_decay=conf.training.weight_decay,
    )

    best_val_mse = 1e5
    best_val_rmse = 1e5
    best_val_mae = 1e5
    run = 0
    patience = conf.training.patience
    counter = 0

    if conf.training.resume:
        banner = pyfiglet.figlet_format('Resuming training', font="slant", justify="center")
        file_logger.info(banner)
        checkpoint = torch.load(conf.training.checkpoint)
        counter = checkpoint['patience'] + 1
        best_val_mse = checkpoint['best_val_mse'].to(conf.training.device)
        best_val_rmse = checkpoint['best_val_rmse'].to(conf.training.device)
        best_val_mae = checkpoint['best_val_mae'].to(conf.training.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        run = checkpoint['epoch'] + 1
        file_logger.info(f"Resuming training from epoch {run} with best val mse: {best_val_mse:.6f} and patience: {counter}")

    while run < conf.training.num_epochs:
        model.train()
        n_batches = train_data.create_batches(shuffle=True)
        epoch_loss = 0
        start_time = time.perf_counter()
        for i in range(n_batches):
            (
                batch_data,
                batch_data_sizes,
                upper_bounds,
                lower_bounds,
            ) = train_data.fetch_batched_data_by_id(i)
            optimizer.zero_grad()
            out = model(batch_data, batch_data_sizes)
            losses = model.compute_loss(lower_bounds, upper_bounds, out)
            losses.backward()
            optimizer.step()
            epoch_loss = epoch_loss + losses.item()


        file_logger.info(
            f"Run: {run} train loss: {epoch_loss/n_batches:.2f} Time: {time.perf_counter()-start_time:.2f}",
        )
        start_time = time.perf_counter()
        mse, rmse, mae = evaluate(model, val_data)
        if mse < best_val_mse:
            best_val_mse = mse
            best_val_rmse = rmse
            best_val_mae = mae
            torch.save({
                'epoch': run,
                'model_state_dict': model.state_dict(),
                'config': conf,
                'patience': counter,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mse': best_val_mse,
                'best_val_rmse': best_val_rmse,
                'best_val_mae': best_val_mae
            }, f'{conf.base_dir}/{conf.training.weights_dir}/{conf.task.name}_best.pt')
            file_logger.info(f'Saving best model to {conf.base_dir}/{conf.training.weights_dir}/{conf.task.name}_best.pt')
            counter = 0
        else:
            counter += 1

        file_logger.info(f"Run: {run} VAL mse (best): {mse:.6f} ({best_val_mse:.6f})\t rmse: {rmse:.6f} ({best_val_rmse:.6f})\t mae: {mae:.6f} ({best_val_mae:.6f})\t Time: {time.perf_counter()-start_time:.6f}")
        torch.save({
                'epoch': run,
                'model_state_dict': model.state_dict(),
                'config': conf,
                'optimizer_state_dict': optimizer.state_dict(),
                'patience': counter,
                'best_val_mse': best_val_mse,
                'best_val_rmse': best_val_rmse,
                'best_val_mae': best_val_mae,
                'val_rmse': rmse,
                'val_mse': mse,
                'val_mae': mae
            }, f'{conf.base_dir}/{conf.training.weights_dir}/{conf.task.name}_latest.pt')

        run += 1
        if counter > patience:
            file_logger.info(f"Breaking out of training loop since patience crossed {conf.training.patience}")
            break


    test()
    




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

    data_conf = OmegaConf.load(f"configs/{data_config_dir}/{cli_conf.dataset.name}.yaml")
    model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, model_conf, cli_conf)

    if not conf.dataset.use_cost_features:
        data_gen = GraphEditDatasetCombination
    else:
        data_gen = GraphEditDatasetCombinationCost

    run_time = "{date:%Y-%m-%d||%H:%M:%S}".format(date=datetime.datetime.now())
    conf.task.name = f"{conf.model.name}_{conf.dataset.name}_{conf.dataset.node_ins_cost}{conf.dataset.node_del_cost}{conf.dataset.node_rel_cost}{conf.dataset.edge_ins_cost}{conf.dataset.edge_del_cost}{conf.dataset.edge_rel_cost}_{conf.task.name}_{run_time}"
    if not os.path.exists(conf.log.dir):
        os.makedirs(conf.log.dir)
    if conf.training.overwrite:
        open(f"{conf.log.dir}/{conf.task.name}.log", "w").close()  # Clear log file
    # logger.add(f"{conf.log.dir}/{conf.task.name}.log")
    file_logger = get_logger(conf.task.name, f"{conf.log.dir}/{conf.task.name}.log")
    if conf.data_mode == "unequal" and conf.dataset.use_cost_features:
        conf.dataset.one_hot_dim = 4
        banner = pyfiglet.figlet_format('Using costs as features', font="slant", justify="center")
        file_logger.info(banner)

    file_logger.info(banner)
    file_logger.info(OmegaConf.to_yaml(conf))

    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    set_seed(conf.training.seed)
    if conf.gmn.variant == 'shallow':
        gmn_func = modify_gmn_main_config_shallow
    else:
        gmn_func = modify_gmn_main_config
    gmn_config = gmn_func(get_default_gmn_config(conf), conf, file_logger)
    train()
