import sys
import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from MulaTOVA import TopologyOptimizer
import matplotlib.pyplot as plt
from utils import PytorchMinMaxScaler, plot_latent,setDevice,set_seed
import torch
import time

## Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./struct.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

## Set up WandB logging
if config.wandb.log:
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.scope,
                config.nn_type,
                config.width,
                config.window_size,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)
else: 
    wandb_init_args = None

## Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

overrideGPU = False
device = setDevice(overrideGPU) 
torch.autograd.set_detect_anomaly(True)

plt.close('all') 
start = time.perf_counter()

topOpt = TopologyOptimizer(config)
#topOpt.selecting_loading(config.example)
#topOpt.initialzeExperiment() 
#topOpt.initializeFE() 
#topOpt.initializeOptimizer() 
#topOpt.InitializeMaterialModel()
topOpt.optimizeDesign(config)
#topOpt.check_validality()  
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))
print(topOpt.exper_name)
topOpt.plotConvergence() 
