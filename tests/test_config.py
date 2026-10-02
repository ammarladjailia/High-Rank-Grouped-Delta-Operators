from utils.config import load_config
import os

def test_config_loading_and_inheritance():
    cfg_base = load_config("configs/cifar10_base.yaml")
    assert cfg_base["experiment"]["name"] == "cifar10_hrgdelta_base"
    
    cfg_02 = load_config("configs/cifar10_beta_02.yaml")
    assert cfg_02["experiment"]["name"] == "cifar10_beta_02"
    assert cfg_02["model"]["beta_max"] == 2.0
    assert cfg_02["model"]["backbone"] == "ResNet8" # inherited
