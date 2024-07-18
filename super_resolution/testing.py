import argparse
import yaml
from super_resolution import SuperResolutionInference

if __name__ == "__main__":
    #get the config, env, and pre_train network
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train.yaml",
        help="config yaml file that stores hyper-parameters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print verbose detail during training, recommand True when you are not sure about hyper-parameters",
    )
    parser.add_argument(
        "-d",
        "--deter",
        default=False,
        action="store_true",
        help="set determinism for model (seed=0)",
    )
    args = parser.parse_args()
    config_dict = yaml.safe_load(open(args.config_file, "r"))
    pretrained_model = None
    if args.model:
        config_dict['model_path'] = args.model
    debug_dict = {} #full test
    debug_dict['use_train'] = False
    if args.deter:
        debug_dict["set_deter"] = True
    #
    inferer = SuperResolutionInference(config_dict=config_dict, debug_dict=debug_dict, verbose=args.verbose)
    inferer.compute(pretrain_network=pretrained_model)