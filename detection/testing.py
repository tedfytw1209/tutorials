import argparse
import yaml
from objdetect_inference import OBJDetectInference

if __name__ == "__main__":
    '''
    Only for testing all functions in OBJDetectInference.
    set the part what to test (full obj detect, only train, only test, with pre-trained model).
    '''
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
    debug_dict = {} #full test
    debug_dict['use_train'] = False
    if args.deter:
        debug_dict["set_deter"] = True
    #
    inferer = OBJDetectInference(config_dict=config_dict, debug_dict=debug_dict, verbose=args.verbose)
    inferer.compute(pretrain_network=pretrained_model)