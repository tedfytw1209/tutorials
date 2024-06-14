import argparse
import json
from objdetect_inference import OBJDetectInference,transform_vitkeys_from_basemodel,load_model

if __name__ == "__main__":
    '''
    Only for testing all functions in OBJDetectInference.
    set the part what to test (full obj detect, only train, only test, with pre-trained model).
    '''
    #get the config, env, and pre_train network
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print verbose detail during training, recommand True when you are not sure about hyper-parameters",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="pre-trained model path for inference",
    )
    parser.add_argument(
        "-d",
        "--deter",
        default=False,
        action="store_true",
        help="set determinism for model (seed=0)",
    )
    args = parser.parse_args()
    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))
    keys_trans = None
    if config_dict.get("model","")=="vitdet":
        keys_trans = transform_vitkeys_from_basemodel
    pretrained_model = load_model(args.model,keys_trans)
    test_mode = args.testmode
    debug_dict = {} #full test
    debug_dict['use_test'] = False
    if args.deter:
        debug_dict["set_deter"] = True
    #
    inferer = OBJDetectInference(env_dict=env_dict, config_dict=config_dict, debug_dict=debug_dict, verbose=args.verbose)
    inferer.compute(pretrain_network=pretrained_model)