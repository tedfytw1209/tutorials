import argparse
import yaml
from super_resolution import SuperResolutionInference
from ..utils.utils import load_model

if __name__ == "__main__":
    '''
    Only for testing all functions in SuperResolutionInference.
    set the part what to test (full obj detect, only train, only test, with pre-trained model).
    '''
    #get the config, env, and pre_train network
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.yaml",
        help="environment yaml file that stores environment path",
    )
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
    env_dict = yaml.load(open(args.environment_file, "r"))
    config_dict = yaml.load(open(args.config_file, "r"))
    transform_dic = {
        '.patch_embed.proj': '.patch_embedding.patch_embeddings', 
        '.fc': '.linear',
    }
    pretrained_model = load_model(args.model,transform_dic)
    debug_dict = {} #full test
    debug_dict['use_test'] = False
    if args.deter:
        debug_dict["set_deter"] = True
    #
    inferer = SuperResolutionInference(env_dict=env_dict, config_dict=config_dict, debug_dict=debug_dict, verbose=args.verbose)
    inferer.compute(pretrain_network=pretrained_model)