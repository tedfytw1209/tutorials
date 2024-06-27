#For all utils functions
from collections import OrderedDict
import torch

def transform_keys(state_dict: OrderedDict, transform_dic: dict = {}, visualize: bool = False):
    """
        Create new state dic with matched name base on transform dict

        Args:
            state_dict: OrderedDict, model's origin state dic
            transform_dic: dict {orgin substr : new substr}
            visualize: bool, print out {old names: new names}

        Returns:
            new_state_dict: OrderedDict, new state dict
        """
    new_state_dict = OrderedDict()
    params_names = [k for k in state_dict.keys()]
    names_dict = OrderedDict()
    for name in params_names:
        new_name = name
        #not transform encoder_pos_embed
        for k,v in transform_dic.items():
            new_name = new_name.replace(k, v)
        #encoder. => feature_extractor.body.
        #new_name = new_name.replace('encoder.', 'feature_extractor.body.')
        new_state_dict[new_name] = state_dict.pop(name)
        names_dict[name] = new_name
    #return
    if visualize:
        print('Transform param name:')
        print([(k,v) for k, v in names_dict.items()])
    return new_state_dict

def load_model(path=None,transform_dic={}):
    if path:  # make sure to load pretrained model
        if '.ckpt' in path:
            state = torch.load(path, map_location='cpu')
            model = state
        elif '.pth' in path:
            state = torch.load(path, map_location='cpu')
            model = state['state_dict']
        model = transform_keys(model,transform_dic,True)
    else:
        model = None
    return model