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

def load_model(path=None,state_key='state_dict',transform_dic={}):
    if path:  # make sure to load pretrained model
        if '.ckpt' in path:
            state = torch.load(path, map_location='cpu')
            model = state
        elif '.pth' in path:
            state = torch.load(path, map_location='cpu')
            if state_key:
                model = state[state_key]
            else:
                model = state
        model = transform_keys(model,transform_dic,True)
    else:
        model = None
    return model

def make_weights(data,max_weight=10):
    data_len = len(data)
    data_boxlens = []
    for row in data:
        boxes = row['box']
        bb_lens = [n[-1] for n in boxes]
        if len(bb_lens) > 0:
            mean_len = sum(bb_lens) / len(bb_lens)
            data_boxlens.append(mean_len)
        else:
            data_boxlens.append(0)
    #give small boxes more weights
    avg_len = sum(data_boxlens) / data_len
    data_weight = []
    for each_box in data_boxlens:
        if each_box > 0:
            weight = max(min(avg_len / each_box, max_weight),1)
        else:
            weight = 0
        data_weight.append(weight)
    #give negative avg weights
    avg_weight = sum(data_weight) / data_len
    out_weight = []
    for each_weight in data_weight:
        if each_weight > 0:
            out_weight.append(each_weight)
        else:
            out_weight.append(avg_weight)
    #print weight statistics
    print('Weight statistics:')
    print('Max weight:',max(out_weight))
    print('Min weight:',min(out_weight))
    print('Avg weight:',sum(out_weight) / data_len)
    return out_weight