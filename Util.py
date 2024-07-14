import re
import torch


def load_state_dict(model, dict_path, strict=False):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(dict_path, map_location=torch.device('cpu'))

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()

    state_dict_alter = {k: v for k, v in state_dict.items() if k in model_dict}

    model_dict.update(state_dict_alter)

    model.load_state_dict(model_dict, strict=strict)
