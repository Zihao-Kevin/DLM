import torch
import  copy

def dp_defense(net,dp_strength=0.001 ,type="laplace",device="cuda"):
    location = 0.0
    threshold = 0.2  # 1e9
    new_dict = copy.deepcopy(net.state_dict())
    if dp_strength == 0:
        return new_dict
    for param_name in net.state_dict():
        param=net.state_dict()[param_name]
        if type == "laplace":
            scale = dp_strength
            # clip 2-norm per sample
            if len(param.shape) == 1:
                norm_factor_a = torch.div(torch.max(torch.norm(param)),
                                      threshold + 1e-6).clamp(min=1.0)
            else:
                norm_factor_a = torch.div(torch.max(torch.norm(param, dim=1)),
                                      threshold + 1e-6).clamp(min=1.0)
            # add laplace noise
            dist_a = torch.distributions.laplace.Laplace(location, scale)
            param_clone = torch.div(param, norm_factor_a) + \
                          dist_a.sample(param.shape).to(device)
        elif type == "gaussian":
            scale = dp_strength
            if len(param.shape) == 1:
                norm_factor_a = torch.div(torch.max(torch.norm(param)),
                                      threshold + 1e-6).clamp(min=1.0)
            else:
                norm_factor_a = torch.div(torch.max(torch.norm(param, dim=1)),
                                      threshold + 1e-6).clamp(min=1.0)
            # add gaussian noise
            dist_a = torch.distributions.normal.Normal(location,scale)
            # param_clone = torch.div(param, norm_factor_a) + \
            #               torch.normal(location, scale, param.shape).to(
            #                   device)
            param_clone = torch.div(param, norm_factor_a) + \
                      dist_a.sample(param.shape).to(device)
        new_dict[param_name] = param_clone * norm_factor_a
    return new_dict

def sparsification(net,spars=5 ,device="cuda"):
    new_dict = copy.deepcopy(net.state_dict())
    for param_name in net.state_dict():
        param = net.state_dict()[param_name]

        percent = spars / 100.0
        up_thr = torch.quantile(torch.abs(param), percent)
        active_up_param_res = torch.where(
            torch.abs(param).double() < up_thr.item(),
            param.double(), float(0.)).to(device)
        param_clone = param - active_up_param_res

        new_dict[param_name] = param_clone
        # new_dict[param_name] = active_up_param_res
    return new_dict