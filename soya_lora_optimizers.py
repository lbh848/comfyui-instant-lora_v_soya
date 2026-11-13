"""Optimizer entry points shared by the public ANIMA and sd-scripts trainers."""

from pytorch_optimizer import Muon as _Muon


class Muon(_Muon):
    """Preserve trainer parameter groups and use AdamW for non-matrix parameters."""

    def __init__(self, params, lr=0.02, **kwargs):
        params = list(params)
        groups = params if params and isinstance(params[0], dict) else [{"params": params}]
        split_groups = []
        for group in groups:
            parameters = list(group["params"])
            for use_muon in (True, False):
                selected = [parameter for parameter in parameters if (parameter.ndim >= 2) == use_muon]
                if selected:
                    split_groups.append({
                        **group, "params": selected, "use_muon": use_muon,
                        "lr": group.get("lr", lr),
                    })
        super().__init__(split_groups, lr=lr, **kwargs)
