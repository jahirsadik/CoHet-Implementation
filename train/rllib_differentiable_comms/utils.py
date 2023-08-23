from typing import List, Optional, Union, Tuple, Iterable
import numpy as np
import torch

def to_torch(x: Union[torch.Tensor, dict, np.ndarray],
             dtype: Optional[torch.dtype] = None,
             device: Union[str, int] = 'cpu'
             ) -> Union[dict, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k == 'obs':
                x[k] = to_torch(v, dtype, device)
            # try:
            #     x[k] = to_torch(v, dtype, device)
            # except TypeError:
            #     print(f"The object with key {k} could not be converted.")

    # elif isinstance(x, Batch):
    #     x.to_torch(dtype, device)
    return x
