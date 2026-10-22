import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import torch, numpy as np
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
mod=Moirai2Module.from_pretrained(_IN + "/ZeroShotSchedule/moirai-2.0")
m=Moirai2Forecast(module=mod,prediction_length=24,context_length=168,target_dim=1,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0).eval()
x=torch.tensor(np.sin(np.arange(168)/5.0)[None,:,None]+0.1*np.random.randn(3,168,1),dtype=torch.float32)
with torch.no_grad():
    out=m(past_target=x,past_observed_target=torch.ones_like(x,dtype=torch.bool),past_is_pad=torch.zeros(3,168,dtype=torch.bool))
print(out.shape, mod.quantile_levels)
