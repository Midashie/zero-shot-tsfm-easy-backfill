import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import numpy as np, torch, timesfm, inspect
print(inspect.signature(timesfm.TimesFM_2p5_200M_torch.from_pretrained))
m=timesfm.TimesFM_2p5_200M_torch.from_pretrained(_IN + "/ZeroShotSchedule/timesfm-2.5",torch_compile=False)
m.compile(timesfm.ForecastConfig(max_context=192,max_horizon=24,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=False,fix_quantile_crossing=True))
x=[np.sin(np.arange(168)/5.0)+0.1*np.random.randn(168) for _ in range(3)]
p,q=m.forecast(horizon=24,inputs=x); print(p.shape,q.shape)
