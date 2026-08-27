# models/

Filled by `code/task3/Group00_MaleBin_task3_improvement_ablation.ipynb` when you
run it on Kaggle. Download these two from the notebook's output and commit them:

| File | What it is |
|---|---|
| `Group00_MaleBin_best.pth` | Final ByteAttnNet checkpoint — `state_dict`, `hparams`, `class_names`, `img_size`, `train_config` and the test metrics, so it can be reloaded without the notebook |
| `Group00_MaleBin_label_map.json` | `label_to_family` / `family_to_label`, `n_classes`, `eval_scope`, `img_size`, architecture and hyper-parameters |

## Reloading the checkpoint



```python
import torch, malebin_common as M
ck = torch.load("Group00_MaleBin_best.pth", map_location="cpu", weights_only=False)
hp = {k: v for k, v in ck["hparams"].items() if k not in ("n_classes", "in_ch")}
model = M.ByteAttnNet(len(ck["class_names"]), in_ch=1, **hp)
model.load_state_dict(ck["state_dict"])
model.eval()
```

`code/task3/Group00_MaleBin_task3_explainability.ipynb` does exactly this and
falls back to training a model if the checkpoint is absent.
