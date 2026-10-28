# Summary

VUMPS output dataclass.

::: alice.algorithm.vumps.Summary
    options:
      heading_level: 2

## Serialization

```python
import torch

payload = summary.serialize()
torch.save(payload, "state.pt")

# Reload
data    = torch.load("state.pt", weights_only=True)
summary = vumps.Summary.deserialize(data, device="cpu")
```

## See Also

- [Options](options.md) — input options.
- [run](run.md) — returns a `Summary`.
