# Summary

DMRG output dataclass.

::: alice.algorithm.dmrg.Summary
    options:
      heading_level: 2

## Serialization

```python
import torch

payload = summary.serialize()
torch.save(payload, "result.pt")

# Reload
data    = torch.load("result.pt", weights_only=True)
summary = dmrg.Summary.deserialize(data, device="cpu")
```

## See Also

- [Options](options.md) — input options.
- [run](run.md) — returns a `Summary`.
