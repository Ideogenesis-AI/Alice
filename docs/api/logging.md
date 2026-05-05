# configure_logging

Set up Alice's two-handler logging scheme.

::: alice.configure_logging
    options:
      heading_level: 2

## Examples

```python
import alice

# Default: write to .logging/alice_YYYY-MM-DD_HH-MM-SS.log
alice.configure_logging()

# Explicit log file path
alice.configure_logging("my_run.log")

# Via environment variable
import os
os.environ["ALICE_LOGGING"] = "/scratch/my_run.log"
alice.configure_logging()
```

## Log format

The stream handler (console) uses plain `%(message)s` with no timestamp, keeping INFO output readable. The file handler uses:

```
2026-05-01 12:14:35 [INFO ] sweep 1 / 10: E = -1.234567890123, |ΔE| = 1.2345e-08
2026-05-01 12:14:35 [DEBUG]   site  3 / 15  local E = -1.234567890123
```
