# custom_mods/training/checkpoints/

Deploy-ready policy checkpoint bundles. Each bundle is a subdirectory containing:

- `manifest.yaml` — bundle metadata and validation schema
- A model file (e.g. `policy.onnx`) referenced by `manifest.yaml`
- `source.json` — optional, tracks bundle origin (`local` / `huggingface` / `training`)

Example layout:

```
custom_mods/training/checkpoints/
  my_walk_policy/
    manifest.yaml
    policy.onnx
    source.json        # optional: {"type": "local", "src": "/path/to/original"}
```

## Bundle origins

| source.json `type` | Added by |
|--------------------|---------|
| `local`            | Checkpoint panel → Local Import |
| `huggingface`      | Checkpoint panel → HuggingFace Download |
| `training`         | Training pipeline → Bundle Exporter |

Binary model files (`*.onnx`, `*.pt`, `*.safetensors`, `*.bin`) are git-ignored.
Only `manifest.yaml`, `source.json`, `README.md`, and `.gitkeep` files are tracked.
