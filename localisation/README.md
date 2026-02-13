# Localisation

This directory contains localisation files for the Celebrimbor application.

## File Structure

```
localisation/
├── README.md           # This file
├── en.json             # English (default)
└── [future: zh.json]   # Chinese (future)
```

## File Format

Each language file is a JSON file with the following structure:

```json
{
  "_meta": {
    "language": "English",
    "code": "en",
    "version": "1.0.0"
  },
  "category": {
    "key": "Translated text"
  }
}
```

## Usage

```python
from bin.core.localisation import tr, tr_list

# Get translated text
text = tr("toolbar.new", "New")

# Get translated list
features = tr_list("modules.logic_control.features", ["If", "While"])

# With format arguments
message = tr("status.ready", "Ready | Robot: {robot}", robot="go2")
```

## Adding a New Language

1. Copy `en.json` to `{lang_code}.json` (e.g., `zh.json`)
2. Translate all text values
3. Update the `_meta` section
4. The language will be automatically detected

## Supported Categories

| Category | Description |
|----------|-------------|
| app | Application-level text |
| toolbar | Toolbar labels and buttons |
| status | Status bar messages |
| messages | Dialog messages |
| log | Log messages |
| console | Console widget text |
| modules | Module palette text |
| nodes | Node-related text |
| simulation | Simulation messages |
| code_gen | Code generation comments |
