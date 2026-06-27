# Dataset Workspace

This directory is intentionally kept lightweight. Put downloaded or generated
datasets here when preparing local benchmark runs.

Recommended layout:

```text
datasets/
  asr/              # ASR command audio and manifests
  noise/            # MUSAN/DEMAND/car cabin noise
  tts_mos/          # MOS datasets such as VoiceMOS/SOMOS/BVCC
  can/              # CAN/OBD logs and DBC files
  ui/               # In-vehicle UI screenshots and templates
  full_duplex/      # Full-duplex conversation datasets
  semantic/         # Intent/slot fixtures
  dialogue/         # Multi-turn conversation fixtures
  prepared/         # Generated carvoice-bench audio + test_plan.yaml outputs
```

Use `scripts/prepare_dataset.py` to convert supported source layouts into
`carvoice-bench` test plans.
