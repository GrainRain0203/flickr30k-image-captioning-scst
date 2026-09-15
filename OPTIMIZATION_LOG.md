# Image Captioning Optimization Log

This file records optimization attempts, decisions, and the next planned experiment for the Flickr30k image captioning project. Update it after every new optimization or major result.

## Current Reference Setup

- Dataset: Flickr30k.
- Main metric backend for report-quality numbers: `official` via `pycocoevalcap`.
- Main selection metric for caption quality: CIDEr, with BLEU-4, METEOR, and ROUGE-L tracked as supporting metrics.
- Current active preprocessing in `src/data/dataset.py`: direct `transforms.Resize((image_size, image_size))`, no crop, no resize-pad.
- Current `Flickr30k/splits.json`: restored 8:1:1 split.
  - Train: 25426 images
  - Val: 3178 images
  - Test: 3179 images
- Preserved 8:1:1 source backup: `Flickr30k/splits.backup_20260603_223619.json`.
- The previous 9:0.5:0.5 split was backed up as `Flickr30k/splits.9_0p5_0p5_*.json` and is no longer used for follow-up experiments.

## Completed Experiments

| Experiment | Main Change | Official Test CIDEr | Notes |
| --- | --- | ---: | --- |
| `m1_resnet50_attlstm` | ResNet50 + Attention LSTM baseline | 0.4664 | Baseline architecture. |
| `m2_resnet101_attlstm` | ResNet101 encoder | 0.4670 | Slight gain over m1, not meaningful enough. |
| `m3_resnet101_transformer` | Transformer decoder | 0.4690 | Decoder change alone gave only small improvement. |
| `m4_convnext_tiny_transformer` | ConvNeXt-Tiny + Transformer | 0.4940 | Clear architecture gain over ResNet variants. |
| `m4_convnext_tiny_transformer_lrB` | Stage-2 LR variant, beam7 lp0.7 | 0.5010 | Training strategy gave small but limited gain. |
| `m4_convnext_tiny_resizepad384_lra` | Resize-pad preprocessing ablation | 0.5008 | Did not establish resize-pad as a reliable improvement. |
| `m5_convnext_base_resizepad384_lra` | ConvNeXt-Base + Transformer | 0.5313 | Best completed 8:1:1-style m5 result so far, but run naming/preprocessing should be cleaned up before SCST. |
| `m5_convnext_base_lra` | ConvNeXt-Base + Transformer, direct resize384 | 0.5355 | Clean 8:1:1 m5 baseline before SCST; stored in `outputs/m5_convnext_base_resize384_lra`. |
| `m5_convnext_base_lra_scst_safe` | M5 + SCST CIDEr reward fine-tuning, beam9 lp0.9 | 0.5468 | Best completed result; sequence-level reward and decode search improve test CIDEr over M5 baseline. |
| m5 with 9:0.5:0.5 split | More training images, smaller val/test | 0.5368 | Improvement over 8:1:1 m5 is very small, likely split/sample-distribution noise rather than a strong method gain. |

## Key Decisions So Far

- Architecture upgrades helped more than ordinary training-strategy changes.
- Re-splitting Flickr30k from 8:1:1 to 9:0.5:0.5 only produced a tiny improvement:
  - CIDEr: 0.5313 -> 0.5368
  - BLEU-4: 0.2264 -> 0.2283
  - METEOR: 0.2157 -> 0.2172
- Because the split change does not create a substantial gain, continue with the 8:1:1 split for consistency with earlier experiments.
- Do not use resize-pad as the next SCST base. Use a clean direct-resize baseline first.
- Do not introduce 448 resolution before SCST unless the 384 direct-resize m5 baseline is already complete. A 448 run would add a new resolution variable, reduce batch size or increase memory, and make the SCST comparison less clean.

## Clean Baseline Before SCST

Completed clean m5 baseline:

- Split: 8:1:1, using `Flickr30k/splits.backup_20260603_223619.json`.
- Model: `m5`, ConvNeXt-Base + Transformer decoder.
- Preprocessing: pure direct resize to 384, no pad, no crop.
- Stage 1: keep previous m5 settings.
- Stage 2 LR strategy: lrA-style values:
  - Decoder LR: `5e-5`
  - Encoder LR: `1e-5`
- Other settings: keep previous m5 settings unless explicitly changed.
- Suggested run name: `m5_convnext_base_resize384_lra`.

Potential command once the 8:1:1 split is active as `Flickr30k/splits.json`:

```bash
python scripts/train.py \
  --model m5 \
  --data-root Flickr30k \
  --image-size 384 \
  --max-len 32 \
  --batch-size 32 \
  --epochs-stage1 12 \
  --epochs-stage2 8 \
  --lr-decoder 3e-4 \
  --lr-encoder 3e-5 \
  --lr-decoder-stage2 5e-5 \
  --lr-encoder-stage2 1e-5 \
  --encoder-weights convnext_base-6075fbad.pth \
  --run-name m5_convnext_base_resize384_lra
```

Current status: `Flickr30k/splits.json` has been restored to the 8:1:1 split, so `scripts/train.py` can be used directly.

## Planned SCST Experiment

After the clean 8:1:1 m5 resize384 baseline is trained and evaluated:

- Initialize from `outputs/m5_convnext_base_resize384_lra/best.pt`.
- Reward: CIDEr.
- Encoder: freeze for the first SCST attempt.
- Trainable modules: adapter + Transformer decoder.
- LR: start around `1e-5` for trainable non-encoder parameters.
- Mixed objective:

```text
loss = loss_scst + 0.05 * loss_ce
```

- Decode during SCST:
  - Greedy caption: baseline reward.
  - Sampled caption: policy sample with recorded log probabilities.
- Checkpoint selection: use validation CIDEr, not validation CE loss.
- Suggested output directory: `outputs/m5_convnext_base_resize384_lra_scst_cider`.

SCST stability note, 2026-06-08:

- First SCST run showed validation CIDEr collapse, from the clean m5 test CIDEr around `0.5355` to validation CIDEr around `0.03`.
- The SCST `best.pt` and `last.pt` files in that run were 0 bytes, so they must not be used for test evaluation.
- Checkpoint saving was changed to atomic temp-file replacement to avoid leaving 0-byte checkpoint files after interrupted or failed saves.
- SCST sequence log-prob was changed to length-normalized log-prob by default. This reduces the risk that mostly negative sampled advantages over-penalize long sampled captions and push the model toward very short/generic outputs.
- Root-cause diagnosis: the first SCST script used the caption-pair training dataset, where every image appears once per reference caption. Since SCST reward is image-level, one SCST epoch was effectively about five image-level RL epochs. This made the RL update much stronger than intended.
- SCST training now uses an image-level dataset. Each image appears once per epoch, while the auxiliary CE loss samples one reference caption from that image.
- SCST checkpoints are now model-only by default. Optimizer state can be saved with `--save-optimizer`, but model-only checkpoints are smaller and less likely to fail under limited disk quota.
- Next SCST rerun should use a smaller learning rate, stronger CE anchor, advantage normalization, and safer sampling.

SCST safe rerun result:

- Run: `m5_convnext_base_resize384_lra_scst_cider_safe`.
- Epoch 0 validation CIDEr: `0.5353`.
- Epoch 3 validation CIDEr: `0.5437`.
- Validation BLEU-4 improved from `0.2253` to `0.2330`.
- Initial test with beam7 lp0.8 reached CIDEr `0.5439`.
- Validation decode search selected beam9 and length penalty 0.9, with validation CIDEr `0.5491`.
- Final test with beam9 lp0.9 reached CIDEr `0.5468`, BLEU-4 `0.2342`, METEOR `0.2174`, and ROUGE-L `0.4071`.
- Relative to clean M5 baseline, final SCST improves test CIDEr from `0.5355` to `0.5468`, BLEU-4 from `0.2304` to `0.2342`, and ROUGE-L from `0.4055` to `0.4071`.
- This confirms SCST plus validation-selected decoding improves generation quality on the held-out test split and can be reported as the final sequence-level optimization result.
- `scripts/train_scst.py` now supports continuing from an SCST checkpoint by reading the saved `scst_config` model/data configuration.

Implementation plan:

- Modified `src/models/caption_model.py`.
  - Added `sample_decode(...)` that returns sampled token ids, token log probabilities, and a valid-token mask.
- Added `src/metrics/reward.py`.
  - Implemented a CIDEr-style reward suitable for batch-level SCST training.
- Added `src/engine/scst_loop.py`.
  - Implemented `train_scst_one_epoch(...)` and validation generation metrics.
- Added `scripts/train_scst.py`.
  - Loads checkpoint, dataset, references, model, optimizer, and runs SCST fine-tuning.

SCST command after the clean m5 resize384 baseline is trained:

```bash
python scripts/train_scst.py \
  --checkpoint outputs/m5_convnext_base_resize384_lra/best.pt \
  --data-root Flickr30k \
  --output-dir outputs \
  --run-name m5_convnext_base_resize384_lra_scst_cider \
  --epochs 5 \
  --batch-size 16 \
  --val-batch-size 16 \
  --max-len 32 \
  --lr-decoder 1e-5 \
  --lr-encoder 0 \
  --ce-weight 0.05 \
  --num-samples 1 \
  --temperature 1.0 \
  --beam-size 7 \
  --length-penalty 0.8 \
  --metrics-backend official
```

## 448 Resolution Note

Resize to 448 may help if small objects or fine visual details are limiting caption quality, especially with ConvNeXt-Base. It also increases compute and memory, may require a smaller batch size, and introduces a second variable before SCST. Current decision: postpone 448 until after the clean 384 m5 baseline and first SCST result.
