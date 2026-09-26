# GTSRB evaluation weights

`manifest.json` lists one selected public-pretraining model and 51 tested
GTSRB fine-tuning models. Download the 52 `.pt` assets from the `weights-v1`
release into `weights/files/`, then run the verifier in the repository README.

Each asset contains a `model` state dictionary and provenance digests. The
weights are for inference and attack evaluation; they are not complete
training-resume checkpoints. The manifest maps each asset to its run and,
where applicable, official-test evaluation identifier. It does not include
private examples, attack masks, optimizer state, or RNG state. Per-run DP
accounting does not itself establish a joint privacy guarantee for releasing
multiple models trained on overlapping private data.

The 51 fine-tuned weights include three non-DP Natural baselines and six
zero-noise reviewer controls. These nine models must not be described as
private. The other models' individual accounting statements must not be
mistaken for an epsilon-4 guarantee for the entire collection.
