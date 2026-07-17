# Model brief: pet-official-ucf-qnrf

- Paper: Point-Query Quadtree for Crowd Counting, Localization, and More
- Role: UCF-QNRF point-counting and localization research comparison
- Family: point-query quadtree
- Backbone: VGG16-BN
- Parameter count: 20909385
- Input: one RGB UCF-QNRF image with frozen point annotations
- Preprocessing: official 1536 long-side cap with integer flooring and ImageNet normalization
- Coordinate transform: normalized PET (y,x) coordinates mapped to original (x,y) pixels; out-of-frame predictions clipped with count recorded
- Native output: thresholded point coordinates, confidences, and quadtree split map
- Count derivation: number of test-time point queries retained above probability 0.5
- Zone derivation: count predicted points inside each calibrated image zone
- Official protocol: UCF-QNRF long side 1536, test-time probability threshold 0.5
- Rights status: PASS_RESEARCH_ONLY
- Code rights: official README says academic purposes only
- Dataset rights: Kaggle Apache-2.0 accepted with provenance caveat
- Checkpoint rights: official checkpoint treated as academic research only
- Deployment rights: commercial fine-tuning and deployment blocked
- Upstream commit: `5b4dd7da8b11568a3305a88bb7c99a7fc831a998`
- Checkpoint path: `/workspace/data/checkpoints/pet/UCF_QNRF.pth`
- Checkpoint SHA-256: `58324f86782051591075a923522f33d245ce97f2b50cc43f618313d884b966ad`

## Major blocks

- VGG16-BN convolutional backbone
- progressive rectangular-window encoder
- point-query quadtree decoder
- point/non-point classification and coordinate heads

## Feature scales

- stride-8 sparse queries
- stride-4 dense queries

## Original losses

- point/non-point cross entropy
- smooth L1 point-coordinate loss weight 5
- quadtree split loss weight 0.1

## Official reported metrics

- UCF-QNRF MAE 79.53
- UCF-QNRF RMSE 144.32

## Strengths

- native localization
- adaptive computation in dense regions

## Failure modes

- tiny or occluded people
- viewpoint domain shift
- probability-threshold sensitivity
- zone-boundary coordinate errors

## Runtime risks

- legacy official PyTorch API assumptions
- high-resolution transformer memory and latency

## Reviewed code

- /workspace/upstreams/PET/README.md
- /workspace/upstreams/PET/LICENSE
- /workspace/upstreams/PET/models/pet.py
- /workspace/upstreams/PET/models/backbones/backbone_vgg.py
- /workspace/upstreams/PET/models/backbones/vgg.py
- /workspace/upstreams/PET/models/transformer/prog_win_transformer.py
- /workspace/upstreams/PET/util/misc.py
- /workspace/upstreams/PET/preprocess_dataset.py
- /workspace/upstreams/PET/engine.py

Review status: **approved**
