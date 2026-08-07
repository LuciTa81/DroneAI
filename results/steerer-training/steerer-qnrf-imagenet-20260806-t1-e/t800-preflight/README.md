# STEERER T800 preflight

Status: `PASS_PREFLIGHT_ONLY`. Real T800 training had not started when this
evidence was recorded.

- Project/GitHub/home5090 commit:
  `861e42f0609b15618cd1e38f82ddfeaf869d30a0`
- Resume artifact: epoch-5 `milestone-005.pth`, SHA-256
  `2ce6800bcbbdaace881e5f2ecc9f481616a67b9b96254232b7e8e5285ceeafc4`
- Fixed split: 961 Train-derived training images and 240 Train-derived
  validation images; official Test remains sealed.
- GPU: NVIDIA GeForce RTX 5090, idle at inspection; 495 GB free on the data
  volume.
- Verification: local focused suite 178 passed / 2 platform skips; Linux
  synthetic T800 suite 6 passed; CUDA matrix multiplication passed.

The exact launch, monitoring, reboot recovery, and scoring commands are in
`docs/runbooks/steerer-ucf-qnrf-training-home5090.md`. This evidence authorizes
only the preflight conclusion and is not a training result or production
approval.
