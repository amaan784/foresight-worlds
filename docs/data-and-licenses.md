# Data and licenses

- **Physics-IQ** (Motamed et al., Google DeepMind): benchmark code under Apache-2.0, video materials under
  CC BY 4.0. <https://github.com/google-deepmind/physics-IQ-benchmark>. The code is cloned into `third_party/`
  (not committed) and its metric functions are imported unchanged.
- **Physics-IQ Verified** (Anates Labs Research): <https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified>,
  gated, auto-approved.
- No dataset files are committed. `data/` is gitignored; `phase0/physiq_data.py` downloads only the clips
  needed. The images in `docs/img/` contain a few Physics-IQ frames, credited in their captions (CC BY 4.0).
- Phone recordings made with the stage stay on the machine that runs the backend (`outputs/`, gitignored).
- Foresight's own code: MIT, see [../LICENSE](../LICENSE).
