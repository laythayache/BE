# Multi-Model Sign Recognition Prototype

This repository is a historical OmniSign engineering artifact that combines two
specialized recognition paths in one live webcam loop:

- a TensorFlow LSTM model for three bounded ASL expression sequences
  (`hello`, `thanks`, and `iloveyou`); and
- a PyTorch hand-landmark classifier for the ASL alphabet plus `space`,
  `delete`, and `nothing`.

MediaPipe supplies face, pose, and hand landmarks. The prototype displays each
model's prediction and an observed frames-per-second counter.

## What this demonstrates

- Live camera capture and landmark extraction
- Thirty-frame sequence input for the LSTM path
- Isolated-sign classification from 63 hand-landmark values
- Routing two model families through one interaction loop
- Confidence-based suppression for the isolated-sign classifier

The configured `0.7` value is an inference confidence threshold, not an
accuracy measurement.

## What this does not demonstrate

This repository is not the complete OmniSign platform or the
community-collected Lebanese Sign Language dataset. It does not implement
unrestricted continuous translation, broad clinical vocabulary, validated
multilingual output, or a documented production deployment. It also contains
no retained benchmark report supporting a general accuracy or latency claim.

The committed model files are research artifacts. Their presence does not make
the repository a reproducible evaluation package: dependency versions,
training logs, model cards, dataset lineage, and environment-specific benchmark
results are not fully recorded here.

## Run the historical prototype

Use an isolated Python environment with TensorFlow, PyTorch, OpenCV, MediaPipe,
and NumPy, then run:

```bash
python main.py
```

A webcam is required. Compatibility depends on the TensorFlow/Keras versions
used to load the historical `action.h5` model.

## Related public artifacts

- [ASL alphabet preprocessing and training baseline](https://github.com/laythayache/Training-the-ASL-dataset)
- [OmniSign dataset-collection prototype](https://github.com/laythayache/dataset-collector)
- [Canonical OmniSign case study](https://laythayache.com/projects/omnisign/)
- [Rafik Hariri University award announcement](https://www.rhu.edu.lb/media-room/news/rhu-team-wins-public-choice-award-at-national-fyp-demo-day-2025)

## License

No license file is currently included. Public availability does not by itself
grant permission to reuse the code, model weights, or other repository content.
