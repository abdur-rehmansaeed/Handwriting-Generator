# Handwriting Generator

This project is a handwriting synthesis application built using PyTorch.
It trains a model on sequences of handwritten strokes paired with text and
can generate handwriting for input sentences.

## Features

- Data preprocessing and normalization
- Custom dataset loader
- LSTM-based sequence-to-sequence model with attention-like window mechanism
- Utilities for training and visualization (see `app.py`/`main.py`)
- Minimal frontend example in `frontend/` showing generated output

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your training data in `data/`:
   - `strokes.npy`: numpy array of (x,y,pen) stroke data
   - `sentences.txt`: corresponding text lines
2. Train the model by running `main.py` or `app.py` (see code for details).
3. Use the provided scripts to generate handwriting for new sentences.

## Files

- `main.py` - core dataset and model implementation
- `app.py` - example training/evaluation interface
- `frontend/` - basic web interface example
- `model/` - saved trained model weights
- `data/` - raw input data

## GitHub

Repository: https://github.com/abdur-rehmansaeed/Handwriting-Generator

## Notes

- The `.gitignore` excludes `__pycache__/` and other unnecessary files.

## License

MIT License
