# Pneumonia Detection using Deep Learning

A Flask web application that runs the existing `densenet_pneumonia_v1.h5` model. It does not train or fine-tune the model.

## Requirements
- Python 3.10–3.12 recommended for compatibility with TensorFlow wheels
- Packages listed in `requirements.txt`

## Run on Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

If Python 3.11 is not installed, install a TensorFlow-supported Python version first. If TensorFlow fails to install, check the Python version and platform before changing the model.

## Notes
- Keep `densenet_pneumonia_v1.h5` in the same folder as `app.py`.
- Images and Grad-CAM outputs are saved with unique names under `static/uploads/` and `static/heatmaps/`.
- The current threshold and mapping preserve the original app's assumption that sigmoid output > 0.5 means pneumonia. Confirm this against the original training labels before relying on the displayed class.
- The displayed score is not a validated clinical probability and is not disease severity.
- This is an educational prototype, not a diagnostic tool.
