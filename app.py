from pathlib import Path
import os
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
HEATMAP_DIR = STATIC_DIR / "heatmaps"
MODEL_PATH = BASE_DIR / "densenet_pneumonia_v1.h5"

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Create required directories.
for folder in (UPLOAD_DIR, HEATMAP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Load the original trained model without retraining or modifying it.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Saved model not found at {MODEL_PATH}. Keep "
        "'densenet_pneumonia_v1.h5' beside app.py."
    )

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def allowed_file(filename):
    """Check whether the uploaded filename has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image):
    """Prepare one RGB image as a float32 [1, 224, 224, 3] batch."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name):
    """
    Generate a Grad-CAM heatmap for the model's single sigmoid output.

    This visualisation highlights regions that influence the selected
    model output. It is not a medical diagnostic or severity map.
    """
    last_conv_layer = loaded_model.get_layer(last_conv_layer_name)

    grad_model = Model(
        inputs=loaded_model.inputs,
        outputs=[last_conv_layer.output, loaded_model.output],
    )

    with tf.GradientTape() as tape:
        grad_outputs = grad_model(img_array, training=False)

        # Keras models may return outputs as a list or tuple.
        if not isinstance(grad_outputs, (list, tuple)):
            raise RuntimeError(
                "Unexpected Grad-CAM model output structure."
            )

        conv_outputs = grad_outputs[0]
        predictions = grad_outputs[1]

        # Normalise prediction output handling.
        if isinstance(predictions, (list, tuple)):
            if len(predictions) != 1:
                raise RuntimeError(
                    "Grad-CAM expects a model with one prediction output."
                )
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)
        conv_outputs = tf.convert_to_tensor(conv_outputs)

        if predictions.shape.rank != 2 or predictions.shape[-1] != 1:
            raise RuntimeError(
                f"Expected one sigmoid output per image; got shape "
                f"{predictions.shape}."
            )

        target_score = predictions[:, 0]

    grads = tape.gradient(target_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Could not calculate gradients for the selected Grad-CAM layer."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, max_value)

    return heatmap.numpy()


def save_heatmap(heatmap, image):
    """Save a unique Grad-CAM overlay and return its static-relative path."""
    original = cv2.cvtColor(
        np.asarray(image.convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

    resized = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0]),
    )

    colored = cv2.applyColorMap(
        np.uint8(255 * resized),
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(original, 0.6, colored, 0.4, 0)

    filename = f"{uuid.uuid4().hex}_heatmap.jpg"
    output_path = HEATMAP_DIR / filename

    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError("Could not save the Grad-CAM heatmap.")

    return f"heatmaps/{filename}"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_path = None
    heatmap_path = None
    model_score = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Choose a chest X-ray image before submitting."

        elif not allowed_file(uploaded_file.filename):
            error = (
                "Unsupported file type. Upload a JPG, JPEG, PNG, BMP, "
                "or WEBP image."
            )

        else:
            safe_name = secure_filename(uploaded_file.filename)

            if not safe_name or "." not in safe_name:
                error = "Invalid filename. Please choose another image."
            else:
                suffix = safe_name.rsplit(".", 1)[-1].lower()
                unique_name = f"{uuid.uuid4().hex}.{suffix}"
                saved_path = UPLOAD_DIR / unique_name

                try:
                    uploaded_file.save(saved_path)

                    # Verify that the uploaded file is a readable image.
                    with Image.open(saved_path) as source:
                        source.verify()

                    with Image.open(saved_path) as source:
                        image = source.convert("RGB")

                    batch = preprocess_image(image)

                    # Run prediction using the original trained model.
                    raw_prediction = np.asarray(
                        model.predict(batch, verbose=0)
                    ).reshape(-1)

                    if (
                        raw_prediction.size != 1
                        or not np.isfinite(raw_prediction[0])
                    ):
                        raise ValueError(
                            "The model did not return one finite prediction."
                        )

                    score = float(raw_prediction[0])

                    if not 0.0 <= score <= 1.0:
                        raise ValueError(
                            "Expected the saved model to return a sigmoid "
                            "score between 0 and 1."
                        )

                    # This is the project's existing assumed class mapping.
                    # Verify it against the original training labels.
                    result = (
                        "Pneumonia Detected" if score > 0.5 else "Normal"
                    )
                    model_score = round(score * 100, 2)
                    image_path = f"uploads/{unique_name}"

                    # A Grad-CAM failure must not discard a valid prediction.
                    try:
                        heatmap = make_gradcam_heatmap(
                            batch,
                            model,
                            "conv5_block16_concat",
                        )
                        heatmap_path = save_heatmap(heatmap, image)

                    except Exception:
                        app.logger.exception(
                            "Prediction succeeded, but Grad-CAM generation "
                            "failed."
                        )

                except (UnidentifiedImageError, OSError):
                    error = (
                        "The uploaded file could not be read as an image. "
                        "Try another X-ray file."
                    )
                    saved_path.unlink(missing_ok=True)

                except Exception:
                    app.logger.exception("Image prediction failed.")
                    error = (
                        "Prediction failed. Check the server console "
                        "for the detailed error."
                    )
                    saved_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        prediction=result,
        model_score=model_score,
        image_path=image_path,
        heatmap_path=heatmap_path,
        error=error,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        prediction=None,
        model_score=None,
        image_path=None,
        heatmap_path=None,
        error="File is too large. Maximum upload size is 16 MB.",
    ), 413


if __name__ == "__main__":
    # Flask's development server is intended for local testing only.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"

    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=debug_mode,
    )