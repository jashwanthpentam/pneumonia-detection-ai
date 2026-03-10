from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
from tensorflow.keras.models import Model

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("densenet_pneumonia_v1.h5")

IMG_SIZE = 224


# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# Grad-CAM Heatmap
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


# -----------------------------
# Overlay Heatmap on Image
# -----------------------------
def overlay_heatmap(heatmap, image_path):

    img = cv2.imread(image_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    output_path = os.path.join("static", "heatmap.jpg")

    cv2.imwrite(output_path, result)

    return output_path


# -----------------------------
# Main Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    severity = None
    image_path = None
    heatmap_path = None

    if request.method == "POST":

        file = request.files["file"]

        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        image_path = filepath

        image = Image.open(filepath).convert("RGB")

        img = preprocess_image(image)

        pred = model.predict(img)[0][0]

        if pred > 0.5:
            prediction = "Pneumonia Detected"
            severity = round(pred * 100, 2)
        else:
            prediction = "Normal"
            severity = round((1 - pred) * 100, 2)

        # DenseNet last convolution layer
        heatmap = make_gradcam_heatmap(img, model, "conv5_block16_concat")

        heatmap_path = overlay_heatmap(heatmap, filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        image_path=image_path,
        heatmap_path=heatmap_path
    )


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)