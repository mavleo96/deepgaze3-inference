from io import BytesIO
from typing import List, Tuple

import deepgaze_pytorch
import numpy as np
import orjson
import torch
from flask import Flask, request
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
from torchvision import transforms

# Flask server
app = Flask("deepgaze3-model-server")
app.logger.setLevel("DEBUG")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device}")

# Image transform
transform = transforms.Compose([transforms.ToTensor()])

# Load model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True)
model.to(device)
model.eval()
app.logger.info("Model loaded.")

if device.type == "cuda":
    model = torch.compile(model)

    # Trigger compilation
    img_tensor = torch.randn(1, 3, 1024, 1024)
    centerbias_tensor = torch.randn(1, 1, 1024, 1024)
    x_hist_tensor = torch.randint(0, 1024, (1, 4))
    y_hist_tensor = torch.randint(0, 1024, (1, 4))
    log_density = model(img_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
    app.logger.info("Model compiled.")


def process_stimulus(
    stimulus_files: List[BytesIO],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a batch of items and return stacked tensors."""
    image_arrays = [np.array(Image.open(f)) for f in stimulus_files]
    image_tensor = torch.stack([transform(image) for image in image_arrays]).to(device)

    B, _, H, W = image_tensor.shape
    centerbias_tensor = torch.tensor(create_centerbias((H, W), B), device=device)

    return image_tensor, centerbias_tensor


def create_centerbias(stimulus_shape: Tuple[int, int], batch_size: int) -> np.ndarray:
    """Create and normalize centerbias for the given stimulus shape."""
    # Note: Uniform bias is used below
    centerbias_template = np.zeros((1024, 1024))
    centerbias = zoom(
        centerbias_template,
        (
            stimulus_shape[0] / centerbias_template.shape[0],
            stimulus_shape[1] / centerbias_template.shape[1],
        ),
        order=0,
        mode="nearest",
    )
    centerbias = centerbias - logsumexp(centerbias)
    return np.stack([centerbias] * batch_size, axis=0)


def process_fixations(
    x_hist: List[List[int]], y_hist: List[List[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process fixations and return stacked tensors."""
    x_hist_tensor = torch.tensor(get_fixation_history(x_hist), device=device)
    y_hist_tensor = torch.tensor(get_fixation_history(y_hist), device=device)
    return x_hist_tensor, y_hist_tensor


def get_fixation_history(fixation_coordinates: List[List[int]]) -> np.ndarray:
    """Extract fixation history for the model's included fixations."""
    history = []
    for item in fixation_coordinates:
        item_history = []
        # Last 4 fixations from history reversed
        for index in model.included_fixations:
            try:
                item_history.append(item[index])
            except IndexError:
                item_history.append(np.nan)
        history.append(item_history)
    return np.array(history)


def sample_log_density(
    log_density: torch.Tensor, temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = log_density.shape

    # Conditional log density with temperature
    # Temperature is a hyperparameter that controls the randomness of the sampling
    scaled_log_density = log_density / temperature
    prob = torch.exp(scaled_log_density)
    prob = prob / prob.sum(dim=(2, 3), keepdim=True)

    flattened_prob = prob.reshape(B, 1, -1)
    sample = torch.distributions.Categorical(probs=flattened_prob).sample()
    batch_h, batch_w = torch.unravel_index(sample, (H, W))

    # Return pixel coordinates
    return batch_w, batch_h


@app.route("/conditional_log_density", methods=["POST"])
def conditional_log_density():
    """Handle conditional log density computation for batch requests."""
    batch_data = orjson.loads(request.form["json_data"])
    stimulus_files = request.files.getlist("stimulus")

    # Validate input
    if not isinstance(batch_data, dict):
        return orjson.dumps({"error": "Request must contain x_hist and y_hist arrays"}), 400
    if "x_hist" not in batch_data or not isinstance(batch_data["x_hist"], list):
        return orjson.dumps({"error": "Request must contain x_hist array"}), 400
    if "y_hist" not in batch_data or not isinstance(batch_data["y_hist"], list):
        return orjson.dumps({"error": "Request must contain y_hist array"}), 400
    if len(batch_data["x_hist"]) != len(batch_data["y_hist"]):
        return orjson.dumps({"error": "x_hist and y_hist lengths must match"}), 400
    if len(stimulus_files) != len(batch_data["x_hist"]):
        return (orjson.dumps({"error": "stimulus files and hist lengths must match"}), 400)

    image_tensor, centerbias_tensor = process_stimulus(stimulus_files)
    x_hist_tensor, y_hist_tensor = process_fixations(batch_data["x_hist"], batch_data["y_hist"])

    log_density = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

    return orjson.dumps({"log_density": log_density.cpu().tolist()})


@app.route("/sample_fixations", methods=["POST"])
def sample_fixations():
    """Sample fixations from log density."""
    try:
        n_fixations = int(request.form["n_fixations"])
        temperature = float(request.form.get("temperature", "1.0"))
    except (ValueError, KeyError) as e:
        return orjson.dumps({"error": f"Invalid parameter: {str(e)}"}), 400

    if n_fixations <= 0:
        return orjson.dumps({"error": "n_fixations must be a positive integer"}), 400
    if temperature <= 0:
        return orjson.dumps({"error": "temperature must be positive"}), 400

    stimulus_files = request.files.getlist("stimulus")
    if len(stimulus_files) == 0:
        return orjson.dumps({"error": "No stimulus files provided"}), 400

    image_tensor, centerbias_tensor = process_stimulus(stimulus_files)
    B, _, H, W = image_tensor.shape

    # Fixation history is initialized with the center of the image in pixel coordinates
    x_hist_tensor = torch.tensor(
        [[W // 2] + [np.nan] * (len(model.included_fixations) - 1)] * B, device=device
    )
    y_hist_tensor = torch.tensor(
        [[H // 2] + [np.nan] * (len(model.included_fixations) - 1)] * B, device=device
    )

    x_fixations = [[] for _ in range(B)]
    y_fixations = [[] for _ in range(B)]
    # We sample fixations sequentially conditioned on the previous fixations
    for _ in range(n_fixations):
        log_density = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
        x_fix, y_fix = sample_log_density(log_density, temperature)
        x_hist_tensor = torch.cat([x_fix, x_hist_tensor[:, :-1]], dim=1)
        y_hist_tensor = torch.cat([y_fix, y_hist_tensor[:, :-1]], dim=1)
        for i in range(B):
            x_fixations[i].append(x_fix[i].item())
            y_fixations[i].append(y_fix[i].item())

    return orjson.dumps({"x_fixations": x_fixations, "y_fixations": y_fixations})


@app.route("/type", methods=["GET"])
def type():
    """Return model type and version information."""
    return orjson.dumps({"type": "DeepGazeIII", "version": "v1.1.0"})


def main():
    app.run(host="localhost", port="4000", debug="True", threaded=True)


if __name__ == "__main__":
    main()
