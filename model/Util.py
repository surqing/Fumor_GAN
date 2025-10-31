# Util.py
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os


############################ Utility Functions #######################################

def softmax(x):
    """Numpy-based softmax, used for evaluation."""
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


############################ Save & Load Model #######################################

def save_model(path, model):
    """
    Save entire GAN model (generator + discriminator) state.
    model must have model.generator and model.discriminator attributes (both torch.nn.Module).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict()
    }
    torch.save(state, path)
    print(f"[INFO] Saved GAN model parameters to {path}.")


def load_model(path, model, map_location="cpu"):
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}, training from scratch.")
        return model

    try:
        # Try PyTorch format
        state = torch.load(path, map_location=map_location)
        model.load_state_dict(state)
        print(f"[INFO] Loaded PyTorch model from {path}")
    except Exception as e:
        print(f"[WARN] Could not load model from {path}: {e}")
        print("[INFO] Skipping old model file, starting fresh training.")
        return model

    return model

########################## Generator / Discriminator Only ############################

def save_model_gen(path, generator):
    """Save only generator parameters."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(generator.state_dict(), path)
    print(f"[INFO] Saved generator parameters to {path}.")


def load_model_gen(path, generator, map_location=None):
    """Load only generator parameters."""
    state_dict = torch.load(path, map_location=map_location)
    generator.load_state_dict(state_dict)
    print(f"[INFO] Loaded generator parameters from {path}.")
    return generator


def save_model_dis(path, discriminator):
    """Save only discriminator parameters."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(discriminator.state_dict(), path)
    print(f"[INFO] Saved discriminator parameters to {path}.")


def load_model_dis(path, discriminator, map_location=None):
    """Load only discriminator parameters."""
    state_dict = torch.load(path, map_location=map_location)
    discriminator.load_state_dict(state_dict)
    print(f"[INFO] Loaded discriminator parameters from {path}.")
    return discriminator

def load_data(data_path):
    """
    Load dataset for rumor detection (compatible with original RumorGAN format).
    Expected to load a pickle file containing:
        train_x, train_y, test_x, test_y
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Data file not found: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Compatible with dict or tuple format
    if isinstance(data, dict):
        train_x = data.get("train_x")
        train_y = data.get("train_y")
        test_x = data.get("test_x")
        test_y = data.get("test_y")
    elif isinstance(data, (list, tuple)) and len(data) == 4:
        train_x, train_y, test_x, test_y = data
    else:
        raise ValueError("[ERROR] Unsupported data format in pickle file")

    print(f"[INFO] Loaded dataset from {data_path}:")
    print(f"  Train size = {len(train_x)} | Test size = {len(test_x)}")

    return train_x, train_y, test_x, test_y