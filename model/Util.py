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

def _get_submodules(model):
    """
    Return a tuple describing how to access submodules.
    Supports:
      - New naming: G_NR, G_RN, D
      - Old naming: generator, discriminator
    """
    has_new = hasattr(model, "G_NR") and hasattr(model, "G_RN") and hasattr(model, "D")
    has_old = hasattr(model, "generator") and hasattr(model, "discriminator")
    return has_new, has_old


def save_model(path, model):
    """
    Save GAN model parameters.
    - If model has (G_NR, G_RN, D), save those three.
    - Else if model has (generator, discriminator), save those two.
    - Else fallback to model.state_dict().
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    has_new, has_old = _get_submodules(model)

    if has_new:
        state = {
            "G_NR": model.G_NR.state_dict(),
            "G_RN": model.G_RN.state_dict(),
            "D": model.D.state_dict(),
            "__format__": "G_NR_G_RN_D",
        }
        torch.save(state, path)
        print(f"[INFO] Saved GAN (G_NR, G_RN, D) parameters to {path}.")
        return

    if has_old:
        state = {
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict(),
            "__format__": "generator_discriminator",
        }
        torch.save(state, path)
        print(f"[INFO] Saved GAN (generator, discriminator) parameters to {path}.")
        return

    # Fallback
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved generic model state_dict to {path}.")


def load_model(path, model, map_location="cpu"):
    """
    Load GAN model parameters saved by save_model.
    Tries multiple formats gracefully and skips unknown/legacy npz.
    """
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}, training from scratch.")
        return model

    try:
        state = torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"[WARN] Could not load model from {path}: {e}")
        print("[INFO] Skipping old model file, starting fresh training.")
        return model

    has_new, has_old = _get_submodules(model)

    try:
        if isinstance(state, dict) and "__format__" in state:
            fmt = state["__format__"]
            if fmt == "G_NR_G_RN_D" and has_new:
                model.G_NR.load_state_dict(state["G_NR"])
                model.G_RN.load_state_dict(state["G_RN"])
                model.D.load_state_dict(state["D"])
                print(f"[INFO] Loaded GAN (G_NR, G_RN, D) from {path}")
                return model
            if fmt == "generator_discriminator" and has_old:
                model.generator.load_state_dict(state["generator"])
                model.discriminator.load_state_dict(state["discriminator"])
                print(f"[INFO] Loaded GAN (generator, discriminator) from {path}")
                return model

        # If keys look like new naming without __format__
        if isinstance(state, dict) and "G_NR" in state and "G_RN" in state and "D" in state and has_new:
            model.G_NR.load_state_dict(state["G_NR"])
            model.G_RN.load_state_dict(state["G_RN"])
            model.D.load_state_dict(state["D"])
            print(f"[INFO] Loaded GAN (G_NR, G_RN, D) from {path}")
            return model

        # If keys look like old naming without __format__
        if isinstance(state, dict) and "generator" in state and "discriminator" in state and has_old:
            model.generator.load_state_dict(state["generator"])
            model.discriminator.load_state_dict(state["discriminator"])
            print(f"[INFO] Loaded GAN (generator, discriminator) from {path}")
            return model

        # As a last resort, try to load as a flat state_dict into model
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded generic state_dict into model from {path}")
            return model

        print(f"[WARN] Unknown model format in {path}. Skipping load.")
        return model

    except Exception as e:
        print(f"[WARN] Failed to map saved state to current model: {e}")
        print("[INFO] Continuing with randomly initialized weights.")
        return model


########################## Generator / Discriminator Only ############################

def save_model_gen(path, generator_or_model):
    """Save only generator parameters, supports both naming schemes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = generator_or_model
    # If a container model is passed, pick a generator submodule
    if hasattr(obj, "generator"):
        gen = obj.generator
    elif hasattr(obj, "G_NR"):
        # 默认保存 NR 生成器；如果需要也可以扩展为保存两者
        gen = obj.G_NR
    else:
        gen = obj  # assume it's already a module
    torch.save(gen.state_dict(), path)
    print(f"[INFO] Saved generator parameters to {path}.")


def load_model_gen(path, generator_or_model, map_location=None):
    """Load only generator parameters, supports both naming schemes."""
    state_dict = torch.load(path, map_location=map_location)
    obj = generator_or_model
    if hasattr(obj, "generator"):
        obj.generator.load_state_dict(state_dict)
        print(f"[INFO] Loaded generator parameters (generator) from {path}.")
        return obj
    if hasattr(obj, "G_NR"):
        obj.G_NR.load_state_dict(state_dict)
        print(f"[INFO] Loaded generator parameters (G_NR) from {path}.")
        return obj
    # assume it's already a generator module
    obj.load_state_dict(state_dict)
    print(f"[INFO] Loaded generator parameters from {path}.")
    return obj


def save_model_dis(path, discriminator_or_model):
    """Save only discriminator parameters. Accepts container model or submodule."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = discriminator_or_model
    if hasattr(obj, "discriminator"):
        dis = obj.discriminator
    elif hasattr(obj, "D"):
        dis = obj.D
    else:
        dis = obj  # assume it's already a module
    torch.save(dis.state_dict(), path)
    print(f"[INFO] Saved discriminator parameters to {path}.")


def load_model_dis(path, discriminator_or_model, map_location=None):
    """Load only discriminator parameters. Accepts container model or submodule."""
    state_dict = torch.load(path, map_location=map_location)
    obj = discriminator_or_model
    if hasattr(obj, "discriminator"):
        obj.discriminator.load_state_dict(state_dict)
        print(f"[INFO] Loaded discriminator parameters (discriminator) from {path}.")
        return obj
    if hasattr(obj, "D"):
        obj.D.load_state_dict(state_dict)
        print(f"[INFO] Loaded discriminator parameters (D) from {path}.")
        return obj
    # assume it's already a discriminator module
    obj.load_state_dict(state_dict)
    print(f"[INFO] Loaded discriminator parameters from {path}.")
    return obj