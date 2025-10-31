# train.py
import os
import time
import datetime
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np  # 新增

from Util import save_model, save_model_dis, load_model  # keep existing utilities
from evaluate import evaluation_2class  # used by evaluateDis


def _get_device(model: torch.nn.Module):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_tensor_x(x_seq: List[List[int]], device) -> torch.Tensor:
    x = torch.tensor(x_seq, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)  # [1, T, V]
    return x


def _to_tensor_y(y: List[int], device) -> torch.Tensor:
    y_t = torch.tensor([y], dtype=torch.float32, device=device)
    return y_t


######################### split true/false instances ########################
def splitData_t_f(x_word, Len, y, yg, indexs_sub):
    x_word_sub, Len_sub, y_sub, yg_sub = [], [], [], []
    for i in indexs_sub:
        x_word_sub.append(x_word[i])
        Len_sub.append(Len[i])
        y_sub.append(y[i])
        yg_sub.append(yg[i])
    print(len(x_word_sub), len(Len_sub), len(y_sub), len(yg_sub))
    return x_word_sub, Len_sub, y_sub, yg_sub


###################### pretrain individual D/G #########################
def pre_train_Generator(flag, model, x_word, indexs_sub, Len, y, yg, lr_g, Nepoch_G, modelPath):
    print(f"pre training Generator {flag} ...")
    device = _get_device(model)
    G = model.G_NR if flag == 'nr' else model.G_RN
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_g)

    losses, num_examples_seen = [], 0
    for epoch in range(Nepoch_G):
        random.shuffle(indexs_sub)
        epoch_loss = 0.0

        for i in indexs_sub:
            x_real = _to_tensor_x(x_word[i], device)
            target = _to_tensor_y(yg[i], device)  # generator tries to make D output yg

            seq_len = Len[i]
            x_fake = G(x_real, seq_len)
            pred_fake = model.D(x_fake)

            loss_g = F.mse_loss(pred_fake, target)
            optimizer_G.zero_grad()
            loss_g.backward()
            optimizer_G.step()

            epoch_loss += loss_g.item()
            num_examples_seen += 1

        if epoch % 5 == 0:
            avg_loss = epoch_loss / max(1, len(indexs_sub))
            Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Time}: train num=={num_examples_seen} epoch={epoch}: lossg={avg_loss:.4f}")
            save_model(modelPath, model)

            if len(losses) > 0 and avg_loss > losses[-1]:
                for g in optimizer_G.param_groups:
                    g['lr'] *= 0.5
                print(f"Setting gen lr to {optimizer_G.param_groups[0]['lr']:.4f}")

        losses.append(epoch_loss / max(1, len(indexs_sub)))
        if epoch > 10 and (epoch_loss / max(1, len(indexs_sub))) < 1e-4:
            break

        print(f"epoch={epoch}: lossg={epoch_loss / max(1, len(indexs_sub)):.4f}")


def pre_train_Discriminator(model, x_word, y, x_word_test, y_test, lr_d, Nepoch_D, modelPath_dis):
    print("pre training Discriminator ...")
    device = _get_device(model)
    optimizer_D = torch.optim.Adam(model.D.parameters(), lr=lr_d)

    indexs = [i for i in range(len(y))]
    losses, num_examples_seen = [], 0
    for epoch in range(Nepoch_D):
        random.shuffle(indexs)
        epoch_loss = 0.0

        for i in indexs:
            x_real = _to_tensor_x(x_word[i], device)
            y_real = _to_tensor_y(y[i], device)

            pred_real = model.D(x_real)
            loss_d = F.mse_loss(pred_real, y_real)

            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

            epoch_loss += loss_d.item()
            num_examples_seen += 1

        avg_loss = epoch_loss / max(1, len(indexs))
        if epoch % 5 == 0:
            Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Time}: train num={num_examples_seen} epoch={epoch}: lossd={avg_loss:.4f}")
            try:
                save_model_dis(modelPath_dis, model)
            except TypeError:
                save_model_dis(modelPath_dis, model.D)

            if len(losses) > 0 and avg_loss > losses[-1]:
                for g in optimizer_D.param_groups:
                    g['lr'] *= 0.5
                print(f"Setting dis lr to {optimizer_D.param_groups[0]['lr']:.4f}")

        losses.append(avg_loss)
        print(f"epoch={epoch}: lossd={avg_loss:.4f}")


###################### evaluation #########################
def evaluateDis(model, x_word_test, Y_test):
    """
    Build predictions with model.D and evaluate with evaluation_2class.
    Convert Python lists → numpy arrays to satisfy evaluation_2class.
    - predictions: shape (N, 2), float32
    - targets: class indices (N,), int64
    """
    device = _get_device(model)
    preds = []
    for j in range(len(Y_test)):
        x = _to_tensor_x(x_word_test[j], device)
        with torch.no_grad():
            p = model.D(x).detach().cpu().numpy()[0]  # (2,)
        preds.append(p)

    predictions_np = np.asarray(preds, dtype=np.float32)               # (N, 2)
    targets_np = np.asarray([int(np.argmax(t)) for t in Y_test], dtype=np.int64)  # (N,)

    res = evaluation_2class(predictions_np, targets_np)
    return res


###################### joint training #########################
def train_Gen_Dis(model,
                  x_word, Len, y, yg,
                  index_t, index_f,
                  x_word_test, y_test,
                  lr_g, lr_d, Nepoch, modelPath):
    print("training Generator & Discriminator together ...")
    device = _get_device(model)

    x_word_t, Len_t, y_t, yg_t = splitData_t_f(x_word, Len, y, yg, index_t)
    x_word_f, Len_f, y_f, yg_f = splitData_t_f(x_word, Len, y, yg, index_f)
    indexs = list(index_t) + list(index_f)
    random.shuffle(indexs)

    optimizer_G_NR = torch.optim.Adam(model.G_NR.parameters(), lr=lr_g)
    optimizer_G_RN = torch.optim.Adam(model.G_RN.parameters(), lr=lr_g)
    optimizer_D = torch.optim.Adam(model.D.parameters(), lr=lr_d)

    # Warmup D
    for _ in range(2):
        random.shuffle(indexs)
        for j in indexs:
            x_real = _to_tensor_x(x_word[j], device)
            y_real = _to_tensor_y(y[j], device)

            if j in index_t:
                x_fake = model.G_NR(x_real, Len[j]).detach()
            else:
                x_fake = model.G_RN(x_real, Len[j]).detach()

            pred_real = model.D(x_real)
            pred_fake = model.D(x_fake)

            loss_real = F.mse_loss(pred_real, y_real)
            y_fake = _to_tensor_y(yg[j], device)
            loss_fake = F.mse_loss(pred_fake, y_fake)
            loss_d = 0.5 * (loss_real + loss_fake)

            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

    best_acc = 0.0
    batchsize, f = 200, 0
    for epoch in range(Nepoch):
        random.shuffle(indexs)
        start = f * batchsize
        end = (f + 1) * batchsize
        batch = indexs[start:end] if end <= len(indexs) else indexs[start:]
        if not batch:
            batch = indexs

        # Train G
        for j in batch:
            x_real = _to_tensor_x(x_word[j], device)
            target_fake = _to_tensor_y(yg[j], device)

            if j in index_t:
                x_fake = model.G_NR(x_real, Len[j])
                pred_fake = model.D(x_fake)
                loss_g = F.mse_loss(pred_fake, target_fake)
                optimizer_G_NR.zero_grad()
                loss_g.backward()
                optimizer_G_NR.step()
            else:
                x_fake = model.G_RN(x_real, Len[j])
                pred_fake = model.D(x_fake)
                loss_g = F.mse_loss(pred_fake, target_fake)
                optimizer_G_RN.zero_grad()
                loss_g.backward()
                optimizer_G_RN.step()

        # Train D
        for _ in range(2):
            for j in batch:
                x_real = _to_tensor_x(x_word[j], device)
                y_real = _to_tensor_y(y[j], device)
                if j in index_t:
                    x_fake = model.G_NR(x_real, Len[j]).detach()
                else:
                    x_fake = model.G_RN(x_real, Len[j]).detach()

                pred_real = model.D(x_real)
                pred_fake = model.D(x_fake)

                loss_real = F.mse_loss(pred_real, y_real)
                y_fake = _to_tensor_y(yg[j], device)
                loss_fake = F.mse_loss(pred_fake, y_fake)
                loss_d = 0.5 * (loss_real + loss_fake)

                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()

        # Evaluation
        res = evaluateDis(model, x_word_test, y_test)
        acc = res["accuracy"] if isinstance(res, dict) else (res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else float(res))
        if acc > best_acc:
            save_model(modelPath, model)
            best_acc = acc
            print("new RES:", res)

        Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{Time}: epoch={epoch} acc={acc:.4f}")

        f = (f + 1) % max(1, (len(indexs) // batchsize) if len(indexs) >= batchsize else 1)

    # final output
    try:
        model = load_model(modelPath, model)
    except Exception:
        pass
    RES = evaluateDis(model, x_word_test, y_test)
    print("final Res:", RES)