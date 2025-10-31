# train.py

import os
import time
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model_GAN_RNN import GAN  # your updated PyTorch GAN model
from evaluate import evaluateDis  # assuming it returns (precision, acc, recall, f1)
from Util import load_data  # adjust based on your project

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_generator(model, dataloader, lr_g=1e-3, n_epochs=10, save_path="gan_gen_pretrain.pt"):
    print(f"[Pretrain] Generator for {n_epochs} epochs (lr={lr_g})")
    optimizer_g = optim.Adam(model.generator.parameters(), lr=lr_g)
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        model.generator.train()
        total_loss = 0.0
        for real_x, _, target in dataloader:
            real_x, target = real_x.to(device), target.to(device)
            fake_out = model.generator(real_x)
            loss = criterion(fake_out, target)
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:03d} | Loss_G = {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.generator.state_dict(), save_path)
            print(f"✅ Saved generator model: {save_path}")


def pretrain_discriminator(model, dataloader, lr_d=1e-3, n_epochs=10, save_path="gan_dis_pretrain.pt"):
    print(f"[Pretrain] Discriminator for {n_epochs} epochs (lr={lr_d})")
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=lr_d)
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        model.discriminator.train()
        total_loss = 0.0
        for real_x, _, labels in dataloader:
            real_x, labels = real_x.to(device), labels.to(device)
            preds = model.discriminator(real_x)
            loss = criterion(preds, labels)
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:03d} | Loss_D = {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.discriminator.state_dict(), save_path)
            print(f"✅ Saved discriminator model: {save_path}")


def train_gan(model, dataloader, val_loader, lr_g=1e-3, lr_d=1e-3, n_epochs=50, model_path="gan_full.pt"):
    print(f"[Train] GAN for {n_epochs} epochs (lr_g={lr_g}, lr_d={lr_d})")
    optimizer_g = optim.Adam(model.generator.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=lr_d)
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        total_g_loss, total_d_loss = 0.0, 0.0

        for real_x, _, real_labels in dataloader:
            real_x, real_labels = real_x.to(device), real_labels.to(device)

            # ========== Train Discriminator ==========
            optimizer_d.zero_grad()
            real_preds = model.discriminator(real_x)
            real_loss = criterion(real_preds, real_labels)

            # Fake samples
            fake_x = model.generator(real_x).detach()
            fake_labels = torch.zeros_like(real_labels)
            fake_preds = model.discriminator(fake_x)
            fake_loss = criterion(fake_preds, fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # ========== Train Generator ==========
            optimizer_g.zero_grad()
            gen_x = model.generator(real_x)
            preds = model.discriminator(gen_x)
            # generator tries to fool discriminator → labels=1
            g_loss = criterion(preds, torch.ones_like(real_labels))
            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_g = total_g_loss / len(dataloader)
        avg_d = total_d_loss / len(dataloader)
        print(f"Epoch {epoch:03d} | Loss_G={avg_g:.4f} | Loss_D={avg_d:.4f}")

        # === Evaluate ===
        acc = evaluateDis(model, val_loader)
        if acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = acc
            print(f"🌟 New best model saved! Acc={acc:.4f}")

        if abs(avg_g - avg_d) < 1e-3 and epoch > 10:
            print("Early stopping: G and D converged.")
            break


if __name__ == "__main__":
    print("Initializing GAN-RNN training...")
    model = GAN().to(device)

    # ===== Load data =====
    train_loader, val_loader = load_data(batch_size=64)

    # ===== Pretrain =====
    pretrain_generator(model, train_loader, n_epochs=10)
    pretrain_discriminator(model, train_loader, n_epochs=10)

    # ===== Joint Training =====
    train_gan(model, train_loader, val_loader, n_epochs=100)
