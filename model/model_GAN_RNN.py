# model_GAN_RNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility functions
# -----------------------------
def init_matrix(shape):
    return torch.randn(shape) * 0.1

def init_vector(shape):
    return torch.zeros(shape)

# -----------------------------
# Main GAN model
# -----------------------------
class GAN(nn.Module):
    def __init__(self, vocab_size, hidden_size=100, n_class=2, momentum=0.9, device="cpu"):
        super(GAN, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.momentum = momentum

        # Define submodules
        self.G_NR = self.Generator(vocab_size, hidden_size)
        self.G_RN = self.Generator(vocab_size, hidden_size)
        self.D = self.Discriminator(vocab_size, hidden_size, n_class)

        # Move to device
        self.to(device)

    # -----------------------------
    # Inner Generator
    # -----------------------------
    class Generator(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size

            # Encoder + Decoder GRU
            self.encoder = nn.GRU(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
            self.decoder = nn.GRU(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
            self.fc_out = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, length):
            """
            x: [batch, seq_len, vocab_size]
            """
            # Encode
            _, h = self.encoder(x)
            # Decode autoregressively
            out = []
            decoder_input = torch.zeros(x.size(0), 1, self.vocab_size, device=x.device)
            state = h
            for t in range(length):
                out_t, state = self.decoder(decoder_input, state)
                logits = self.fc_out(out_t)
                wt = F.relu(logits)
                out.append(wt)
                decoder_input = wt.detach()  # feeding generated token
            return torch.cat(out, dim=1)

    # -----------------------------
    # Discriminator
    # -----------------------------
    class Discriminator(nn.Module):
        def __init__(self, vocab_size, hidden_size, n_class):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.n_class = n_class

            self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, n_class)

        def forward(self, x):
            _, h = self.gru(x)
            logits = self.fc(h[-1])
            return torch.softmax(logits, dim=-1)

        def contCmp(self, xw, xe):
            """Continuous similarity (MSE)."""
            return F.mse_loss(xw, xe)

    # -----------------------------
    # Training step methods
    # -----------------------------
    def generate_nr(self, x, length):
        return self.G_NR(x, length)

    def generate_rn(self, x, length):
        return self.G_RN(x, length)

    def discriminate(self, x):
        return self.D(x)

    def loss_generator(self, fake_pred, target):
        """Mean squared loss"""
        return F.mse_loss(fake_pred, target)

    def loss_discriminator(self, real_pred, fake_pred, real_target, fake_target):
        loss_real = F.mse_loss(real_pred, real_target)
        loss_fake = F.mse_loss(fake_pred, fake_target)
        return (loss_real + loss_fake) / 2

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    vocab_size = 20
    hidden_size = 16

    model = GAN(vocab_size=vocab_size, hidden_size=hidden_size, device="cpu")
    optimizer_G = torch.optim.Adam(list(model.G_NR.parameters()) + list(model.G_RN.parameters()), lr=1e-3)
    optimizer_D = torch.optim.Adam(model.D.parameters(), lr=1e-3)

    x_real = torch.rand(batch_size, seq_len, vocab_size)
    y_real = torch.tensor([[1., 0.]] * batch_size)  # real label
    y_fake = torch.tensor([[0., 1.]] * batch_size)  # fake label

    # --- Step 1: Generator NR ---
    x_fake_nr = model.generate_nr(x_real, seq_len)
    pred_fake_nr = model.discriminate(x_fake_nr)
    loss_g_nr = model.loss_generator(pred_fake_nr, y_fake)

    optimizer_G.zero_grad()
    loss_g_nr.backward()
    optimizer_G.step()

    # --- Step 2: Discriminator ---
    pred_real = model.discriminate(x_real)
    loss_d = model.loss_discriminator(pred_real, pred_fake_nr, y_real, y_fake)

    optimizer_D.zero_grad()
    loss_d.backward()
    optimizer_D.step()

    print("loss_g:", loss_g_nr.item(), "loss_d:", loss_d.item())
