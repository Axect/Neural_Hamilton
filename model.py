import torch
from torch import nn
import torch.nn.functional as F
import math
from mambapy.mamba import Mamba, MambaConfig


def create_net(sizes):
    net = []
    for i in range(len(sizes) - 1):
        net.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            net.append(nn.GELU())
    return nn.Sequential(*net)


class ICEmbedding(nn.Module):
    """Embeds initial conditions (q0, p0) into a conditioning vector."""

    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, ic):
        # ic: (B, 2) -> (B, d_model)
        return self.net(ic)


@torch.compile
class DeepONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        nodes = hparams["nodes"]
        layers = hparams["layers"]
        branches = hparams["branches"]
        input_size = 100
        output_size = 1

        self.branch_net = create_net(
            [input_size] + [nodes] * (layers - 1) + [2 * branches]
        )
        self.trunk_net = create_net(
            [output_size] + [nodes] * (layers - 1) + [2 * branches]
        )
        self.bias = nn.Parameter(torch.randn(2), requires_grad=True)

        # IC embedding: (q0, p0) -> output offset
        self.ic_embed = nn.Sequential(
            nn.Linear(2, nodes),
            nn.GELU(),
            nn.Linear(nodes, 2),
        )

    def forward(self, u, y, ic):
        """
        - u: (B, 100) - potential function
        - y: (B, W) - time query points
        - ic: (B, 2) - initial conditions (q0, p0)
        """
        B, _ = u.shape
        window = y.shape[1]
        branch_out = self.branch_net(u)  # B x 2p
        branch_out = branch_out.view(B, -1, 2)  # B x p x 2
        trunk_out = torch.stack(
            [self.trunk_net(y[:, i : i + 1]).view(B, -1, 2) for i in range(window)],
            dim=3,
        )
        pred = torch.einsum("bpq,bpqw->bqw", branch_out, trunk_out)
        pred = pred.permute(0, 2, 1)  # B x W x 2
        pred = pred + self.bias

        # Additive IC conditioning
        ic_offset = self.ic_embed(ic).unsqueeze(1)  # (B, 1, 2)
        pred = pred + ic_offset

        return pred[:, :, 0], pred[:, :, 1]


class Encoder(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, x):
        """
        - x: (B, W, 1)
        - h_n: (D * L, B, H) (D = 2 for bidirectional)
        - c_n: (D * L, B, H) (D = 2 for bidirectional)
        """
        _, (h_n, c_n) = self.rnn(x)
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x, h_c):
        """
        - x: (B, W, 1)
        - h_c: (D * L, B, H) (D = 2 for bidirectional)
        - o: (B, W, D * H) (D = 2 for bidirectional)
        - out: (B, W, 1)
        """
        o, _ = self.rnn(x, h_c)
        out = self.fc(o)
        return out


class VaRONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hidden_size = hparams["hidden_size"]
        num_layers = hparams["num_layers"]
        latent_size = hparams["latent_size"]
        dropout = hparams["dropout"]
        kl_weight = hparams["kl_weight"]

        self.branch_net = Encoder(hidden_size, num_layers, dropout)
        self.trunk_x_net = Decoder(hidden_size, num_layers, dropout)
        self.trunk_p_net = Decoder(hidden_size, num_layers, dropout)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        self.fc_z_x = nn.Linear(latent_size, hidden_size)
        self.fc_z_p = nn.Linear(latent_size, hidden_size)
        self.kl_weight = kl_weight
        self.reparametrize = True

        # IC embedding for hidden state conditioning
        self.ic_embed = ICEmbedding(hidden_size)

    def forward(self, u, y, ic):
        """
        - u: (B, 100) - potential function
        - y: (B, W) - time query points
        - ic: (B, 2) - initial conditions (q0, p0)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        (h0, c0) = self.branch_net(u)

        # Reparameterize (VAE)
        mu = self.fc_mu(h0)  # D*L, B, Z
        logvar = self.fc_var(h0)  # D*L, B, Z
        mu = mu.permute(1, 0, 2).contiguous()  # B, D*L, Z
        logvar = logvar.permute(1, 0, 2).contiguous()  # B, D*L, Z
        if self.reparametrize:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # IC embedding
        ic_emb = self.ic_embed(ic).unsqueeze(1)  # (B, 1, H)

        # Decoding with IC conditioning
        hz_x = self.fc_z_x(z) + ic_emb  # B, D * L, H
        hz_p = self.fc_z_p(z) + ic_emb  # B, D * L, H
        hzp_x = hz_x.permute(1, 0, 2).contiguous()  # D * L, B, H
        hzp_p = hz_p.permute(1, 0, 2).contiguous()  # D * L, B, H
        h_c_x = (hzp_x, c0)
        h_c_p = (hzp_p, c0)
        o_x = self.trunk_x_net(y, h_c_x)  # B, W2, 1
        o_p = self.trunk_p_net(y, h_c_p)  # B, W2, 1
        return o_x.squeeze(-1), o_p.squeeze(-1), mu, logvar


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        - x: (B, W, d_model)
        - self.pe: (1, M, d_model)
        - self.pe[:, :x.size(1), :]: (1, W, d_model)
        - output: (B, W, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class TFEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # self.pos_encoder = LearnablePositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, x):
        """
        - x: (B, W1, 1)
        - x (after embedding): (B, W1, d_model)
        - out: (B, W1, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class TFDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.ic_embed = ICEmbedding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers, norm=nn.LayerNorm(d_model)
        )
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x, memory, ic):
        """
        - x: (B, W2, 1)
        - x (after embedding): (B, W2, d_model)
        - memory: (B, W1, d_model)
        - ic: (B, 2) - initial conditions (q0, p0)
        - out: (B, W2, d_model)
        - out (after fc): (B, W2, 2)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Additive IC conditioning: broadcast to all time steps
        ic_emb = self.ic_embed(ic).unsqueeze(1)  # (B, 1, d_model)
        x = x + ic_emb

        out = self.transformer_decoder(x, memory)
        out = self.fc(out)
        return out


@torch.compile
class TraONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        d_model = hparams["d_model"]
        nhead = hparams["nhead"]
        num_layers = hparams["num_layers"]
        dim_feedforward = hparams["dim_feedforward"]
        dropout = hparams["dropout"]

        self.branch_net = TFEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.trunk_net = TFDecoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, u, y, ic):
        """
        - u: (B, W1) - potential function
        - y: (B, W2) - time query points
        - ic: (B, 2) - initial conditions (q0, p0)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        memory = self.branch_net(u)

        # Decoding with IC
        o = self.trunk_net(y, memory, ic)
        return o[:, :, 0], o[:, :, 1]


# ┌──────────────────────────────────────────────────────────┐
#  Fourier Neural Operator (FNO)
# └──────────────────────────────────────────────────────────┘
class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer - learns weights in Fourier space."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep

        # Complex weights for Fourier modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        - x: (B, C_in, W)
        - out: (B, C_out, W)
        """
        B, C, W = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            B, self.out_channels, W // 2 + 1, device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bcm,com->bom", x_ft[:, :, : self.modes], self.weights
        )

        # Inverse FFT
        out = torch.fft.irfft(out_ft, n=W, dim=-1)
        return out


class FourierLayer(nn.Module):
    """Single Fourier layer: spectral conv + pointwise conv + residual."""

    def __init__(self, channels, modes):
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, modes)
        self.pointwise_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        - x: (B, C, W)
        - out: (B, C, W)
        """
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        return F.gelu(x1 + x2)


class FNOEncoder(nn.Module):
    """Fourier Neural Operator encoder for processing potential functions."""

    def __init__(self, d_model, num_layers, modes):
        super().__init__()

        # Lifting layer: 1 -> d_model
        self.lifting = nn.Linear(1, d_model)

        # Fourier layers
        self.layers = nn.ModuleList(
            [FourierLayer(d_model, modes) for _ in range(num_layers)]
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        - x: (B, W, 1)
        - out: (B, W, d_model)
        """
        # Lift to higher dimension
        x = self.lifting(x)  # (B, W, d_model)

        # Apply Fourier layers (need channel-first format)
        x = x.permute(0, 2, 1)  # (B, d_model, W)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)  # (B, W, d_model)

        # Normalize
        x = self.norm(x)
        return x


class FNO(nn.Module):
    """
    Fourier Neural Operator for Hamiltonian mechanics.
    Uses FNO encoder for potential and Transformer decoder for time queries.
    """

    def __init__(self, hparams):
        super().__init__()

        d_model = hparams["d_model"]
        num_layers1 = hparams["num_layers1"]  # FNO encoder layers
        modes = hparams["modes"]  # Number of Fourier modes
        n_head = hparams["n_head"]
        num_layers2 = hparams["num_layers2"]  # Decoder layers
        d_ff = hparams["d_ff"]
        dropout = hparams.get("dropout", 0.0)

        self.encoder = FNOEncoder(d_model, num_layers1, modes)
        self.decoder = TFDecoder(d_model, n_head, num_layers2, d_ff, dropout)

    def forward(self, u, y, ic):
        """
        - u: (B, W1) - potential function values (100 points)
        - y: (B, W2) - time query points
        - ic: (B, 2) - initial conditions (q0, p0)
        - returns: (q, p) each (B, W2)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encode potential with FNO
        memory = self.encoder(u)

        # Decode to (q, p) at time points with IC
        o = self.decoder(y, memory, ic)
        return o[:, :, 0], o[:, :, 1]


# ┌──────────────────────────────────────────────────────────┐
#  Mamba
# └──────────────────────────────────────────────────────────┘
class MambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()

        self.embedding = nn.Linear(1, d_model)
        config = MambaConfig(d_model=d_model, n_layers=num_layers)
        self.mamba = Mamba(config)

    def forward(self, x):
        """
        - x: (B, W, 1)
        - x (after embedding): (B, W, d_model)
        - out: (B, W, d_model)
        """
        x = self.embedding(x)
        out = self.mamba(x)
        return out


class MambONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        d_model = hparams["d_model"]  # hidden_size
        # d_state     = hparams["d_state"]        # SSM state expansion factor
        # d_conv      = hparams["d_conv"]         # Local convolution width
        # expand      = hparams["expand"]         # Block expansion factor
        num_layers1 = hparams["num_layers1"]  # Number of layers (Mamba)
        n_head = hparams["n_head"]  # Number of heads
        num_layers2 = hparams["num_layers2"]  # Number of layers (Decoder)
        d_ff = hparams["d_ff"]  # Feedforward dimension

        self.encoder = MambaEncoder(d_model, num_layers1)
        self.decoder = TFDecoder(d_model, n_head, num_layers2, d_ff, 0.0)

    def forward(self, u, y, ic):
        """
        - u: (B, W1) - potential function
        - y: (B, W2) - time query points
        - ic: (B, 2) - initial conditions (q0, p0)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        memory = self.encoder(u)

        # Decoding with IC
        o = self.decoder(y, memory, ic)
        return o[:, :, 0], o[:, :, 1]
