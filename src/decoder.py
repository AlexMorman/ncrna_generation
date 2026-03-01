"""decoder.py — GRU decoder with Bahdanau attention.

Autoregressive nucleotide generator that attends over encoder-produced
structural embeddings at each decoding step.

Token vocabulary::

    A = 0,  U = 1,  G = 2,  C = 3   (output classes)
    SOS = 4,  PAD = 5               (input-only special tokens)

Architecture per time-step *t*:
    1. Embed previous token → ``(B, embed_dim)``
    2. Bahdanau attention over encoder outputs → context ``(B, encoder_dim)``
    3. GRU step on ``[embed; context]`` → hidden ``(num_layers, B, hidden_dim)``
    4. Linear projection → logits ``(B, vocab_size)``

Initial hidden state is derived from masked mean-pooling of encoder outputs
passed through a linear projection + tanh.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDecoder(nn.Module):
    """GRU decoder with Bahdanau (additive) attention.

    Args:
        vocab_size:  Number of output nucleotide classes (4: A/U/G/C).
        embed_dim:   Token embedding dimension.
        hidden_dim:  GRU hidden state dimension.
        num_layers:  Number of stacked GRU layers.
        dropout:     Dropout probability.
        encoder_dim: Dimensionality of encoder output embeddings.
    """

    SOS_TOKEN = 4
    PAD_TOKEN = 5

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        encoder_dim: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Token embedding (vocab + SOS + PAD)
        self.embedding = nn.Embedding(
            vocab_size + 2, embed_dim, padding_idx=self.PAD_TOKEN
        )

        # Bahdanau attention components
        self.attn_W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_W_s = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

        # Encoder → initial hidden state
        self.init_hidden_proj = nn.Linear(encoder_dim, hidden_dim * num_layers)

        self.dropout = nn.Dropout(dropout)

    # ── Attention ───────────────────────────────────────────────────────────

    def _compute_attention(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bahdanau (additive) attention.

        Args:
            hidden:          Top-layer GRU hidden, ``(B, hidden_dim)``.
            encoder_outputs: Encoder node embeddings, ``(B, L, encoder_dim)``.
            encoder_mask:    Boolean mask, ``(B, L)`` — ``True`` = valid.

        Returns:
            context:      Weighted sum of encoder outputs, ``(B, encoder_dim)``.
            attn_weights: Attention distribution, ``(B, L)``.
        """
        query = self.attn_W_h(hidden).unsqueeze(1)  # (B, 1, hidden_dim)
        keys = self.attn_W_s(encoder_outputs)  # (B, L, hidden_dim)

        energy = torch.tanh(query + keys)  # (B, L, hidden_dim)
        scores = self.attn_v(energy).squeeze(-1)  # (B, L)

        scores = scores.masked_fill(~encoder_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)  # (B, L)

        # (B, 1, L) @ (B, L, encoder_dim) → (B, 1, encoder_dim) → (B, encoder_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

    # ── Hidden-state initialisation ─────────────────────────────────────────

    def _init_hidden(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Derive initial GRU hidden state from encoder outputs.

        Performs masked mean-pooling followed by a linear projection.

        Args:
            encoder_outputs: ``(B, L, encoder_dim)``
            encoder_mask:    ``(B, L)``

        Returns:
            ``(num_layers, B, hidden_dim)``
        """
        mask_f = encoder_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (encoder_outputs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(
            min=1.0
        )  # (B, encoder_dim)

        h0 = self.init_hidden_proj(pooled)  # (B, hidden_dim * num_layers)
        h0 = h0.view(-1, self.num_layers, self.hidden_dim)  # (B, layers, hidden)
        h0 = h0.permute(1, 0, 2).contiguous()  # (layers, B, hidden)
        return torch.tanh(h0)

    # ── Single step (used by beam search) ───────────────────────────────────

    def step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one autoregressive decoding step.

        Args:
            input_token:     Previous token indices, ``(B,)``.
            hidden:          GRU hidden state, ``(num_layers, B, hidden_dim)``.
            encoder_outputs: ``(B, L, encoder_dim)``
            encoder_mask:    ``(B, L)``

        Returns:
            logits:  ``(B, vocab_size)``
            hidden:  Updated GRU hidden, ``(num_layers, B, hidden_dim)``.
        """
        embedded = self.dropout(self.embedding(input_token))  # (B, embed_dim)

        top_hidden = hidden[-1]  # (B, hidden_dim)
        context, _ = self._compute_attention(
            top_hidden, encoder_outputs, encoder_mask
        )  # (B, encoder_dim)

        gru_input = torch.cat([embedded, context], dim=-1)  # (B, embed+enc)
        gru_input = gru_input.unsqueeze(1)  # (B, 1, embed+enc)

        output, hidden = self.gru(gru_input, hidden)  # output: (B, 1, hidden)
        output = output.squeeze(1)  # (B, hidden_dim)

        logits = self.out_proj(output)  # (B, vocab_size)
        return logits, hidden

    # ── Full-sequence forward (training) ────────────────────────────────────

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Decode a full sequence with optional teacher forcing.

        Args:
            encoder_outputs:       ``(B, L, encoder_dim)``
            targets:               Ground-truth nucleotide indices, ``(B, L)``.
            encoder_mask:          ``(B, L)``
            teacher_forcing_ratio: Probability of feeding ground-truth token
                                   instead of own prediction at each step.

        Returns:
            Logits tensor, ``(B, L, vocab_size)``.
        """
        B, L, _ = encoder_outputs.shape
        device = encoder_outputs.device

        hidden = self._init_hidden(encoder_outputs, encoder_mask)

        # First input is SOS for every sample
        input_token = torch.full(
            (B,), self.SOS_TOKEN, dtype=torch.long, device=device
        )

        all_logits: list[torch.Tensor] = []

        for t in range(L):
            logits, hidden = self.step(
                input_token, hidden, encoder_outputs, encoder_mask
            )
            all_logits.append(logits)

            if random.random() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        return torch.stack(all_logits, dim=1)  # (B, L, vocab_size)
