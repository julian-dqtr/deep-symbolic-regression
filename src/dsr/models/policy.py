import torch
import torch.nn as nn

from ..core.config import MODEL_CONFIG


class DeepSetsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        points = torch.cat([x, y], dim=-1)
        phi_out = self.phi(points)
        pooled = torch.sum(phi_out, dim=0)
        return self.rho(pooled)


class SymbolicPolicy(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embedding_dim = MODEL_CONFIG["token_embedding_dim"]
        self.hidden_dim = MODEL_CONFIG["hidden_dim"]
        self.dataset_embedding_dim = MODEL_CONFIG["dataset_embedding_dim"]
        self.num_lstm_layers = MODEL_CONFIG["num_lstm_layers"]

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.token_embedding_dim,
        )

        self.sequence_encoder = nn.LSTM(
            input_size=self.token_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )

        self.dataset_encoder = None

        self.state_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.dataset_embedding_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(self.hidden_dim, vocab_size)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.cached_dataset_embedding = None

    def set_dataset_embedding(self, x: torch.Tensor, y: torch.Tensor):
        self.cached_dataset_embedding = self.encode_dataset(x, y).detach()

    def _build_dataset_encoder_if_needed(self, num_features: int):
        if self.dataset_encoder is None:
            self.dataset_encoder = DeepSetsEncoder(
                input_dim=num_features + 1,
                hidden_dim=self.dataset_embedding_dim,
                output_dim=self.dataset_embedding_dim,
            ).to(self.token_embedding.weight.device)

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        is_1d = token_ids.dim() == 1
        
        if token_ids.numel() == 0:
            device = self.token_embedding.weight.device
            if not is_1d and token_ids.dim() == 2:
                B = token_ids.size(0)
                return torch.zeros(B, self.hidden_dim, device=device)
            return torch.zeros(self.hidden_dim, device=device)

        inputs = token_ids.unsqueeze(0) if is_1d else token_ids
        embedded = self.token_embedding(inputs)
        _, (h_n, _) = self.sequence_encoder(embedded)
        
        out = h_n[-1]
        return out[0] if is_1d else out

    def encode_dataset(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        self._build_dataset_encoder_if_needed(num_features=x.shape[1])
        return self.dataset_encoder(x, y)

    def forward(
        self,
        token_ids: torch.Tensor,
        pending_slots,
        length,
        action_mask: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ):
        seq_embedding = self.encode_tokens(token_ids)
        
        if self.cached_dataset_embedding is not None:
            dataset_embedding = self.cached_dataset_embedding
        else:
            dataset_embedding = self.encode_dataset(x, y)

        if isinstance(pending_slots, int) or (isinstance(pending_slots, torch.Tensor) and pending_slots.dim() == 0):
            scalar_features = torch.tensor(
                [float(pending_slots), float(length)],
                dtype=torch.float32,
                device=token_ids.device,
            )
            state_vector = torch.cat(
                [seq_embedding, dataset_embedding, scalar_features],
                dim=0,
            )
        else:
            B = seq_embedding.shape[0]
            dataset_emb_batch = dataset_embedding.unsqueeze(0).expand(B, -1)
            scalar_features = torch.cat([pending_slots.float(), length.float()], dim=-1)
            state_vector = torch.cat([seq_embedding, dataset_emb_batch, scalar_features], dim=-1)

        hidden = self.state_mlp(state_vector)
        logits = self.action_head(hidden)
        value = self.value_head(hidden).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value