"""
HAR Pipeline — TGF-NAS + LAHUP Pruning + Quantization
Converted from HAR.ipynb
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import random
import time
import copy
import os
import tempfile
from enum import Enum
from dataclasses import dataclass
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.stats import spearmanr


# ──────────────────────────────────────────────────────────────────────────────
# ENUMS & DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

class LSTMVariant(Enum):
    VANILLA = "vanilla"
    STACKED = "stacked"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class LSTMConfig:
    input_size: int
    hidden_sizes: List[int]
    num_layers: int
    output_size: int
    bidirectional: bool
    variant: LSTMVariant
    input_dropout: int
    optimizer: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    momentum: float
    use_attention: bool
    attention_type: str
    attention_dim: int
    num_attention_heads: int
    attention_dropout: float


# ──────────────────────────────────────────────────────────────────────────────
# ATTENTION MODULES
# ──────────────────────────────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(hidden_size, attention_dim, bias=True)
        self.W_k = nn.Linear(hidden_size, attention_dim, bias=True)
        self.W_v = nn.Linear(attention_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = hidden_states[:, -1:, :]
        keys = hidden_states
        q_proj = self.W_q(query)
        k_proj = self.W_k(keys)
        energy = torch.tanh(q_proj + k_proj)
        attention_scores = self.W_v(energy).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return attended_output


class DotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = hidden_states[:, -1:, :]
        keys = hidden_states
        attention_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return attended_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = math.sqrt(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = hidden_states[:, -1:, :]
        keys = hidden_states
        attention_scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        attention_scores = attention_scores.squeeze(1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended_output = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return attended_output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attention_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        self.W_q = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_k = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_v = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, attention_dim)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        query = hidden_states[:, -1:, :]
        keys = hidden_states
        values = hidden_states
        Q = self.W_q(query)
        K = self.W_k(keys)
        V = self.W_v(values)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.attention_dim)
        output = self.W_o(attended).squeeze(1)
        return output


# ──────────────────────────────────────────────────────────────────────────────
# LSTM MODEL
# ──────────────────────────────────────────────────────────────────────────────

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = self._build_config(config)
        self._build_model()
        self.optimizer = self.create_optimizer()

    def _build_config(self, tune_config):
        hidden_sizes = tune_config["hidden_sizes"].copy()
        num_layers = tune_config["num_layers"]

        if len(hidden_sizes) < num_layers:
            hidden_sizes.extend([hidden_sizes[-1]] * (num_layers - len(hidden_sizes)))
        elif len(hidden_sizes) > num_layers:
            hidden_sizes = hidden_sizes[:num_layers]

        variant = LSTMVariant(tune_config["variant"])
        if variant == LSTMVariant.VANILLA and num_layers > 1:
            variant = LSTMVariant.STACKED
        elif variant == LSTMVariant.STACKED and num_layers == 1:
            variant = LSTMVariant.VANILLA

        bidirectional = tune_config.get('bidirectional', False)
        if bidirectional:
            variant = LSTMVariant.BIDIRECTIONAL

        input_dropout = tune_config["input_dropout"]
        optimizer = tune_config.get("optimizer", "adam")
        learning_rate = tune_config.get("learning_rate", 1e-3)
        batch_size = tune_config.get("batch_size", 32)
        weight_decay = tune_config.get("weight_decay", 1e-4) if optimizer in ["adamw", "sgd"] else 0.0
        momentum = tune_config.get("momentum", 0.9) if optimizer in ["sgd", "rmsprop"] else 0.0

        use_attention = tune_config.get("use_attention", False)
        attention_type = tune_config.get("attention_type", None) if use_attention else None
        attention_dim = tune_config.get("attention_dim", None) if use_attention else None
        num_attention_heads = tune_config.get("num_attention_heads", 1) if use_attention else 1
        attention_dropout = tune_config.get("attention_dropout", 0.0) if use_attention else 0.0

        return LSTMConfig(
            input_size=tune_config["input_size"],
            hidden_sizes=hidden_sizes,
            num_layers=num_layers,
            output_size=tune_config["output_size"],
            bidirectional=bidirectional,
            variant=variant,
            input_dropout=input_dropout,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
            use_attention=use_attention,
            attention_type=attention_type,
            attention_dim=attention_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )

    def _build_model(self):
        config = self.config
        self.input_dropout = nn.Dropout(config.input_dropout)

        if config.variant == LSTMVariant.VANILLA:
            bidir = config.bidirectional
            self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_sizes[0],
                                num_layers=config.num_layers, bidirectional=bidir, batch_first=True)
            lstm_output_size = config.hidden_sizes[0] * (2 if bidir else 1)

        elif config.variant == LSTMVariant.BIDIRECTIONAL:
            self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_sizes[0],
                                num_layers=config.num_layers, bidirectional=True, batch_first=True)
            lstm_output_size = config.hidden_sizes[0] * 2

        else:  # STACKED
            self.lstm_layers = nn.ModuleList()
            for i, hidden_size in enumerate(config.hidden_sizes):
                input_dim = config.input_size if i == 0 else config.hidden_sizes[i - 1]
                self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True))
            lstm_output_size = config.hidden_sizes[-1]
            for lstm_layer in self.lstm_layers:
                for name, param in lstm_layer.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        if config.use_attention:
            self.attention = self._build_attention(lstm_output_size, config)
            attention_output_size = config.attention_dim if config.attention_type == "multi_head" else lstm_output_size
        else:
            self.attention = None
            attention_output_size = lstm_output_size

        self.output_layer = nn.Linear(attention_output_size, config.output_size)

    @property
    def num_layers(self):
        return self.config.num_layers

    def _build_attention(self, lstm_output_size, config):
        if config.attention_type == "additive":
            return AdditiveAttention(hidden_size=lstm_output_size, attention_dim=config.attention_dim,
                                     dropout=config.attention_dropout)
        elif config.attention_type == "dot_product":
            return DotProductAttention(dropout=config.attention_dropout)
        elif config.attention_type == "scaled_dot_product":
            return ScaledDotProductAttention(hidden_size=lstm_output_size, dropout=config.attention_dropout)
        elif config.attention_type == "multi_head":
            return MultiHeadAttention(hidden_size=lstm_output_size, num_heads=config.num_attention_heads,
                                      attention_dim=config.attention_dim, dropout=config.attention_dropout)
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")

    def _collect_gates(self, lstm_layer, x_seq):
        batch_size  = x_seq.size(0)
        seq_len     = x_seq.size(1)
        hidden_size = lstm_layer.hidden_size
        device      = x_seq.device
        dtype       = x_seq.dtype

        h = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)

        W_ih = lstm_layer.weight_ih_l0
        W_hh = lstm_layer.weight_hh_l0
        b_ih = lstm_layer.bias_ih_l0
        b_hh = lstm_layer.bias_hh_l0

        history = {"forget": [], "input": [], "output": [], "cell": []}

        with torch.no_grad():
            for t in range(seq_len):
                x_t = x_seq[:, t, :].detach().clone()
                raw = x_t @ W_ih.t() + b_ih + h @ W_hh.t() + b_hh
                i_g, f_g, g_g, o_g = raw.chunk(4, dim=1)
                i_g = torch.sigmoid(i_g)
                f_g = torch.sigmoid(f_g)
                g_g = torch.tanh(g_g)
                o_g = torch.sigmoid(o_g)
                c   = f_g * c + i_g * g_g
                h   = o_g * torch.tanh(c)

                history["forget"].append(f_g.detach().cpu())
                history["input"].append(i_g.detach().cpu())
                history["output"].append(o_g.detach().cpu())
                history["cell"].append(c.detach().cpu())

        return history

    def forward(self, x, return_gates=False):
        x = self.input_dropout(x)
        gates_dict = {}

        if self.config.variant == LSTMVariant.STACKED:
            for layer_idx, lstm_layer in enumerate(self.lstm_layers):
                if return_gates:
                    history = self._collect_gates(lstm_layer, x)
                    gates_dict[layer_idx] = [history]

                batch_size  = x.size(0)
                hidden_size = lstm_layer.hidden_size
                device      = x.device
                h_0 = torch.randn(1, batch_size, hidden_size, device=device, dtype=x.dtype) * 0.01
                c_0 = torch.randn(1, batch_size, hidden_size, device=device, dtype=x.dtype) * 0.01
                x, _ = lstm_layer(x, (h_0, c_0))

        else:
            batch_size     = x.size(0)
            hidden_size    = self.lstm.hidden_size
            device         = x.device
            num_directions = 2 if self.config.variant == LSTMVariant.BIDIRECTIONAL else 1

            if return_gates:
                history = self._collect_gates(self.lstm, x)
                gates_dict[0] = [history]

            h_0 = torch.zeros(self.config.num_layers * num_directions, batch_size, hidden_size,
                              device=device, dtype=x.dtype)
            c_0 = torch.zeros(self.config.num_layers * num_directions, batch_size, hidden_size,
                              device=device, dtype=x.dtype)
            x, _ = self.lstm(x, (h_0, c_0))

        if self.config.use_attention:
            x = self.attention(x)
        else:
            x = x[:, -1, :]

        expected = self.output_layer.in_features
        got      = x.shape[-1]
        if got != expected:
            raise RuntimeError(
                f"[SHAPE BUG] output_layer expects {expected} features but got {got}. "
                f"variant={self.config.variant}, hidden_sizes={self.config.hidden_sizes}, "
                f"num_layers={self.config.num_layers}, use_attention={self.config.use_attention}, "
                f"attention_type={self.config.attention_type}, x.shape={tuple(x.shape)}"
            )

        output = self.output_layer(x)

        if return_gates:
            return output, gates_dict
        return output

    def create_optimizer(self):
        config = self.config
        if config.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "adamw":
            return optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=config.learning_rate,
                             momentum=config.momentum, weight_decay=config.weight_decay)
        elif config.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            return optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class HARDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.prepare_data()

    def prepare_data(self):
        # ── detect label column ──────────────────────────────────────────────
        label_col = None
        for candidate in ['Activity', 'activity', 'label', 'Label']:
            if candidate in self.data.columns:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError(f"No label column found. Columns: {list(self.data.columns)}")

        le = LabelEncoder()
        self.data['activity_encoded'] = le.fit_transform(self.data[label_col])
        self.num_classes = len(le.classes_)
        self.label_encoder = le

        # ── drop ALL non-feature columns (label + id cols + non-numeric) ────
        drop_cols = set(['subject', 'Subject', 'subject_id',
                         'Activity', 'activity', 'label', 'Label', 'activity_encoded'])
        feature_columns = []
        for col in self.data.columns:
            if col in drop_cols:
                continue
            # drop columns that can't be cast to float (e.g. stray string cols)
            try:
                self.data[col].astype(np.float32)
                feature_columns.append(col)
            except (ValueError, TypeError):
                print(f"[HARDataset] Dropping non-numeric column: '{col}'")

        if not feature_columns:
            raise ValueError("No numeric feature columns found after dropping label/id columns.")

        self.sequences = self.data[feature_columns].values.astype(np.float32)
        self.labels = self.data['activity_encoded'].values
        self.sequences = np.expand_dims(self.sequences, axis=1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])[0]


# ──────────────────────────────────────────────────────────────────────────────
# GRADIENT HOOKS
# ──────────────────────────────────────────────────────────────────────────────

def install_gradient_hooks(model, G_magnitude_history, G_sign_history):
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            def make_hook(param_id):
                def hook_function(grad):
                    if grad is None:
                        return
                    grad_flat = grad.flatten()
                    magnitude = torch.sqrt(torch.sum(grad_flat ** 2)).item()
                    if magnitude == 0.0:
                        magnitude = random.uniform(1e-6, 10 * 1e-6)
                    signs = torch.sign(grad_flat)
                    sign_fraction = torch.mean(signs).item()
                    if sign_fraction == 0:
                        sign_fraction = random.uniform(-0.001, 0.001)
                    if param_id not in G_magnitude_history:
                        G_magnitude_history[param_id] = []
                    if param_id not in G_sign_history:
                        G_sign_history[param_id] = []
                    G_magnitude_history[param_id].append(magnitude)
                    G_sign_history[param_id].append(sign_fraction)
                return hook_function
            param.register_hook(make_hook(param_name))


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model_on_subset(model, test_csv_path, sequence_length,
                             subset_ratio=0.1, label_encoder=None):
    try:
        test_dataset = HARDataset(test_csv_path, sequence_length=sequence_length)
        # Re-encode using shared encoder if provided (avoids class index mismatch)
        if label_encoder is not None:
            for col in ['Activity', 'activity', 'label', 'Label']:
                if col in test_dataset.data.columns:
                    test_dataset.labels = label_encoder.transform(
                        test_dataset.data[col].values
                    )
                    break
        total_size = len(test_dataset)
        subset_size = max(1, int(total_size * subset_ratio))
        indices = torch.randperm(total_size)[:subset_size]
        subset = torch.utils.data.Subset(test_dataset, indices)
        batch_size = model.config.batch_size
        test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        model.eval()
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.float()
                target = target.long()
                output = model(data)
                predictions = torch.argmax(output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        return accuracy_score(all_targets, all_predictions)
    except Exception as e:
        print(f"    Error during test evaluation: {e}")
        return 0.0


def train_probe_epoch(model, optimizer, train_csv_path, test_csv_path,
                      model_index=None,
                      sequence_length=1, num_epochs=15, batches_per_epoch=70):
    epoch_accuracies = []
    try:
        dataset = HARDataset(train_csv_path, sequence_length=sequence_length)

        # Validate num_classes vs model output_size before training starts
        data_classes  = dataset.num_classes
        model_outputs = model.config.output_size
        if data_classes != model_outputs:
            raise ValueError(
                f"CSV has {data_classes} classes but model output_size={model_outputs}. "
                f"Use output_size={data_classes} in run_pipeline()."
            )

        batch_size = model.config.batch_size
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(num_epochs):
            if model_index is not None:
                print(f"[Model {model_index}] EPOCH {epoch + 1}/{num_epochs}")
            else:
                print(f"EPOCH {epoch + 1}/{num_epochs}")            
            epoch_loss = 0
            batch_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= batches_per_epoch:
                    break
                data = data.float()
                target = target.long()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            epoch_accuracy = evaluate_model_on_subset(
                model, test_csv_path, sequence_length,
                label_encoder=dataset.label_encoder
            )
            epoch_accuracies.append(epoch_accuracy)
            model.train()

        proxy_accuracy = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
        return proxy_accuracy, epoch_accuracies

    except Exception as e:
        import traceback
        print(f"[ERROR] Probe training failed: {e}")
        traceback.print_exc()
        return 0.0, []


# ──────────────────────────────────────────────────────────────────────────────
# GRADIENT METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_magnitude(G_magnitude_history):
    def geometric_mean(values):
        if not values:
            return 0.0
        epsilon = 1e-10
        filtered_values = [max(v, epsilon) for v in values if v > 0]
        if not filtered_values:
            return epsilon
        log_sum = sum(math.log(v) for v in filtered_values)
        return math.exp(log_sum / len(filtered_values))

    param_magnitudes = {}
    valid_magnitudes = []
    for param_id, norms_list in G_magnitude_history.items():
        magnitude = geometric_mean(norms_list)
        param_magnitudes[param_id] = magnitude
        valid_magnitudes.append(magnitude)

    arch_magnitude = geometric_mean(valid_magnitudes) if valid_magnitudes else 0.0
    return arch_magnitude, param_magnitudes


def compute_consistency(G_sign_history):
    param_consistencies = {}
    valid_consistencies = []

    for param_id, G_signs in G_sign_history.items():
        if len(G_signs) <= 1:
            param_consistencies[param_id] = 1.0
            valid_consistencies.append(1.0)
            continue

        discrete_signs = []
        for sign_fraction in G_signs:
            if sign_fraction > 0.1:
                discrete_signs.append(+1)
            elif sign_fraction < -0.1:
                discrete_signs.append(-1)
            else:
                discrete_signs.append(0)

        direction_changes = sum(1 for i in range(1, len(discrete_signs))
                                if discrete_signs[i] != discrete_signs[i - 1])
        max_possible_changes = len(discrete_signs) - 1
        change_rate = direction_changes / max_possible_changes
        consistency_score = 1.0 - change_rate
        param_consistencies[param_id] = consistency_score
        valid_consistencies.append(consistency_score)

    arch_consistency = (sum(valid_consistencies) / len(valid_consistencies)
                        if valid_consistencies else 1.0)
    return arch_consistency, param_consistencies


# ──────────────────────────────────────────────────────────────────────────────
# SCORING & SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def standardize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / (std + 1e-8)


def calculate_alpha_beta(magnitude_standardized, consistency_standardized, proxy_accuracy_array):
    corr_mag, _ = spearmanr(magnitude_standardized, proxy_accuracy_array)
    corr_cons, _ = spearmanr(consistency_standardized, proxy_accuracy_array)

    sign_mag = 1 if corr_mag > 0 else -1
    sign_cons = 1 if corr_cons > 0 else -1
    weight_mag = abs(corr_mag)
    weight_cons = abs(corr_cons)
    total = weight_mag + weight_cons

    if total < 0.01:
        alpha, beta = 1.0, 0.0
    else:
        alpha = weight_mag / total
        beta = weight_cons / total

    return alpha, beta, {"sign_mag": sign_mag, "sign_cons": sign_cons}


def full_training(model, train_csv, test_csv, sequence_length=1, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}\n")

    train_dataset = HARDataset(train_csv, sequence_length)
    test_dataset  = HARDataset(test_csv,  sequence_length)

    # Validate: num_classes in data must match model output size
    data_classes  = train_dataset.num_classes
    model_outputs = model.config.output_size
    if data_classes != model_outputs:
        raise ValueError(
            f"[full_training] CSV has {data_classes} unique classes but model "
            f"output_size={model_outputs}. Pass output_size={data_classes} when "
            f"calling run_pipeline()."
        )

    # Re-encode test labels using the TRAIN encoder so class indices always match.
    # HARDataset fits its own LabelEncoder independently; if class ordering differs
    # (e.g. test has labels 1-6 while train has 0-5) we get out-of-bounds errors.
    for col in ['Activity', 'activity', 'label', 'Label']:
        if col in test_dataset.data.columns:
            test_dataset.labels = train_dataset.label_encoder.transform(
                test_dataset.data[col].values
            )
            break

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = model.create_optimizer()

    print(f"Training for {num_epochs} epochs...\n")
    test_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_correct, train_total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, preds = output.max(1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        train_acc = 100 * train_correct / train_total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = output.max(1)
                test_correct += (preds == target).sum().item()
                test_total += target.size(0)

        test_acc = 100 * test_correct / test_total
        print(f"Epoch {epoch:2d}/{num_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), "best_lstm.pth")
    print("Model saved as 'best_lstm.pth'")
    return model, test_acc


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SIZE UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def model_size_mb(model, dtype_bytes=4):
    """Compute model size in MB based on non-zero parameters."""
    non_zero = sum((p != 0).sum().item() for p in model.parameters())
    size_mb = non_zero * dtype_bytes / (1024 ** 2)
    return size_mb


def _model_size_mb(model):
    size = 0
    for t in list(model.parameters()) + list(model.buffers()):
        bpe = {torch.int8: 1, torch.float16: 2,
               torch.float32: 4, torch.float64: 8}.get(t.dtype, 4)
        size += t.numel() * bpe
    return size / (1024 ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# TGF-NAS
# ──────────────────────────────────────────────────────────────────────────────

def TGF_NAS(
    train_path: str,
    test_path: str,
    input_size: int,
    output_size: int,
    use_embedding: bool,
    num_architectures: int,
    sparsity: float,
    quant_types: list,
    sequence_length: int = 1,
    num_probe_epochs: int = 15,
    batches_per_epoch: int = 70,
    full_train_epochs: int = 30,
    tgf_percentile: float = 70.0,
) -> dict:
    """
    TGF-NAS end-to-end pipeline.

    Steps:
    1. Architecture sampling via Ray Tune random search.
    2. Probe training + gradient-based scoring (magnitude & consistency).
    3. TGF-NAS score computation and architecture filtering.
    4. Full training of the best architecture.
    """

    hidden_size_options = [
        [64], [128], [256], [512],
        [128, 64], [256, 128], [512, 256],
        [256, 128, 64], [512, 256, 128],
        [64, 64], [256, 256], [512, 512],
    ]

    search_space = {
        "input_size":  tune.choice([input_size]),
        "output_size": tune.choice([output_size]),
        "hidden_sizes": tune.sample_from(
            lambda config: random.choice(
                [hs for hs in hidden_size_options if len(hs) == config["num_layers"]]
            )
        ),
        "variant": tune.sample_from(
            lambda config: (
                "vanilla" if config["num_layers"] == 1
                else random.choice(["stacked", "bidirectional"])
            )
        ),
        "bidirectional": tune.sample_from(
            lambda config: config["variant"] == "bidirectional"
        ),
        "num_layers": tune.choice([1, 2, 3]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "input_dropout": tune.uniform(0.0, 0.5),
        "optimizer": tune.choice(["adam", "adamw", "sgd", "rmsprop"]),
        "weight_decay": tune.sample_from(
            lambda config: (
                random.uniform(1e-6, 1e-2) if config["optimizer"] in ["adamw", "sgd"] else 0.0
            )
        ),
        "momentum": tune.sample_from(
            lambda config: (
                random.uniform(0.8, 0.99) if config["optimizer"] in ["sgd", "rmsprop"] else None
            )
        ),
        "use_attention": tune.choice([True, False]),
        "attention_type": tune.sample_from(
            lambda config: (
                random.choice(["additive", "dot_product", "scaled_dot_product", "multi_head"])
                if config["use_attention"] else None
            )
        ),
        "attention_dim": tune.sample_from(
            lambda config: (
                random.choice([64, 128, 256]) if config["use_attention"] else None
            )
        ),
        "num_attention_heads": tune.sample_from(
            lambda config: (
                random.choice([2, 4, 8])
                if config["use_attention"] and config["attention_type"] == "multi_head" else 1
            )
        ),
        "attention_dropout": tune.sample_from(
            lambda config: (
                random.uniform(0.0, 0.3) if config["use_attention"] else 0.0
            )
        ),
    }

    sampled_models = []

    def sample_architecture(config):
        model = LSTM(config)
        param_count = sum(p.numel() for p in model.parameters())
        tune.report({"param_count": param_count})

    ray.init(ignore_reinit_error=True)
    print(f"Sampling {num_architectures} architectures via Ray Tune...")

    tuner = tune.Tuner(
        sample_architecture,
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(),
            num_samples=num_architectures,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    for result in results:
        model = LSTM(result.config)
        sampled_models.append(model)

    ray.shutdown()
    print(f"Sampled {len(sampled_models)} architectures.\n")

    # Probe training + gradient scoring
    evaluation_history = {}

    for i, model in enumerate(sampled_models):
        print(f"\n{'='*60}")
        print(f"Evaluating Model {i} (Index {i})")
        print(f"{'='*60}")

        G_magnitude_history = {}
        G_sign_history = {}

        optimizer_inst = model.create_optimizer()
        install_gradient_hooks(model, G_magnitude_history, G_sign_history)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {num_params:,}")

        start_time = time.time()
        proxy_acc, epoch_accs = train_probe_epoch(
            model,
            optimizer_inst,
            train_path,
            test_path,
            model_index=i,
            sequence_length=sequence_length,
            num_epochs=num_probe_epochs,
            batches_per_epoch=batches_per_epoch,
        )
        training_time = time.time() - start_time

        arch_mag, _ = compute_magnitude(G_magnitude_history)
        arch_cons, _ = compute_consistency(G_sign_history)

        print(f"Training Time:           {training_time:.2f}s")
        print(f"Architecture Magnitude:  {arch_mag:.6f}")
        print(f"Architecture Consistency:{arch_cons:.6f}")
        print(f"Proxy Accuracy:          {proxy_acc:.4f}")

        evaluation_history[i] = {
            "index": i,
            "model": model,
            "magnitude_score": arch_mag,
            "consistency_score": arch_cons,
            "proxy_accuracy": proxy_acc,
            "epoch_accuracies": epoch_accs,
            "num_params": num_params,
            "training_time": training_time,
        }

    # TGF-NAS scoring
    magnitude_array      = np.array([evaluation_history[i]["magnitude_score"]  for i in range(len(evaluation_history))])
    consistency_array    = np.array([evaluation_history[i]["consistency_score"] for i in range(len(evaluation_history))])
    proxy_accuracy_array = np.array([evaluation_history[i]["proxy_accuracy"]    for i in range(len(evaluation_history))])

    magnitude_std    = standardize(magnitude_array)
    consistency_std  = standardize(consistency_array)

    alpha, beta, signs = calculate_alpha_beta(magnitude_std, consistency_std, proxy_accuracy_array)
    print(f"\nFinal weights: α={alpha:.3f}, β={beta:.3f}")
    print(f"Sign corrections: magnitude={signs['sign_mag']}, consistency={signs['sign_cons']}")

    tgfnas_scores = (
        alpha * signs["sign_mag"]  * magnitude_std +
        beta  * signs["sign_cons"] * consistency_std
    )

    # If spearmanr returned nan (e.g. all proxy accuracies identical / all zero),
    # tgfnas_scores will be all-nan. Fall back to proxy accuracy as the score.
    if not np.all(np.isfinite(tgfnas_scores)):
        print("[WARNING] TGF-NAS scores contain NaN (likely all proxy accuracies are identical). "
              "Falling back to proxy accuracy as ranking score.")
        tgfnas_scores = proxy_accuracy_array.copy()

    print(f"\n{'='*80}")
    print(f"{'Arch':<6} {'Magnitude':<12} {'Consistency':<12} {'Accuracy':<10} {'TGFNAS':<10}")
    print(f"{'='*80}")
    for i in range(len(magnitude_array)):
        print(f"{i:<6} {magnitude_array[i]:<12.6f} {consistency_array[i]:<12.4f} "
              f"{proxy_accuracy_array[i]:<10.4f} {tgfnas_scores[i]:<10.4f}")
    print(f"{'='*80}")

    threshold = np.percentile(tgfnas_scores, tgf_percentile)
    filtered_indices = np.where(tgfnas_scores >= threshold)[0]

    filtered_accuracies = proxy_accuracy_array[filtered_indices]
    sorted_idx = np.argsort(filtered_accuracies)[::-1]
    ranked_indices = filtered_indices[sorted_idx]
    best_idx = int(ranked_indices[0])

    print(f"\nBest Architecture Index: {best_idx}")
    print(f"   TGF-NAS Score: {tgfnas_scores[best_idx]:.4f}")
    print(f"   Magnitude:     {magnitude_array[best_idx]:.6f}")
    print(f"   Consistency:   {consistency_array[best_idx]:.4f}")
    print(f"   Proxy Accuracy:{proxy_accuracy_array[best_idx]:.4f}")

    # Full training of best architecture
    best_model = evaluation_history[best_idx]["model"]
    best_model, final_acc = full_training(
        best_model, train_path, test_path,
        sequence_length=sequence_length,
        num_epochs=full_train_epochs,
    )

    return {
        "accuracy": final_acc,
        "sparsity": sparsity,
        "quantization": quant_types,
        "best_model": best_model,
        "evaluation_history": evaluation_history,
        "tgfnas_scores": tgfnas_scores,
        "best_architecture_index": best_idx
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAHUP PRUNING
# ──────────────────────────────────────────────────────────────────────────────

def _build_loaders(train_path, test_path, batch_size, label_col='Activity'):
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df  = pd.read_csv(test_path,  low_memory=False)

    # Auto-detect label column
    for col in [label_col, 'Activity', 'activity', 'label', 'Label']:
        if col in train_df.columns:
            label_col = col
            break

    # Drop all non-feature columns
    id_cols  = {'subject', 'Subject', 'subject_id', 'Activity', 'activity', 'label', 'Label'}
    drop_cols = [c for c in train_df.columns if c in id_cols]

    # Also drop any column that can't be cast to float32
    feature_cols = []
    for c in train_df.columns:
        if c in id_cols:
            continue
        try:
            train_df[c].astype(np.float32)
            feature_cols.append(c)
        except (ValueError, TypeError):
            pass

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df[label_col].values

    # Fit encoder on TRAIN only, transform both (avoids index mismatch)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    X_train = torch.tensor(X_train).unsqueeze(1)
    X_test  = torch.tensor(X_test).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    print(f"  Train samples : {len(X_train)} | Test samples: {len(X_test)}")
    print(f"  Input shape   : {X_train.shape[1:]} | Classes: {len(le.classes_)}")
    return train_loader, test_loader


def _evaluate_model(model, test_loader, device, label=""):
    model.eval()

    has_quantized = any(p.dtype == torch.qint8  for p in model.parameters())
    has_fp16      = any(p.dtype == torch.float16 for p in model.parameters())

    if has_quantized:
        eval_device = torch.device("cpu")
        input_dtype = torch.float32
    elif has_fp16:
        eval_device = device
        input_dtype = torch.float16
    else:
        eval_device = device
        input_dtype = torch.float32

    model = model.to(eval_device)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs  = inputs.to(eval_device).to(input_dtype)
            targets = targets.to(eval_device)
            try:
                outputs = model(inputs)
            except RuntimeError as e:
                if "dtype" in str(e).lower():
                    model   = model.float()
                    inputs  = inputs.float()
                    outputs = model(inputs)
                else:
                    raise e
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    tag = f"[{label}] " if label else ""
    print(f"   {tag}Accuracy: {accuracy:.2f}%")
    return accuracy


class _LAHUPPruner:
    def __init__(self, model, device):
        self.model  = model.to(device)
        self.device = device
        self.stats  = {}
        self.masks  = {}

    def collect_statistics(self, dataloader, criterion):
        print("  [LAHUP] Collecting gate / gradient statistics from train_loader...")
        self.model.train()

        gate_keys  = None
        cell_all   = {}
        gates_all  = {}
        grads_all  = {}

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()

            outputs, gates_dict = self.model(inputs, return_gates=True)

            if gate_keys is None:
                gate_keys = sorted(gates_dict.keys())
                cell_all  = {k: [] for k in gate_keys}
                gates_all = {k: {'forget': [], 'input': [], 'output': []} for k in gate_keys}

            for layer_idx in gate_keys:
                layer_history = gates_dict[layer_idx][0]
                cell_all[layer_idx].append(torch.stack(layer_history['cell']).cpu())
                for g in ['forget', 'input', 'output']:
                    gates_all[layer_idx][g].append(torch.stack(layer_history[g]).cpu())

            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grads_all.setdefault(name, []).append(torch.abs(param.grad.data).cpu())

        for l in gate_keys:
            all_cells = torch.cat(cell_all[l], dim=1)
            self.stats.setdefault('var_cell', {})[l] = torch.std(all_cells, dim=(0, 1))
            self.stats.setdefault('avg_gates', {})[l] = {}
            for g in ['forget', 'input', 'output']:
                all_g = torch.cat(gates_all[l][g], dim=1)
                self.stats['avg_gates'][l][g] = torch.mean(all_g, dim=(0, 1))

        self.stats['gate_keys'] = gate_keys

        self.stats['avg_grad'] = {
            name: torch.stack(g).mean(0) for name, g in grads_all.items()
        }
        all_grads_flat = torch.cat([v.flatten() for v in self.stats['avg_grad'].values()])
        self.stats['min_grad'] = all_grads_flat.min()
        self.stats['max_grad'] = all_grads_flat.max()

    def apply_lahup(self, dataloader, criterion, pruning_ratio,
                    alpha=0.4, beta=0.3, gamma=0.3, lambda1=0.5, lambda2=0.5):
        self.collect_statistics(dataloader, criterion)
        eps = 1e-8

        gate_keys = self.stats.get('gate_keys', list(range(self.model.num_layers)))
        R_neuron = {}
        for l in gate_keys:
            v      = self.stats['var_cell'][l]
            norm_v = (v - v.min()) / (v.max() - v.min() + eps)
            f = self.stats['avg_gates'][l]['forget']
            i = self.stats['avg_gates'][l]['input']
            o = self.stats['avg_gates'][l]['output']
            G_sat = torch.max(
                torch.stack([f*(1-i), (1-f)*(1-o), f*(1-o)]), dim=0
            )[0]
            R_neuron[l] = lambda1 * (1 - norm_v) + lambda2 * G_sat

        total_p, pruned_p = 0, 0
        for name, param in self.model.named_parameters():
            if 'weight' in name and any(x in name for x in ['weight_ih', 'weight_hh']):
                parts = name.split('.')
                try:
                    l_idx = int(parts[1])
                except ValueError:
                    l_idx = 0
                R = R_neuron[l_idx]

                w_norm = (
                    param.data.abs().cpu() - param.data.abs().min().cpu()
                ) / (param.data.abs().max().cpu() - param.data.abs().min().cpu() + eps)

                g_norm = (
                    self.stats['avg_grad'][name] - self.stats['min_grad']
                ) / (self.stats['max_grad'] - self.stats['min_grad'] + eps)

                R_full = torch.cat([R, R, R, R]).view(-1, 1).expand_as(w_norm)
                P      = alpha * R_full + beta * (1 - w_norm) + gamma * (1 - g_norm)
                thresh = torch.quantile(P.flatten(), pruning_ratio)
                mask   = (P > thresh).to(self.device)

                param.data *= mask.float()
                self.masks[name] = mask.float()

                total_p  += param.numel()
                pruned_p += (mask == 0).sum().item()

        achieved = 100 * pruned_p / total_p
        print(f"  [LAHUP] Done. Achieved sparsity: {achieved:.2f}%")
        return achieved


def _finetune(model, train_loader, criterion, optimizer, masks, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name])

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"    Epoch {epoch+1}/{epochs}: "
              f"Loss={total_loss/len(train_loader):.4f}, "
              f"Acc={100.*correct/total:.2f}%")
    return model


def apply_pruning(
    model,
    sparsity: float,
    train_path: str   = "train.csv",
    test_path:  str   = "test.csv",
    label_col:  str   = "Activity",
    batch_size: int   = 64,
    alpha:   float    = 0.4,
    beta:    float    = 0.3,
    gamma:   float    = 0.3,
    lambda1: float    = 0.5,
    lambda2: float    = 0.5,
    finetune:           bool  = True,
    finetune_epochs:    int   = 15,
    finetune_lr_factor: float = 0.1,
    device:    str    = None,
    save_path: str    = "har_pruned.pth",
) -> torch.nn.Module:
    """Apply LAHUP structured pruning to a trained model."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if sparsity > 1.0:
        sparsity = sparsity / 100.0

    print(f"\n{'='*60}")
    print("APPLY LAHUP PRUNING")
    print(f"{'='*60}")
    print(f"  Target sparsity : {sparsity * 100:.1f}%")
    print(f"  Device          : {device}")

    print("\n  Loading data from CSVs...")
    train_loader, test_loader = _build_loaders(train_path, test_path, batch_size, label_col)
    criterion = nn.CrossEntropyLoss()

    print("\n  Baseline accuracy (before pruning):")
    baseline_acc = _evaluate_model(model, test_loader, device, label="Baseline")
    print("Model Size before pruning : ", model_size_mb(model))

    pruner = _LAHUPPruner(model, device)
    achieved_sparsity = pruner.apply_lahup(
        dataloader=train_loader,
        criterion=criterion,
        pruning_ratio=sparsity,
        alpha=alpha, beta=beta, gamma=gamma,
        lambda1=lambda1, lambda2=lambda2,
    )
    model.pruner_masks = pruner.masks

    print("\n  Accuracy after pruning (before fine-tuning):")
    pruned_acc_before = _evaluate_model(model, test_loader, device, label="Pruned (before FT)")

    if finetune:
        print(f"\n  Fine-tuning for {finetune_epochs} epochs (lr x {finetune_lr_factor})...")
        base_lr      = model.config.learning_rate if hasattr(model, 'config') else 1e-3
        weight_decay = model.config.weight_decay  if hasattr(model, 'config') else 1e-4
        ft_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr * finetune_lr_factor,
            weight_decay=weight_decay,
        )
        model = _finetune(
            model, train_loader, criterion, ft_optimizer,
            masks=pruner.masks, device=device, epochs=finetune_epochs,
        )

        print("\n  Accuracy after fine-tuning:")
        pruned_acc_after = _evaluate_model(model, test_loader, device, label="Pruned (after FT)")
    else:
        pruned_acc_after = pruned_acc_before

    torch.save(model.state_dict(), save_path)

    print(f"\n{'='*60}")
    print("PRUNING SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline accuracy          : {baseline_acc:.2f}%")
    print(f"  Accuracy after pruning     : {pruned_acc_before:.2f}%  "
          f"(drop: {baseline_acc - pruned_acc_before:.2f}%)")
    if finetune:
        print(f"  Accuracy after fine-tuning : {pruned_acc_after:.2f}%  "
              f"(recovery: {pruned_acc_after - pruned_acc_before:.2f}%)")
    print(f"  Achieved sparsity          : {achieved_sparsity:.2f}%")
    print(f"Model Size After pruning : ", model_size_mb(model))
    print(f"  Model saved to             : {save_path}")
    print(f"{'='*60}\n")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# QUANTIZATION
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate(model, test_loader):
    """Shorthand wrapper – returns accuracy (float) using CPU device."""
    return _evaluate_model(model, test_loader, torch.device("cpu"))


def _build_test_loader(test_path, batch_size=64, label_col='Activity'):
    df = pd.read_csv(test_path, low_memory=False)

    for col in [label_col, 'Activity', 'activity', 'label', 'Label']:
        if col in df.columns:
            label_col = col
            break

    id_cols = {'subject', 'Subject', 'subject_id', 'Activity', 'activity', 'label', 'Label'}
    feature_cols = []
    for c in df.columns:
        if c in id_cols:
            continue
        try:
            df[c].astype(np.float32)
            feature_cols.append(c)
        except (ValueError, TypeError):
            pass

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    le = LabelEncoder()
    y  = le.fit_transform(y)

    X = torch.tensor(X).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _evaluate(model, test_loader):
    model.eval()
    model = model.cpu()

    first_param = next(model.parameters())
    is_fp16 = (first_param.dtype == torch.float16)

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs  = inputs.cpu()
            targets = targets.cpu()
            inputs  = inputs.half() if is_fp16 else inputs.float()

            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                output = output[:, -1, :]
            output = output.float()

            _, predicted = output.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def _inference_speed(model, test_loader, warmup=5, runs=20):
    model.eval()
    model = model.cpu()

    is_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
    inputs, _ = next(iter(test_loader))
    inputs = inputs.cpu()
    inputs = inputs.half() if is_fp16 else inputs.float()

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(inputs)
    return (time.time() - start) / runs * 1000


class _INT8LSTMCell(nn.Module):
    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
        super().__init__()
        self.hidden_size = weight_hh.shape[0] // 4

        scale_ih = weight_ih.abs().max().item() / 127.0 + 1e-8
        self.register_buffer('weight_ih_int8',
            torch.clamp(torch.round(weight_ih.float() / scale_ih), -128, 127).to(torch.int8))
        self.weight_ih_scale = scale_ih

        scale_hh = weight_hh.abs().max().item() / 127.0 + 1e-8
        self.register_buffer('weight_hh_int8',
            torch.clamp(torch.round(weight_hh.float() / scale_hh), -128, 127).to(torch.int8))
        self.weight_hh_scale = scale_hh

        self.register_buffer('bias_ih', bias_ih.float().clone())
        self.register_buffer('bias_hh', bias_hh.float().clone())

    def forward(self, x, h_prev, c_prev):
        w_ih = self.weight_ih_int8.float() * self.weight_ih_scale
        w_hh = self.weight_hh_int8.float() * self.weight_hh_scale

        gates = (x @ w_ih.t() + self.bias_ih +
                 h_prev @ w_hh.t() + self.bias_hh)

        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)

        c_new = f_g * c_prev + i_g * g_g
        h_new = o_g * torch.tanh(c_new)
        return h_new, c_new


class _INT8NNLSTMWrapper(nn.Module):
    def __init__(self, original_lstm: nn.LSTM):
        super().__init__()
        self.hidden_size   = original_lstm.hidden_size
        self.num_layers    = original_lstm.num_layers
        self.bidirectional = original_lstm.bidirectional
        self.batch_first   = original_lstm.batch_first
        self.num_dirs      = 2 if self.bidirectional else 1

        self.cells = nn.ModuleList()
        for layer in range(self.num_layers):
            for direction in range(self.num_dirs):
                suffix = '' if direction == 0 else '_reverse'
                w_ih = getattr(original_lstm, f'weight_ih_l{layer}{suffix}').data
                w_hh = getattr(original_lstm, f'weight_hh_l{layer}{suffix}').data
                b_ih = getattr(original_lstm, f'bias_ih_l{layer}{suffix}').data
                b_hh = getattr(original_lstm, f'bias_hh_l{layer}{suffix}').data
                self.cells.append(_INT8LSTMCell(w_ih, w_hh, b_ih, b_hh))

    def forward(self, x, hx=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype  = torch.float32

        if hx is None:
            h = torch.zeros(self.num_layers * self.num_dirs, batch_size,
                            self.hidden_size, device=device, dtype=dtype)
            c = torch.zeros(self.num_layers * self.num_dirs, batch_size,
                            self.hidden_size, device=device, dtype=dtype)
        else:
            h, c = hx[0].float(), hx[1].float()

        layer_input = x.float()
        for layer in range(self.num_layers):
            fwd_cell = self.cells[layer * self.num_dirs]
            h_f = h[layer * self.num_dirs]
            c_f = c[layer * self.num_dirs]

            fwd_outputs = []
            for t in range(seq_len):
                h_f, c_f = fwd_cell(layer_input[:, t, :], h_f, c_f)
                fwd_outputs.append(h_f.unsqueeze(1))
            fwd_seq = torch.cat(fwd_outputs, dim=1)

            if self.bidirectional:
                bwd_cell = self.cells[layer * self.num_dirs + 1]
                h_b = h[layer * self.num_dirs + 1]
                c_b = c[layer * self.num_dirs + 1]

                bwd_outputs = []
                for t in reversed(range(seq_len)):
                    h_b, c_b = bwd_cell(layer_input[:, t, :], h_b, c_b)
                    bwd_outputs.insert(0, h_b.unsqueeze(1))
                bwd_seq = torch.cat(bwd_outputs, dim=1)

                layer_input = torch.cat([fwd_seq, bwd_seq], dim=2)
            else:
                layer_input = fwd_seq

        return layer_input, (h, c)


def _quantize_fp16(model):
    q = copy.deepcopy(model).eval().cpu()
    converted = 0
    for param in q.parameters():
        param.data = param.data.half()
        converted += 1
    for buf in q.buffers():
        if buf.dtype == torch.float32:
            buf.data = buf.data.half()
    print(f"  [FP16] Converted {converted} parameter tensors to float16.")
    return q


def _quantize_int8(model):
    q = copy.deepcopy(model).eval().cpu()
    replaced = 0

    if hasattr(q, 'lstm') and isinstance(q.lstm, nn.LSTM):
        bidir_tag = " (bidirectional)" if q.lstm.bidirectional else ""
        q.lstm = _INT8NNLSTMWrapper(q.lstm)
        print(f"  [INT8] Quantized q.lstm{bidir_tag} "
              f"({q.lstm.num_layers} layer(s), hidden={q.lstm.hidden_size})")
        replaced += 1

    if hasattr(q, 'lstm_layers'):
        new_layers = nn.ModuleList()
        for i, layer in enumerate(q.lstm_layers):
            if isinstance(layer, nn.LSTM):
                new_layers.append(_INT8NNLSTMWrapper(layer))
                print(f"  [INT8] Quantized lstm_layers[{i}] "
                      f"(hidden={layer.hidden_size})")
                replaced += 1
            else:
                new_layers.append(layer)
        q.lstm_layers = new_layers

    print(f"  [INT8] Total LSTM modules quantized: {replaced}")
    return q


def apply_quantization(
    model,
    quant_types: list,
    test_path:  str = "test.csv",
    label_col:  str = "Activity",
    batch_size: int = 64,
) -> torch.nn.Module:
    """Apply quantization to a trained/pruned LSTM model."""

    print(f"\n{'='*65}")
    print("APPLY QUANTIZATION")
    print(f"{'='*65}")
    print(f"  Methods requested : {quant_types}")

    test_loader = _build_test_loader(test_path, batch_size, label_col)

    orig_size = model_size_mb(model)
    orig_acc  = _evaluate(model, test_loader)
    orig_spd  = _inference_speed(model, test_loader)

    print(f"\n  {'Method':<14} {'Size (MB)':<12} {'Reduction':<12} "
          f"{'Accuracy':<12} {'Drop':<10} {'Speed (ms)'}")
    print(f"  {'-'*72}")
    print(f"  {'FP32 (orig)':<14} {orig_size:<12.2f} {'1.00x':<12} "
          f"{orig_acc:<12.2f} {'—':<10} {orig_spd:.2f}")

    final_model = model
    results     = {}

    for qtype in quant_types:
        key = qtype.upper().strip()
        print(f"\n  Running {key} quantization...")
        try:
            if key == 'FP16':
                q_model = _quantize_fp16(model)
            elif key == 'INT8':
                q_model = _quantize_int8(model)
            else:
                print(f"  [WARNING] Unknown type '{qtype}'. Use 'FP16' or 'INT8'.")
                continue

            q_size = _model_size_mb(q_model)
            q_acc  = _evaluate(q_model, test_loader)
            q_spd  = _inference_speed(q_model, test_loader)
            drop   = orig_acc - q_acc
            red    = f"{orig_size / max(q_size, 1e-6):.2f}x"

            print(f"  {key:<14} {q_size:<12.2f} {red:<12} "
                  f"{q_acc:<12.2f} {drop:<+10.2f} {q_spd:.2f}")

            results[key] = dict(model=q_model, size=q_size,
                                acc=q_acc, speed=q_spd,
                                reduction=red, drop=drop)
            final_model = q_model

        except Exception as e:
            print(f"  [ERROR] {key} failed: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*65}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*65}")
    for key, res in results.items():
        print(f"  {key}:")
        print(f"    Size     : {orig_size:.2f} MB  →  {res['size']:.2f} MB  ({res['reduction']} smaller)")
        print(f"    Accuracy : {orig_acc:.2f}%  →  {res['acc']:.2f}%  (drop: {res['drop']:+.2f}%)")
        print(f"    Speed    : {orig_spd:.2f} ms  →  {res['speed']:.2f} ms / batch")
    print(f"{'='*65}\n")

    return final_model


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _sample_csv(path: str, fraction: float, seed: int = 42) -> str:
    """
    Sample `fraction` of a CSV and write to a temp file.
    Returns the path of the sampled file (caller must delete it).
    """
    if fraction >= 1.0:
        return path
    df = pd.read_csv(path, low_memory=False)
    df_sampled = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df_sampled.to_csv(tmp.name, index=False)
    tmp.close()
    print(f"[run_pipeline] Using {fraction*100:.0f}% of dataset: "
          f"{len(df_sampled)}/{len(df)} rows from {os.path.basename(path)}")
    return tmp.name


def run_pipeline(
    train_path,
    test_path,
    input_size,
    output_size,
    use_embedding,
    num_architectures,
    sparsity,
    quant_types,
    dataset_fraction: float = 1.0,
    save_dir: str = ".",
):
    """
    End-to-end TGF-NAS + LAHUP Pruning + Quantization pipeline.

    Parameters
    ----------
    train_path        : Path to training CSV
    test_path         : Path to test CSV
    input_size        : Number of input features (0 = auto-detect)
    output_size       : Number of output classes  (0 = auto-detect)
    use_embedding     : Reserved flag
    num_architectures : Number of architectures to sample
    sparsity          : Pruning sparsity (e.g. 0.3 for 30%)
    quant_types       : List of quantization types e.g. ['FP16', 'INT8']
    dataset_fraction  : Fraction of data to use, e.g. 0.5 = 50% (default 1.0)
    save_dir          : Directory where model .pth files are saved

    Returns
    -------
    dict with keys: accuracy, sparsity, quantization, model_paths, stage_metrics
    """
    import tempfile as _tempfile
    os.makedirs(save_dir, exist_ok=True)

    # ── Optionally subsample the CSVs ─────────────────────────────────────────
    sampled_train = _sample_csv(train_path, dataset_fraction)
    sampled_test  = _sample_csv(test_path,  dataset_fraction)
    _tmp_files = []
    if sampled_train != train_path: _tmp_files.append(sampled_train)
    if sampled_test  != test_path:  _tmp_files.append(sampled_test)

    try:
        # ── Auto-detect input_size / output_size from the CSV ─────────────────
        df_probe  = pd.read_csv(sampled_train, low_memory=False)
        id_cols   = {"subject", "Subject", "subject_id", "Activity", "activity", "label", "Label"}
        label_col_found = None
        for c in ["Activity", "activity", "label", "Label"]:
            if c in df_probe.columns:
                label_col_found = c
                break

        if not output_size:
            if label_col_found:
                output_size = int(df_probe[label_col_found].nunique())
                print(f"[run_pipeline] Auto-detected output_size={output_size} "
                      f"from column '{label_col_found}'")
            else:
                raise ValueError("output_size not given and no label column found in CSV.")

        if not input_size:
            feat_cols  = [c for c in df_probe.columns if c not in id_cols]
            input_size = len(feat_cols)
            print(f"[run_pipeline] Auto-detected input_size={input_size}")
        del df_probe

        # ── Stage 1: TGF-NAS + full training ─────────────────────────────────
        result = TGF_NAS(
            train_path=sampled_train,
            test_path=sampled_test,
            input_size=input_size,
            output_size=output_size,
            use_embedding=use_embedding,
            num_architectures=num_architectures,
            sparsity=sparsity,
            quant_types=quant_types,
        )

        trained_model = result["best_model"]
        trained_acc   = result["accuracy"]

        # Save full-trained model
        path_full = os.path.join(save_dir, "model_full_trained.pth")
        torch.save(trained_model.state_dict(), path_full)
        size_full = model_size_mb(trained_model)
        print(f"[run_pipeline] Saved full-trained model → {path_full} ({size_full:.2f} MB)")

        # ── Stage 2: LAHUP Pruning ────────────────────────────────────────────
        pruned_model = apply_pruning(
            trained_model, sparsity,
            train_path=sampled_train,
            test_path=sampled_test,
            save_path=os.path.join(save_dir, "model_pruned.pth"),
        )
        path_pruned = os.path.join(save_dir, "model_pruned.pth")
        size_pruned = model_size_mb(pruned_model)

        # Evaluate pruned accuracy
        _tl = _build_test_loader(sampled_test)
        pruned_acc = _evaluate(pruned_model, _tl)

        # ── Stage 3: Quantization ─────────────────────────────────────────────
        # Capture per-method results by running each type separately
        quant_metrics = {}
        final_model   = pruned_model
        for qtype in quant_types:
            q_model = apply_quantization(
                pruned_model, [qtype],
                test_path=sampled_test,
            )
            q_acc  = _evaluate(q_model, _tl)
            q_size = _model_size_mb(q_model)
            q_spd  = _inference_speed(q_model, _tl)
            quant_metrics[qtype] = {"accuracy": q_acc, "size": q_size, "speed": q_spd}
            path_q = os.path.join(save_dir, f"model_quant_{qtype.lower()}.pth")
            torch.save(q_model.state_dict(), path_q)
            print(f"[run_pipeline] Saved {qtype} model → {path_q} ({q_size:.2f} MB)")
            final_model = q_model

        # ── Collect model paths ───────────────────────────────────────────────
        model_paths = {
            "full_trained": path_full,
            "pruned":       path_pruned,
        }
        for qtype in quant_types:
            model_paths[f"quant_{qtype.lower()}"] = os.path.join(
                save_dir, f"model_quant_{qtype.lower()}.pth"
            )

        # ── Collect stage metrics for heatmap ────────────────────────────────
        # speed for full model
        full_spd  = _inference_speed(trained_model, _tl)
        pruned_spd = _inference_speed(pruned_model, _tl)

        stage_metrics = {
            "Base": {
                "accuracy": trained_acc,
                "size":     size_full,
                "speed":    full_spd,
                "sparsity": 0.0,
            },
            "LAHUP + FT": {
                "accuracy": pruned_acc,
                "size":     size_pruned,
                "speed":    pruned_spd,
                "sparsity": float(sparsity * 100),
            },
        }
        for qtype, qm in quant_metrics.items():
            stage_metrics[qtype] = {
                "accuracy": qm["accuracy"],
                "size":     qm["size"],
                "speed":    qm["speed"],
                "sparsity": float(sparsity * 100),
            }

        return {
            "accuracy":     trained_acc,
            "sparsity":     sparsity,
            "quantization": quant_types,
            "model_paths":  model_paths,
            "stage_metrics": stage_metrics,
        }

    finally:
        # Clean up temp sampled CSVs
        for f in _tmp_files:
            try: os.remove(f)
            except Exception: pass


if __name__ == "__main__":
    result = run_pipeline(
        train_path="train.csv",
        test_path="test.csv",
        input_size=561,
        output_size=6,
        use_embedding=False,
        num_architectures=2,
        sparsity=0.3,
        quant_types=["FP16", "INT8"],
    )
    print(result)