"""
PredictiveWorldModel - Modello Predittivo del Mondo per GOD AGI
=============================================================

Modello neurale per predizione stato, reward, done flag e incertezza, integrabile con pipeline evolutiva e embodied.
Migrato e modernizzato da: vecchi/models/world_model.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class PredictiveWorldModel(nn.Module):
    """
    Modello predittivo del mondo: dato uno stato latente e un embedding di azione,
    predice il prossimo stato, la ricompensa e il flag di fine episodio.
    Supporta incertezza (media/log-var) e training evolutivo.
    """
    def __init__(self,
                 latent_dim: int,
                 action_input_embedding_dim: int,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 3,
                 lr: float = 1e-4,
                 device: str = 'cpu',
                 predict_done_flag: bool = True,
                 state_loss_weight: float = 1.0,
                 reward_loss_weight: float = 1.0,
                 done_loss_weight: float = 0.5,
                 predict_state_uncertainty: bool = True):
        super().__init__()
        self.action_input_embedding_dim = action_input_embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.predict_done_flag = predict_done_flag
        self.predict_state_uncertainty = predict_state_uncertainty
        self.state_loss_weight = state_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.done_loss_weight = done_loss_weight if predict_done_flag else 0.0
        input_dim = latent_dim + self.action_input_embedding_dim
        # Stato predittivo
        state_predictor_layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
        for _ in range(num_hidden_layers - 1):
            state_predictor_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()])
        output_state_dim = latent_dim * 2 if self.predict_state_uncertainty else latent_dim
        state_predictor_layers.append(nn.Linear(hidden_dim, output_state_dim))
        self.state_predictor = nn.Sequential(*state_predictor_layers).to(device)
        # Reward
        reward_predictor_layers = [nn.Linear(input_dim, hidden_dim // 2), nn.LeakyReLU()]
        for _ in range(num_hidden_layers - 2):
            if hidden_dim // 2 > 1:
                reward_predictor_layers.extend([nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.LeakyReLU()])
        reward_predictor_layers.append(nn.Linear(hidden_dim // 2, 1))
        self.reward_predictor = nn.Sequential(*reward_predictor_layers).to(device)
        # Done
        if self.predict_done_flag:
            done_predictor_layers = [nn.Linear(input_dim, hidden_dim // 2), nn.LeakyReLU()]
            for _ in range(num_hidden_layers - 2):
                if hidden_dim // 2 > 1:
                    done_predictor_layers.extend([nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.LeakyReLU()])
            done_predictor_layers.append(nn.Linear(hidden_dim // 2, 1))
            self.done_predictor = nn.Sequential(*done_predictor_layers).to(device)
        else:
            self.done_predictor = None
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.001)
        if self.predict_state_uncertainty:
            self.state_loss_fn = lambda pred_mean, pred_logvar, target: F.gaussian_nll_loss(pred_mean, target, torch.exp(pred_logvar), reduction='mean')
        else:
            self.state_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()
        if self.predict_done_flag:
            self.done_loss_fn = nn.BCEWithLogitsLoss()
        logger.info(f"PredictiveWorldModel initialized. LatentDim: {latent_dim}, ActionInputEmbDim: {self.action_input_embedding_dim}, HiddenDim: {hidden_dim}, Layers: {num_hidden_layers}, PredictUncertainty: {predict_state_uncertainty}")
    def _prepare_input(self, s_latent: np.ndarray, action_input_embedding: np.ndarray) -> torch.Tensor:
        state_tensor = torch.FloatTensor(s_latent).to(self.device)
        action_emb_tensor = torch.FloatTensor(action_input_embedding).to(self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if action_emb_tensor.ndim == 1:
            action_emb_tensor = action_emb_tensor.unsqueeze(0)
        if state_tensor.shape[0] != action_emb_tensor.shape[0]:
            raise ValueError(f"Batch size mismatch: stato {state_tensor.shape[0]}, azione {action_emb_tensor.shape[0]}")
        if action_emb_tensor.shape[1] != self.action_input_embedding_dim:
            raise ValueError(f"Dimensione embedding azione errata: fornito {action_emb_tensor.shape[1]}, atteso {self.action_input_embedding_dim}")
        model_input = torch.cat((state_tensor, action_emb_tensor), dim=1)
        return model_input
    def forward(self, s_latent: np.ndarray, action_input_embedding: np.ndarray, deterministic_state: bool = False) -> Dict:
        self.eval()
        model_input = self._prepare_input(s_latent, action_input_embedding)
        with torch.no_grad():
            predicted_next_state_params = self.state_predictor(model_input)
            predicted_reward = self.reward_predictor(model_input)
            output_dict = {}
            if self.predict_state_uncertainty:
                mean = predicted_next_state_params[:, :self.latent_dim]
                logvar = predicted_next_state_params[:, self.latent_dim:]
                if deterministic_state:
                    output_dict['next_state'] = mean
                else:
                    output_dict['next_state_mean'] = mean
                    output_dict['next_state_logvar'] = logvar
            else:
                output_dict['next_state'] = predicted_next_state_params
            output_dict['reward'] = predicted_reward
            if self.predict_done_flag and self.done_predictor:
                predicted_done_logits = self.done_predictor(model_input)
                output_dict['done_logits'] = predicted_done_logits
        return output_dict
    def train_step(self, s_latent_pre: np.ndarray, action_embedding: np.ndarray, 
                   reward: float, s_latent_post: np.ndarray, done: bool) -> Dict[str, float]:
        self.train()
        self.optimizer.zero_grad()
        model_input = self._prepare_input(s_latent_pre, action_embedding)
        predicted_next_state_params = self.state_predictor(model_input)
        predicted_reward = self.reward_predictor(model_input).squeeze(-1)
        target_next_state = torch.FloatTensor(s_latent_post).to(self.device)
        if target_next_state.ndim == 1: target_next_state = target_next_state.unsqueeze(0)
        target_reward = torch.FloatTensor([reward]).to(self.device)
        if self.predict_state_uncertainty:
            pred_mean = predicted_next_state_params[:, :self.latent_dim]
            pred_logvar = predicted_next_state_params[:, self.latent_dim:]
            state_loss = self.state_loss_fn(pred_mean, pred_logvar, target_next_state)
        else:
            state_loss = self.state_loss_fn(predicted_next_state_params, target_next_state)
        reward_loss = self.reward_loss_fn(predicted_reward, target_reward)
        total_loss = state_loss * self.state_loss_weight + reward_loss * self.reward_loss_weight
        losses = {
            'total_loss': 0.0,
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item()
        }
        if self.predict_done_flag and self.done_predictor:
            predicted_done_logits = self.done_predictor(model_input).squeeze(-1)
            target_done = torch.FloatTensor([1.0 if done else 0.0]).to(self.device)
            done_loss = self.done_loss_fn(predicted_done_logits, target_done)
            total_loss += done_loss * self.done_loss_weight
            losses['done_loss'] = done_loss.item()
        losses['total_loss'] = total_loss.item()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return losses
    def get_prediction_error(self, s_latent_pre: np.ndarray, action_embedding: np.ndarray, s_latent_post: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            model_input = self._prepare_input(s_latent_pre, action_embedding)
            predicted_next_state_params = self.state_predictor(model_input)
            if self.predict_state_uncertainty:
                predicted_mean_state = predicted_next_state_params[:, :self.latent_dim]
            else:
                predicted_mean_state = predicted_next_state_params
            target_next_state = torch.FloatTensor(s_latent_post).to(self.device)
            if target_next_state.ndim == 1: target_next_state = target_next_state.unsqueeze(0)
            error = F.mse_loss(predicted_mean_state, target_next_state, reduction='mean')
        return error.item()
    def save_model(self, path: str):
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"PredictiveWorldModel salvato in {path}")
        except Exception as e:
            logger.error(f"Errore durante il salvataggio del PredictiveWorldModel: {e}", exc_info=True)
    def load_model(self, path: str, map_location: Optional[str] = None):
        try:
            if map_location:
                self.load_state_dict(torch.load(path, map_location=torch.device(map_location)))
            else:
                self.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"PredictiveWorldModel caricato da {path}")
            self.to(self.device)
        except FileNotFoundError:
            logger.error(f"File del modello non trovato in {path}. Inizializzazione con pesi casuali.")
        except Exception as e:
            logger.error(f"Errore durante il caricamento del PredictiveWorldModel da {path}: {e}", exc_info=True)
    def get_state_summary(self) -> Dict[str, Any]:
        """Provide state summary for checkpoint system"""
        return {
            'latent_dim': self.latent_dim,
            'action_dim': self.action_input_embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_hidden_layers,
            'device': str(self.device),
            'predict_done': self.predict_done_flag,
            'predict_uncertainty': self.predict_state_uncertainty,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'training_mode': self.training
        }
    def get_experience_count(self) -> int:
        """Get number of training experiences processed"""
        return getattr(self, '_experience_count', 0)
    
    def increment_experience_count(self):
        """Increment experience counter"""
        if not hasattr(self, '_experience_count'):
            self._experience_count = 0
        self._experience_count += 1