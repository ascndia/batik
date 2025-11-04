import torch
import torch.nn.functional as F
import numpy as np
from torch.func import jvp

class MeanFlow:
    def __init__(
        self,
        path_type="linear",
        weighting="uniform",
        # meanflow specific params
        time_sampler="logit_normal",  
        time_mu=-0.4,                 
        time_sigma=1.0,               
        ratio_r_not_equal_t=0.75,     
        adaptive_p=1.0,               
        label_dropout_prob=0.1,
        # REPA (Projection Loss) params
        proj_coeff=0.5,
        ):
        
        self.path_type = path_type
        self.weighting = weighting
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.adaptive_p = adaptive_p
        self.label_dropout_prob = label_dropout_prob
        
        # Store REPA params
        self.use_repa_loss = proj_coeff > 0.0
        self.proj_coeff = proj_coeff
    
    def sample_time_steps(self, batch_size, device):
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)
        
        return r, t 
        
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1.0
            d_sigma_t =  1.0
        elif self.path_type == "cosine":
            t_pi_half = t * (np.pi / 2)
            alpha_t = torch.cos(t_pi_half)
            sigma_t = torch.sin(t_pi_half)
            d_alpha_t = - (np.pi / 2) * torch.sin(t_pi_half)
            d_sigma_t = (np.pi / 2) * torch.cos(t_pi_half)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_loss(self, model, images, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        # --- Handle Label Dropout ---
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y']
            try:
                num_classes = model.num_classes 
            except AttributeError:
                try:
                    num_classes = model.module.num_classes
                except AttributeError:
                    raise AttributeError("Model does not have 'num_classes' or 'module.num_classes' attribute. Please set it.")
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            y_clone = y.clone()
            y_clone[dropout_mask] = num_classes
            model_kwargs['y'] = y_clone
            
        zs_target = model_kwargs.pop('zs_target', None) if self.use_repa_loss else None
        
        use_repa = self.use_repa_loss and zs_target is not None

        r, t = self.sample_time_steps(batch_size, device)
        noises = torch.randn_like(images)
        t_view = t.view(-1, *([1] * (images.dim() - 1)))
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_view)
        z_t = alpha_t * images + sigma_t * noises # Interpolated sample
        v_t = d_alpha_t * images + d_sigma_t * noises # Instantaneous velocity
        
        time_diff = (t - r).view(-1, *([1] * (images.dim() - 1)))

        # --- Model Forward Pass ---
        if use_repa:
            # Model MUST return (u, zs_tilde) if use_repa is True
            u, zs_tilde = model(z_t, r, t, **model_kwargs)
            if not isinstance(zs_tilde, (list, tuple)):
                 raise ValueError("Model did not return a list/tuple of hidden states (zs_tilde) for REPA loss.")
            if len(zs_tilde) != len(zs_target):
                raise ValueError(f"Mismatch in REPA layers: expected {len(zs_target)}, model returned {len(zs_tilde)}.")
        else:
            u = model(z_t, r, t, **model_kwargs)
            zs_tilde = None # Not used

        # --- Mean Flow JVP Calculation ---
        def fn_current(z, cur_r, cur_t):
            model_out = model(z, cur_r, cur_t, **model_kwargs)
            # Handle model output consistently
            if isinstance(model_out, (list, tuple)):
                return model_out[0] # Return just `u`
            else:
                return model_out    # This is `u`
            
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        _, dudt = jvp(fn_current, primals, tangents)
        u_target = v_t - time_diff * dudt
                
        # --- 1. Calculate Mean Flow Loss ---
        error = u - u_target.detach()
        loss_per_sample = torch.sum((error**2).reshape(error.shape[0], -1), dim=-1)
        
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_per_sample.detach() + 1e-3).pow(self.adaptive_p)
            mean_flow_loss = (weights * loss_per_sample).mean()
        else:
            mean_flow_loss = loss_per_sample.mean()

        # --- 2. Calculate REPA Projection Loss ---
        proj_loss = torch.tensor(0.0, device=device)
        if use_repa: # zs_target is guaranteed to be non-None here
            layer_losses = []
            for z_target_layer, z_pred_layer in zip(zs_target, zs_tilde):
                z_target_norm = F.normalize(z_target_layer, dim=-1)
                z_pred_norm = F.normalize(z_pred_layer, dim=-1)
                
                cos_sim = (z_target_norm * z_pred_norm).sum(dim=-1)
                layer_loss = -cos_sim.mean()
                layer_losses.append(layer_loss)
            
            if len(layer_losses) > 0:
                proj_loss = torch.stack(layer_losses).mean()

        # --- 3. Combine Losses ---
        total_loss = mean_flow_loss + (self.proj_coeff * proj_loss)
        
        return total_loss, mean_flow_loss, proj_loss