import numpy as np
import torch


class DDPMSampler:
    def __init__(
        self, generator, num_training_steps=1000, beta_start=8.5e-4, beta_end=1.2e-2
    ):
        self.betas = (
            torch.linspace(
                start=beta_start**0.5,
                end=beta_end**0.5,
                steps=1000,
                dtype=torch.float32,
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.arange(
            start=num_training_steps - 1, end=-1, step=-1, dtype=torch.int
        )

    def _get_previous_timestep(self, timestep):
        step_size = self.num_training_steps // self.num_inference_timesteps
        prev_t = timestep - step_size

        return prev_t

    def _get_variance(self, timestep):
        t = timestep
        t_prev = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        variance = beta_prod_t_prev / beta_prod_t * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_inference_timesteps(self, num_inference_timesteps=50):
        self.num_inference_timesteps = num_inference_timesteps
        step_size = self.num_training_steps // self.num_inference_timesteps
        first_timestep = (self.num_inference_timesteps - 1) * step_size
        self.timesteps = torch.linspace(
            start=first_timestep,
            end=0,
            steps=self.num_inference_timesteps,
            dtype=torch.int,
        )

    def set_strength(self, strength):
        # Set the strength of randomness by controlling the number of inference steps to skip
        # Larger Strength (-> 1) -> Skip fewer inference steps -> More noise added -> Output will be further from the input image
        # Smaller Strength (-> 0) -> Skip more inference steps -> Less noise added -> Output will be closer to the input image

        start_step = self.num_inference_timesteps - int(
            self.num_inference_timesteps * strength
        )
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    # Equations from DDPM paper
    def add_noise(self, original_samples, timestep):
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timestep = timestep.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # std
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from N(mean, std): X = mean + std * noise (N(0, 1))
        noise = torch.randn(
            size=original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        mean = sqrt_alpha_prod * original_samples
        noisy_samples = mean + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    # Equations from DDPM paper
    def step(self, timestep, latents, model_output):
        # Get t_prev, the previous timestep (in the direction of adding noise), also the next timestep in the direction of denoising
        # The latent at timestep t_prev has less noise comapred to the latent at timestep t
        t = timestep
        t_prev = self._get_previous_timestep(timestep)

        # Get alpha_prod, beta_prod, current_alpha, current_beta for timesteps t and t_prev
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Given the latent and noise at timestep t, directly predict x_0, the latents at timestep 0
        pred_original_sample = (
            latents - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        # How much the current prediction of x_0 accounts for the latent at timestep t_prev according to alpha and beta values
        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * current_beta_t
        ) / beta_prod_t

        # How much the latent at timestep t accounts for the latent at timestep t_prev according to alpha and beta values
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # Weighted sum of the current prediction of x_0 and the latent at timestep t to get the predicted mean of the latent at timestep t_prev
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        # Add noise to the latent at timestep t_prev if it is not the first timestep for adding noise / last timestep for denoising (t > 0)
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            std = self._get_variance(t) ** 0.5
            variance = std * noise
        else:
            variance = 0

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
