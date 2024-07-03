import numpy as np
import torch
from ddpm import DDPMSampler
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep):
    # positional encoding from Transformer: int -> (1, 320)

    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 2 * 160) -> (1, 320)
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    return x


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        # Validate the strength parameter
        if not 0 < strength <= 1:
            raise ValueError("Strength must be in the range (0, 1]")

        # Define a function to move the model to the idle device if provided
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize the random number generator
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        # Load the CLIP model for extracting text features from prompts
        clip = models["clip"]
        clip.to(device)

        # Classifier-Free Guidance
        if do_cfg:
            # Process the conditional prompt

            # Pad tokens to length 77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids

            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(
                cond_tokens,
                dtype=torch.long,
                device=device,
            )

            # Extract text features from the conditional prompt using CLIP
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            # Process the unconditional prompt

            # Pad tokens to length 77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt],
                padding="max_length",
                max_length=77,
            ).input_ids

            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(
                uncond_tokens,
                dtype=torch.long,
                device=device,
            )

            # Extract text features from the unconditional prompt using CLIP
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # Concatenate conditional and unconditional text features
            context = torch.cat([cond_context, uncond_context], dim=0)

        else:
            # Process the prompt without classifier-free guidance

            # Pad tokens to length 77
            tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids

            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            )

            # Extract text features from the prompt using CLIP
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)

        # Move the CLIP model to idle device
        to_idle(clip)

        # Load the DDPM sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        # Shape of the input latent of the VAE Encoder
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            # Encode the input image to get the latent tensor
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(
                input_image_tensor,
                dtype=torch.float32,
                device=device,
            )
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(
                input_image_tensor,
                old_range=(0, 255),
                new_range=(-1, 1),
                clamp=False,
            )
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device
            )
            # (Batch_Size, 4, Latents_Height, Latents_Width), (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latent tensor according to the strength parameter
            sampler.set_strength(strength=strength)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move the encoder model to idle device
            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Generate a new latent tensor for the VAE Decoder by the diffusion model from the given latent tensor
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # Inference 2 times, one time with the conditional prompt and one time with the unconditional prompt
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (2 * Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width) if do_cfg
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width) if not do_cfg
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # (2 * Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width) X 2
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                # formula of the classifier-free guidance
                # (Batch_Size, 4, Latents_Height, Latents_Width) X 2 -> (Batch_Size, 4, Latents_Height, Latents_Width)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # DDPM sampler removes noise predicted by the UNET from the current latent tensor
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        # Move the diffusion model to idle device
        to_idle(diffusion)

        # Decode the denoised latent tensor to get the output image
        decoder = models["decoder"]
        decoder.to(device)

        # (Batch_Size, 4, Latents_Height, Latents_Width) = (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, Channel, Height, Width)
        images = decoder(latents)

        # Move the decoder model to idle device
        to_idle(decoder)

        images = rescale(
            images,
            old_range=(-1, 1),
            new_range=(0, 255),
            clamp=True,
        )
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]
