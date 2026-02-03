import math
import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Union, List
from dataclasses import dataclass
from tqdm import tqdm
import torch.distributed as dist

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def run_sample_step(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    image_ids, 
    grpo_sample,
    determistic,
):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []

    if "dpmsolver" in args.dpm_algorithm_type:
        dpm_state = DPMState(order=args.dpm_solver_order)
        if args.dpm_apply_strategy == "post":
            assert args.sample_strategy == "progressive", "post strategy is only supported for progressive sampling"
            first_false_index = next(
                (i for i, value in enumerate(determistic) if not value), 
                None
            )
            last_false_index = next(
                (i for i, value in enumerate(reversed(determistic)) if not value), 
                None
            )
            if last_false_index is not None:
                last_false_index = len(determistic) - 1 - last_false_index

            num_post_steps = int(max((sigma_schedule.size(0) - 1 - last_false_index)*args.dpm_post_compress_ratio, 1))
           
            # rebuild post sigma schedule
            post_time_step = torch.linspace(1, 0, sigma_schedule.size(0))[last_false_index+1].item()
            post_sigma_schedule = torch.linspace(post_time_step, 0, num_post_steps).to(z.device)
            post_sigma_schedule = sd3_time_shift(args.shift, post_sigma_schedule)

            sigma_schedule = torch.cat(
                [sigma_schedule[:last_false_index+1], post_sigma_schedule],
                dim=0
            )
            progress_bar = tqdm(
                range(0, sigma_schedule.size(0) - 1), 
                desc="Sampling steps", 
                disable=not dist.is_initialized() or dist.get_rank() != 0,
            )

    for i in progress_bar:
        B = encoder_hidden_states.shape[0]
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
        transformer.eval()
        with torch.autocast("cuda", torch.bfloat16):
            pred=transformer(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps/1000,
                guidance=torch.tensor(
                    [3.5],
                    device=z.device,
                    dtype=torch.bfloat16
                ),
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), # B, L
                pooled_projections=pooled_prompt_embeds,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        if args.dpm_algorithm_type == "null": # disable dpm_solver
            if args.flow_grpo_sampling:
                z, pred_original, log_prob, prev_latents_mean, std_dev_t = flow_grpo_step(
                    model_output=pred,
                    latents=z.to(torch.float32),
                    eta=args.eta,
                    sigmas=sigma_schedule,
                    index=i,
                    prev_sample=None,
                    determistic=determistic[i],
                    sde_type=getattr(args, "sde_type", "sde"),
                    noise_level=getattr(args, "noise_level", None),
                )
            else:
                if determistic[i]:
                    z, pred_original, log_prob = dance_grpo_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=False, sde_type=getattr(args, "sde_type", "sde"), noise_level=getattr(args, "noise_level", None))
                else:
                    z, pred_original, log_prob = dance_grpo_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True, sde_type=getattr(args, "sde_type", "sde"), noise_level=getattr(args, "noise_level", None))
        elif "dpmsolver" in args.dpm_algorithm_type:
            if args.dpm_apply_strategy == "all":
                z, pred_original, log_prob = dpm_step(
                    args,
                    model_output=pred,
                    sample=z.to(torch.float32),
                    step_index=i,
                    timesteps=sigma_schedule[:-1],
                    dpm_state=dpm_state,
                    generator=torch.Generator(device=z.device),
                    sde_solver=(not determistic[i]),
                    sigmas=sigma_schedule,
                )
            elif args.dpm_apply_strategy == "post":
                assert args.sample_strategy == "progressive", "post strategy is only supported for progressive sampling"
                if i <= last_false_index:
                    if args.flow_grpo_sampling:
                        x_0 = convert_model_output(pred, z.to(torch.float32), sigma_schedule, step_index=i)
                        dpm_state.update(x_0)
                        z, pred_original, log_prob, prev_latents_mean, std_dev_t = flow_grpo_step(
                            model_output=pred,
                            latents=z.to(torch.float32),
                            eta=args.eta,
                            sigmas=sigma_schedule,
                            index=i,
                            prev_sample=None,
                            determistic=determistic[i],
                            sde_type=getattr(args, "sde_type", "sde"),
                            noise_level=getattr(args, "noise_level", None),
                        )
                        dpm_state.update_lower_order()

                    else:
                        if determistic[i]:
                            z, pred_original, log_prob = dance_grpo_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=False, sde_type=getattr(args, "sde_type", "sde"), noise_level=getattr(args, "noise_level", None))
                        else:
                            z, pred_original, log_prob = dance_grpo_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True, sde_type=getattr(args, "sde_type", "sde"), noise_level=getattr(args, "noise_level", None))
                else:
                    z, pred_original, log_prob = dpm_step(
                        args,
                        model_output=pred,
                        sample=z.to(torch.float32),
                        step_index=i,
                        timesteps=sigma_schedule[:-1],
                        dpm_state=dpm_state,
                        sde_solver=False,
                        sigmas=sigma_schedule,
                    )
        z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)

    if args.drop_last_sample:
        latents = pred_original
    else:
        latents = z.to(pred_original.dtype)
    all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
    all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    return z, latents, all_latents, all_log_probs

def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    sde_type: str = "sde",
    noise_level: Optional[float] = None,
):
    device = model_output.device
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma  # neg dt

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if sde_type == "sde":
        _noise_level = eta if noise_level is None else noise_level
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * _noise_level
        pred_original_sample = latents - sigma * model_output
        prev_sample_mean = latents * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        if determistic:
            prev_sample = latents + dt * model_output

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=device)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt)

    elif sde_type == "cps":
        _noise_level = (0.8 if noise_level is None else noise_level)
        std_dev_t = sigma_prev * math.sin(_noise_level * math.pi / 2)
        pred_original_sample = latents - sigma * model_output
        noise_estimate = latents + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if determistic:
            prev_sample = latents + dt * model_output

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}. Must be 'sde' or 'cps'.")

def dance_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
    sde_type: str = "sde",
    noise_level: Optional[float] = None,
):
    device = latents.device
    sigma = sigmas[index].to(device) if sigmas[index].device != device else sigmas[index]
    sigma_prev = sigmas[index + 1].to(device) if sigmas[index + 1].device != device else sigmas[index + 1]
    dsigma = sigma_prev - sigma  # neg dt
    pred_original_sample = latents - sigma * model_output

    if sde_type == "sde":
        prev_sample_mean = latents + dsigma * model_output
        delta_t = sigma - sigma_prev  # pos -dt
        _noise_level = eta if noise_level is None else noise_level
        std_dev_t = _noise_level * torch.sqrt(delta_t)

        if sde_solver:
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            log_term = -0.5 * _noise_level**2 * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

        if grpo and prev_sample is None:
            if sde_solver:
                prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean, device=device) * std_dev_t
            else:
                prev_sample = prev_sample_mean

        if grpo:
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
            )
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample

    elif sde_type == "cps":
        _noise_level = 0.8 if noise_level is None else noise_level
        std_dev_t = sigma_prev * math.sin(_noise_level * math.pi / 2)
        noise_estimate = latents + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if grpo and prev_sample is None:
            if sde_solver:
                prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(prev_sample_mean, device=device, dtype=prev_sample_mean.dtype)
            else:
                prev_sample = prev_sample_mean

        if grpo:
            log_prob = -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}. Must be 'sde' or 'cps'.")

@dataclass
class DPMState:
    order: int
    model_outputs: List[torch.Tensor] = None
    lower_order_nums = 0

    def __post_init__(self):
        self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor):
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self):
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1

def dpm_step(
    args,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    timesteps: list,
    sigmas: torch.Tensor,
    dpm_state: DPMState = None,
    generator=None,
    variance_noise: Optional[torch.Tensor] = None,
    sde_solver: bool = False,
) -> torch.Tensor:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
    the multistep DPMSolver.

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`int`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        variance_noise (`torch.Tensor`):
            Alternative to generating noise with `generator` by directly providing the noise for the variance
            itself. Useful for methods such as [`LEdits++`].

    Returns:
        prev_sample (`torch.Tensor`):
            The sample from the previous timestep.
    """

    # Improve numerical stability for small number of steps
    lower_order_final = (step_index == len(timesteps) - 1) 
    lower_order_second = ((step_index == len(timesteps) - 2) and len(timesteps) < 15)

    model_output = convert_model_output(model_output, sample, sigmas, step_index=step_index)
    
    if dpm_state is not None:
        dpm_state.update(model_output)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)
    if sde_solver and variance_noise is None:
        noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
        )
    elif sde_solver:
        noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
    else:
        noise = None

    if dpm_state:
        if args.dpm_solver_order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
            prev_sample, prev_sample_mean, std_dev_t, dt_sqrt = dpm_solver_first_order_update(
                args,
                model_output,
                sigmas,
                step_index,
                sample, 
                noise,
                sde_solver
            )
        elif args.dpm_solver_order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
            prev_sample, prev_sample_mean, std_dev_t, dt_sqrt = multistep_dpm_solver_second_order_update(
                args,
                dpm_state.model_outputs, 
                sigmas,
                step_index,
                sample, 
                noise,
                sde_solver=sde_solver,
            )
        else:
            prev_sample, prev_sample_mean, std_dev_t, dt_sqrt = multistep_dpm_solver_third_order_update(
                args,
                dpm_state.model_outputs, 
                sigmas,
                step_index,
                sample, 
                noise,
                sde_solver=sde_solver,
            )
    else:
        prev_sample, prev_sample_mean, std_dev_t, dt_sqrt = dpm_solver_first_order_update(
            args,
            model_output,
            sigmas,
            step_index,
            sample, 
            noise,
            sde_solver=sde_solver,
        )

    if dpm_state is not None:
        dpm_state.update_lower_order()

    # Cast sample back to expected dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # Compute log_prob
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * dt_sqrt)**2))
        - torch.log(std_dev_t * dt_sqrt)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, model_output, log_prob

def convert_model_output(
    model_output,
    sample,
    sigmas,
    step_index,
) -> torch.Tensor:
    sigma_t = sigmas[step_index]
    x0_pred = sample - sigma_t * model_output

    return x0_pred

def dpm_solver_first_order_update(
    args,
    model_output: torch.Tensor,
    sigmas,
    step_index,
    sample: torch.Tensor = None,
    noise: Optional[torch.Tensor] = None,
    sde_solver: bool = False,
) -> torch.Tensor:
    """
    One step for the first-order DPMSolver (equivalent to DDIM).

    Args:
        model_output (`torch.Tensor`):
            The direct output from the learned diffusion model.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.

    Returns:
        `torch.Tensor`:
            The sample tensor at the previous timestep.
    """

    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
 
    h = lambda_t - lambda_s
    if args.dpm_algorithm_type == "dpmsolver++":
        prev_mean = ((sigma_t / sigma_s * torch.exp(-h)) * sample + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output)
        std_dev_t = sigma_t
        dt_sqrt = torch.sqrt(1.0 - torch.exp(-2 * h))
        if sde_solver:
            assert noise is not None
            x_t = prev_mean + std_dev_t * dt_sqrt * noise
        else:
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    elif args.dpm_algorithm_type == "dpmsolver":
        prev_mean = (alpha_t / alpha_s) * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
        std_dev_t = sigma_t
        dt_sqrt = torch.sqrt(torch.exp(2 * h) - 1.0)
        if sde_solver:
            assert noise is not None
            x_t = prev_mean + std_dev_t * dt_sqrt * noise
        else:
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output

    return x_t, prev_mean, std_dev_t, dt_sqrt

def multistep_dpm_solver_second_order_update(
    args,
    model_output_list: List[torch.Tensor],
    sigmas,
    step_index,
    sample: torch.Tensor = None,
    noise: Optional[torch.Tensor] = None,
    sde_solver: bool = False,
) -> torch.Tensor:
    """
    One step for the second-order multistep DPMSolver.

    Args:
        model_output_list (`List[torch.Tensor]`):
            The direct outputs from learned diffusion model at current and latter timesteps.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.

    Returns:
        `torch.Tensor`:
            The sample tensor at the previous timestep.
    """

    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

    m0, m1 = model_output_list[-1], model_output_list[-2]

    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)
    if args.dpm_algorithm_type == "dpmsolver++":

        if args.dpm_solver_type == "midpoint":
            prev_mean = ((sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1)
            std_dev_t = sigma_t
            dt_sqrt = torch.sqrt(1.0 - torch.exp(-2 * h))
        elif args.dpm_solver_type == "heun":
            prev_mean = ((sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1)
            std_dev_t = sigma_t
            dt_sqrt = torch.sqrt(1.0 - torch.exp(-2 * h))


        if sde_solver:
            assert noise is not None
            if args.dpm_solver_type == "midpoint":
                x_t = prev_mean + std_dev_t  * dt_sqrt * noise
            elif args.dpm_solver_type == "heun":
                x_t = prev_mean + std_dev_t * dt_sqrt * noise
        else:
            # See https://huggingface.co/papers/2211.01095 for detailed derivations
            if args.dpm_solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif args.dpm_solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
    elif args.dpm_algorithm_type == "dpmsolver":
        if args.dpm_solver_type == "midpoint":
            prev_mean = ((alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1)
            std_dev_t = sigma_t
            dt_sqrt = torch.sqrt(torch.exp(2 * h) - 1.0)
        elif args.dpm_solver_type == "heun":
            prev_mean = ((alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1)
            std_dev_t = sigma_t
            dt_sqrt = torch.sqrt(torch.exp(2 * h) - 1.0)
        if sde_solver:
            assert noise is not None
            if args.dpm_solver_type == "midpoint":
                x_t = prev_mean + std_dev_t * dt_sqrt * noise
            elif args.dpm_solver_type == "heun":
                x_t = prev_mean + std_dev_t * dt_sqrt * noise
        else:
            # See https://huggingface.co/papers/2206.00927 for detailed derivations
            if args.dpm_solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif args.dpm_solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )

    return x_t, prev_mean, std_dev_t, dt_sqrt

def multistep_dpm_solver_third_order_update(
    args,
    model_output_list: List[torch.Tensor],
    sigmas,
    step_index,
    sample: torch.Tensor = None,
    noise: Optional[torch.Tensor] = None,
    sde_solver: bool = False,
) -> torch.Tensor:
    """
    One step for the third-order multistep DPMSolver.

    Args:
        model_output_list (`List[torch.Tensor]`):
            The direct outputs from learned diffusion model at current and latter timesteps.
        sample (`torch.Tensor`):
            A current instance of a sample created by diffusion process.

    Returns:
        `torch.Tensor`:
            The sample tensor at the previous timestep.
    """

    sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
        sigmas[step_index - 2],
    )

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)
    alpha_s2, sigma_s2 = _sigma_to_alpha_sigma_t(sigma_s2)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

    m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

    h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
    r0, r1 = h_0 / h, h_1 / h
    D0 = m0
    D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
    if args.dpm_algorithm_type == "dpmsolver++":
        prev_mean = ((sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1.0 - torch.exp(-2.0 * h))) * D0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h) - 2.0 * h) / (2.0 * h) ** 2 - 0.5)) * D2)
        std_dev_t = sigma_t
        dt_sqrt = torch.sqrt(1.0 - torch.exp(-2 * h))
        if sde_solver:
            assert noise is not None
            x_t = prev_mean + std_dev_t * dt_sqrt * noise
        else:
            # See https://huggingface.co/papers/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
    elif args.dpm_algorithm_type == "dpmsolver":
        assert not sde_solver, "SDE solver is not supported for DPMSolver"
        # See https://huggingface.co/papers/2206.00927 for detailed derivations
        x_t = (
            (alpha_t / alpha_s0) * sample
            - (sigma_t * (torch.exp(h) - 1.0)) * D0
            - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
            - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
        )

    return x_t, prev_mean, std_dev_t, dt_sqrt

def _sigma_to_alpha_sigma_t(sigma):
    alpha_t = 1 - sigma
    sigma_t = sigma
    return alpha_t, sigma_t
