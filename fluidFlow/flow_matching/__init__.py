from .transport import Transport, ModelType, WeightType, PathType, Sampler, FlowMatching

def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    use_cosine_loss=None,
    use_lognorm=None,
    partitial_train=None,
    partial_ratio=1.0,
    shift_lg=False,
    equilibrium_matching=False,
    energy_formulation="l2"
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else: # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0
    
    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        use_cosine_loss=use_cosine_loss,
        use_lognorm=use_lognorm,
        partitial_train=partitial_train,
        partial_ratio=partial_ratio,
        shift_lg=shift_lg,
        equilibrium_matching=equilibrium_matching,
        energy_formulation=energy_formulation
    )
    
    return state

def create_flow_matching(
    neural_net,
    input_size,
    cond_scale=1,
    shifted_mu=0, 
    sampler_atol=1e-6,
    sampler_rtol=1e-3,
    num_sampling_steps=50,
    sampler_timestep_shift=0.0,
    sampling_method="euler",
    reverse_sampling=False,
    **transport_kwargs
):
    """function for creating FlowMatching object
    **Note**: model prediction defaults to velocity
    Args:
    - neural_net: the neural network to be trained
    - input_size: the size of the input data
    - cond_scale: the scale of the conditioning (if any)
    - shifted_mu: the amount of shift for the mean in the loss calculation
    - sampler_atol: the absolute tolerance for the sampler
    - sampler_rtol: the relative tolerance for the sampler
    - num_sampling_steps: the number of sampling steps to use during training
    - sampler_timestep_shift: the amount of shift for the timestep during sampling
    - sampling_method: the method to use for sampling (euler or dopri5)
    - reverse_sampling: whether to reverse the sampling direction (from x_T to x_0 instead of x_0 to x_T)
    - transport_kwargs: the keyword arguments for creating the transport object
    """

    transport = create_transport(**transport_kwargs)
    sampler = Sampler(transport)
    flow_matching = FlowMatching(
        sampler,
        neural_net,
        input_size,
        cond_scale,
        shifted_mu,
        sampler_atol,
        sampler_rtol,
        num_sampling_steps,
        sampler_timestep_shift,
        sampling_method,
        reverse_sampling,
    )

    return flow_matching