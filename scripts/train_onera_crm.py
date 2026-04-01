from data.load_onera_crm import load_onera_crm
from fluidFlow.dit import DiT
from fluidFlow.trainer import Trainer
from fluidFlow.flow_matching import create_flow_matching

import torch


# you need first to download the data --> https://entrepot.recherche.data.gouv.fr/file.xhtml?persistentId=doi:10.57745/Z9LDY8&version=2.0
data_dir = "/path/to/onera_data"

dataset_train, dataset_test, coefficients = load_onera_crm(data_dir)

model = DiT(
    depth=12,
    hidden_size=256,
    patch_size=1,
    num_heads=8,
    input_size=dataset_train.tensors[0].shape[2],
    cond_dim=dataset_train.tensors[1].shape[1],
    class_dropout_prob=0.2,
    in_channels=dataset_train.tensors[0].shape[1],
    use_swiglu=True,
    attn_type="linear",
    qk_norm=True, # to avoid stability issues with bf16
    mlp_ratio=4,
    use_rope=True
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

flow_matching = create_flow_matching(
    neural_net=model,
    input_size=dataset_train.tensors[0].shape[2],
    cond_scale=6.0,
    sampling_method="euler",
    num_sampling_steps=500,
)

results_folder = 'results/onera_crm_experiment'
train_steps = 300000
trainer = Trainer(
    flow_matching,
    dataset=dataset_train,
    # dataset_test=dataset_test,
    train_batch_size=2,
    train_lr=2e-4,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=16,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp = True,                       # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    compile_model=True, 
    split_batches=True
)

trainer.train()

# Evaluate the model on the test set and save predictions
trainer.ema.ema_model.eval()
# this will sample with multiple gpus, if available
samples, seqs = trainer.eval_model(dataset_test, batch_size=16, use_autocast=True)

if trainer.accelerator.is_main_process:
    samples = (samples.cpu() * coefficients['train_std']) + coefficients['train_mean']
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")
    test_data = (dataset_test.tensors[0].cpu() * coefficients['train_std']) + coefficients['train_mean']
    mse = torch.mean((samples - test_data) ** 2)
    mae = torch.mean(torch.abs(samples - test_data))
    print(f"Test MSE: {mse.item()}, Test MAE: {mae.item()}")