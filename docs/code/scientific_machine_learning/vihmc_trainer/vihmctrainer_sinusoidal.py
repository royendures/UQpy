"""
Training a Bayesian neural network with VI-HMC
=============================================================

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml
import torch.nn.functional as F
import math
import logging

# logger = logging.getLogger("UQpy")  # Optional, display UQpy logs to console
# logger.setLevel(logging.INFO)
torch.manual_seed(123)

"""
VI training
=============================================================
"""


class SinusoidalDataset(Dataset):
    def __init__(self, n_samples=300, noise=1e-3, train=True):
        self.n_samples = n_samples
        self.noise = noise
        self.train = train
        self.x = (
            torch.cat(
                (
                    torch.linspace(-1.0, -0.2, n_samples // 2, dtype=torch.float),
                    torch.linspace(0.2, 1.0, n_samples // 2, dtype=torch.float),
                )
            ).reshape(-1, 1)
            if train
            else torch.linspace(-1.2, 1.2, n_samples, dtype=torch.float).view(-1, 1)
        )
        self.y = (
                4 * torch.sin(4 * self.x)
                + 5 * torch.cos(12 * self.x)
                + torch.randn_like(self.x) * self.noise
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class GaussianNLLLoss(nn.Module):
    def __init__(self, var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var

    def forward(self, prediction, target):
        assert (
                prediction.shape == target.shape
        ), "Prediction does not match target shape in the loss function"
        return F.gaussian_nll_loss(
            prediction, target, torch.ones(prediction.shape) * self.var, reduction="sum"
        )


def post_process_vi():
    # %% md
    #
    # We compare the initial and final predictions and plot the loss history using matplotlib.

    # %%

    x = train_dataset.x
    y = train_dataset.y
    model.train(False)
    model.sample(False)
    final_prediction = model(x)
    fig, ax = plt.subplots()
    ax.plot(
        x.detach().numpy(),
        initial_prediction.detach().numpy(),
        label="Initial Prediction",
        color="tab:blue",
    )
    ax.plot(
        x.detach().numpy(),
        final_prediction.detach().numpy(),
        label="Final Prediction",
        color="tab:orange",
    )
    ax.plot(
        x.detach().numpy(),
        y.detach().numpy(),
        label="Exact",
        color="black",
        linestyle="dashed",
    )
    ax.set_title("Initial and Final NN Predictions")
    ax.set(xlabel="x", ylabel="f(x)")
    ax.legend()

    train_loss = trainer.history["train_loss"].detach().numpy()
    fig, ax = plt.subplots()
    ax.semilogy(train_loss)
    ax.set_title("Bayes By Backpropagation Training Loss")
    ax.set(xlabel="Epoch", ylabel="Loss")

    plt.show()

    # %% md
    # The Bayesian neural network is a probabilistic model. Each of its parameters, in this case weights and biases,
    # are governed by Gaussian distributions. We can get a deterministic output from the BNN by setting
    # ``model.sample(False)``, which sets each parameter to the mean of its distribution.
    #
    # We can obtain error bars on model's output by sampling the parameters from their governing distribution.
    # This is done by setting ``model.sample(True)`` and computing the forward model evaluation many times,
    # then computing the sample variance

    # %%

    x_test = test_dataset.x
    y_test = test_dataset.y
    model.sample(False)
    mean = model(x_test)

    model.sample(True)
    n = 1000
    samples = torch.zeros(len(x_test), n)
    for i in range(n):
        samples[:, i] = model(x_test).squeeze()
    variance = torch.var(samples, dim=1)
    standard_deviation = torch.sqrt(variance)

    x_plot = x_test.squeeze().detach().numpy()
    mu = mean.squeeze().detach().numpy()
    sigma = standard_deviation.squeeze().detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(x_plot, samples.detach().numpy(), "C0", alpha=0.051)
    ax.plot(
        x_plot,
        y_test.detach().numpy(),
        label="Exact",
        color="red",
        linestyle="dashed",
    )
    ax.plot(x_plot, mu, "k", label="$\mu$", linewidth=3)
    # ax.fill_between(
    #     x_plot,
    #     mu - (3 * sigma),
    #     mu + (3 * sigma),
    #     label="$\mu \pm 3\sigma$,",
    #     alpha=0.3,
    # )
    ax.plot(x, y.detach().numpy(), ".C3", markersize=30, label="x train", alpha=0.6)
    ax.set_title("Bayesian Neural Network predictions")
    ax.set(xlabel="x", ylabel="f(x)")
    ax.legend()
    plt.savefig("VI_prediction.pdf")
    plt.show()


prior_sigma = 1.0
width = 10
network = nn.Sequential(
    sml.BayesianLinear(1, width, prior_sigma=prior_sigma, posterior_rho_initial=(0.2351, 0.1)),
    nn.Tanh(),
    sml.BayesianLinear(width, width, prior_sigma=prior_sigma, posterior_rho_initial=(0.2351, 0.1)),
    nn.Tanh(),
    sml.BayesianLinear(width, 1, prior_sigma=prior_sigma, posterior_rho_initial=(0.2351, 0.1)),
)
model = sml.FeedForwardNeuralNetwork(network)

data_noise = 5e-2
train_dataset = SinusoidalDataset(n_samples=20, noise=data_noise)
test_dataset = SinusoidalDataset(n_samples=300, noise=0.0, train=False)
initial_prediction = model(train_dataset.x)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5_000, min_lr=1e-5, verbose=True)
trainer = sml.BBBTrainer(
    model, optimizer, loss_function=GaussianNLLLoss(var=data_noise ** 2), scheduler=lr_sched)
print("Starting Training...", end="")
trainer.run(
    train_data=train_dataloader,
    test_data=test_dataloader,
    epochs=30_000,
    beta=1 / len(train_dataloader),
    num_samples=10,
)
print("done")

print("Initial loss:", trainer.history["train_loss"][0])
print("Final loss:", trainer.history["train_loss"][-1])
model.summary()
post_process_vi()

"""
VI-HMC sampling
=============================================================
"""
det_network = nn.Sequential(
    nn.Linear(1, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, 1),
)

mean_params = []
for name, param in model.named_parameters():
    if "mu" in name:
        mean_params.append(param.flatten())

mean_params = torch.cat(mean_params)

# HMC Params
step_size = 2e-4
num_samples = 3_000
burn = num_samples // 5
post_var = 0.2024 ** 2
L = max(1, int(math.pi * post_var / (2 * step_size)))
loss = "NLL"  # regression or NLL
tau_out = (
        data_noise ** 2
)  # Measure of precision: 1/variance if Regression or variance if NLL

vihmc_trainer = sml.VIHMCTrainer(det_model=det_network, vi_model=model)
params_hmc, pred_list, _ = vihmc_trainer.run(
    train_data=train_dataloader,
    valid_data=test_dataloader,
    variance_threshold=0.90,
    step_size=step_size,
    num_samples=num_samples,
    burn=burn,
    prior_var=prior_sigma ** 2,
    step_length=L,
    tau_out=tau_out,
    debug=False,
)


def post_process_vihmc():
    x = []
    y = []
    for x_batch, y_batch in test_dataloader:
        y.append(y_batch)
        x.append(x_batch)
    y_val = torch.cat(y)
    x_val = torch.cat(x)

    x = []
    y = []
    for x_batch, y_batch in train_dataloader:
        y.append(y_batch)
        x.append(x_batch)
    y_train = torch.cat(y)
    x_train = torch.cat(x)

    sample_mse = []
    for pred in pred_list:
        assert pred.shape == y_val.shape
        sample_mse.append(((pred - y_val) ** 2).mean())
    print("\nExpected MSE: {:.6f}".format((torch.mean(torch.tensor(sample_mse)))))
    print("\nFinal MSE: {:.6f}".format(sample_mse[-1]))
    print("\nMin MSE:{:.6f}".format(min(sample_mse)))
    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(8, 5))
    plt.plot(
        x_val.cpu().numpy(), pred_list[:].cpu().numpy().squeeze().T, "C0", alpha=0.051
    )
    plt.plot(
        x_val.cpu().numpy(),
        y_val.cpu().numpy(),
        "r",
        linewidth=3,
        label="True function",
    )
    plt.plot(
        x_val.cpu().numpy(),
        pred_list.mean(0).cpu().numpy().squeeze().T,
        "k",
        alpha=0.9,
        linewidth=3,
        label="Mean prediction",
    )

    plt.plot(
        x_train.cpu().numpy(),
        y_train.cpu().numpy(),
        ".C3",
        markersize=30,
        label="x train",
        alpha=0.6,
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("VIHMC_prediction.pdf")
    plt.show()


post_process_vihmc()
