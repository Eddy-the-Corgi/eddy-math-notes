import numpy as np
import matplotlib.pyplot as plt

'''Goal: Verify Conditional-Mean Agreement for FV-QV-StoZINIT-K-ReLU with Squared Loss.'''

def relu(t):
    return np.maximum(t, 0.0)

# Round each real number component s = l + f down to s with probability 1 - f and up to s + 1 with probability f.
def stochastic_round(s, rng):
    lower = np.floor(s)
    fractional_part = s - lower
    round_up = (rng.random(np.shape(s)) < fractional_part)
    return lower + round_up

# Draw the shared Gaussian sources for one width M.
def initialise_sources(M, d_in, d_out, sigma_u, sigma_v, rng):
    u0 = rng.normal(0.0, sigma_u, size=(M, d_in))
    v0 = rng.normal(0.0, sigma_v, size=(M, d_out))
    return u0, v0

# Compute each neuron's frozen features at training input x and test input z.
def frozen_features(u0, x, z, activation=relu):
    a_x = activation(u0 @ x)  # Shape: (M,)
    a_z = activation(u0 @ z)  # Shape: (M,)
    return a_x, a_z

def train_frozen_features(
    v0,
    a_x,
    a_z,
    y,
    eta_v,
    K,
    *,
    quantised=False,
    rng=None,
):

    if quantised and rng is None:
        raise ValueError("A random generator is required for quantised training.")

    M, d_out = v0.shape

    # Q model is rounded at initialisation; FP model is not.
    v = stochastic_round(v0, rng) if quantised else v0.copy()

    predictions = np.empty((K + 1, d_out))

    for k in range(K + 1):
        # Compute the current prediction for the test input z.
        predictions[k] = np.mean(v * a_z[:, None], axis=0)

        if k == K:
            break
        
        # Compute the current prediction for the training input x.
        prediction_x = np.mean(v * a_x[:, None], axis=0)
        error = prediction_x - y

        # All proposals use the same current weights and error.
        proposal = v - eta_v * a_x[:, None] * error[None, :]

        v = stochastic_round(proposal, rng) if quantised else proposal

    return predictions

# Experiment settings
M = 100
K = 30
eta_v = 1.0
sigma_u = 1.0
sigma_v = 1.0

x = np.array([1.0])
z = np.array([1.0])
y = np.array([1.0])

# Separate generators make it clear which randomness is being varied.
source_rng = np.random.default_rng(1337)
rounding_rng = np.random.default_rng(888)

u0, v0 = initialise_sources(
    M=M,
    d_in=x.size,
    d_out=y.size,
    sigma_u=sigma_u,
    sigma_v=sigma_v,
    rng=source_rng,
)

a_x, a_z = frozen_features(u0, x, z)

fp_predictions = train_frozen_features(
    v0, a_x, a_z, y, eta_v, K
)

q_predictions = train_frozen_features(
    v0, a_x, a_z, y, eta_v, K,
    quantised=True,
    rng=rounding_rng,
)

print("FP predictions:", fp_predictions[:, 0])
print("Q predictions: ", q_predictions[:, 0])

# Repeat the quantised trajectory with fixed Gaussian sources and fresh rounding.
B = 1000
visible_paths = 20

# Train the FV-QV model B times with the same frozen features and different rounding.
q_trajectories = np.stack([
    train_frozen_features(
        v0, a_x, a_z, y, eta_v, K,
        quantised=True,
        rng=rounding_rng,
    )
    for _ in range(B)
])

# Every mean Q predictor coordinate agrees with the corresponding FP coordinate. We plot one coordinate for simplicity.
discrepancies = q_trajectories[:, :, 0] - fp_predictions[None, :, 0]
mean_discrepancy = discrepancies.mean(axis=0)

# Calculate the standard error for a confidence interval.
standard_error = discrepancies.std(axis=0, ddof=1) / np.sqrt(B)
steps = np.arange(K + 1)

# Diagram 1: Conditional-Mean Agreement for FV-QV with Squared Loss and ReLU.
# Show individual paths and the pointwise Monte Carlo uncertainty in their mean.
fig, ax = plt.subplots()
for path in discrepancies[:visible_paths]:
    ax.plot(steps, path, color="tab:blue", alpha=0.2, linewidth=1)

ax.axhline(0, color="black", linestyle="--", label="FP reference")
ax.plot(steps, mean_discrepancy, color="tab:red", linewidth=2, label="Mean Q discrepancy")
ax.fill_between(
    steps,
    mean_discrepancy - 1.96 * standard_error,
    mean_discrepancy + 1.96 * standard_error,
    color="tab:red",
    alpha=0.2,
    label="Approximate 95% pointwise interval",
)
ax.set_xlabel("Training step k")
ax.set_ylabel("Q prediction - FP prediction")
ax.set_title(f"Fixed initialisation; repeated rounding (M={M})")
ax.legend()
fig.tight_layout()
plt.show()

'''Goal: Verify Finite-Horizon Agreement with Growing Width for FV-QV-StoZINIT-K-ReLU with Squared Loss.'''

widths = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
width_repetitions = 200
width_source_rng = np.random.default_rng(67)
width_rounding_rng = np.random.default_rng(6767)

# Store the maximum prediction discrepancy over the whole training horizon.
maximum_discrepancies = np.empty((len(widths), width_repetitions))
for i, width in enumerate(widths):
    for b in range(width_repetitions):
        width_u0, width_v0 = initialise_sources(
            width, x.size, y.size, sigma_u, sigma_v, width_source_rng
        )
        width_a_x, width_a_z = frozen_features(width_u0, x, z)
        width_fp = train_frozen_features(
            width_v0, width_a_x, width_a_z, y, eta_v, K
        )
        width_q = train_frozen_features(
            width_v0, width_a_x, width_a_z, y, eta_v, K,
            quantised=True, rng=width_rounding_rng,
        )
        differences = width_q - width_fp
        maximum_discrepancies[i, b] = np.max(np.linalg.norm(differences, axis=1))

# Estimate a typical discrepancy and the middle 90% of sampled discrepancies at each width.
rms_discrepancies = np.sqrt(np.mean(maximum_discrepancies ** 2, axis=1))
lower_90 = np.quantile(maximum_discrepancies, 0.05, axis=1)
upper_90 = np.quantile(maximum_discrepancies, 0.95, axis=1)

# Diagram 2: Finite-Horizon Agreement for FV-QV with Squared Loss and ReLU.
# Compare the observed width dependence with a reference slope of -1/2, anchored at M=64.
# Log-log plot, since D = CM^{-1/2} implies log(D) = log(C) - (1/2) log(M).
width_fig, width_ax = plt.subplots()
width_ax.loglog(widths, rms_discrepancies, "o-", label="RMS maximum discrepancy")
width_ax.fill_between(widths, lower_90, upper_90, alpha=0.2, label="Middle 90% of runs")
reference_index = np.where(widths == 64)[0][0]
width_ax.loglog(
    widths, rms_discrepancies[reference_index] * np.sqrt(widths[reference_index] / widths), "--",
    label="Slope -1/2 reference",
)
width_ax.set_xlabel("Width M")
width_ax.set_ylabel("Maximum Q - FP prediction norm over k")
width_ax.set_title(f"Finite-horizon agreement (K={K})")
width_ax.legend()
width_fig.tight_layout()
plt.show()

'''Goal: Illustrate Conditional-Mean and Finite-Horizon Agreement Failure for V-QV-StoZINIT-K-ReLU with Squared Loss.'''

# Train both layers with simultaneous updates, rounding only the Q output weights.
def train_both_layers(u0, v0, x, z, y, eta, K, quantised=False, rng=None):
    if quantised and rng is None:
        raise ValueError("A random generator is required for quantised training.")

    u = u0.copy()
    v = stochastic_round(v0, rng) if quantised else v0.copy()
    predictions = np.empty((K + 1, y.size))

    for k in range(K + 1):
        predictions[k] = np.mean(v * relu(u @ z)[:, None], axis=0)
        if k == K:
            break

        preactivation = u @ x
        error = np.mean(v * relu(preactivation)[:, None], axis=0) - y
        v_proposal = v - eta * relu(preactivation)[:, None] * error[None, :]
        u_proposal = u - eta * (v @ error)[:, None] * (preactivation > 0)[:, None] * x[None, :]
        v = stochastic_round(v_proposal, rng) if quantised else v_proposal
        u = u_proposal

    return predictions

# Use the two-dimensional diagnostic from the V-QV counterexample.
trainable_x = np.array([1.0, 1.0]) / np.sqrt(2.0)
trainable_z = trainable_x.copy()
trainable_y = np.array([1.0, 0.0])
trainable_eta = 1.0
trainable_K = 10
trainable_M = 100
trainable_repetitions = 5000
trainable_source_rng = np.random.default_rng(1337)
trainable_rounding_rng = np.random.default_rng(888)

# Fix the Gaussian sources and repeat only the Q rounding for conditional-mean agreement.
trainable_u0, trainable_v0 = initialise_sources(
    trainable_M, 2, 2, 1.0, 1.0, trainable_source_rng
)
trainable_fp = train_both_layers(
    trainable_u0, trainable_v0, trainable_x, trainable_z, trainable_y, trainable_eta, trainable_K
)
trainable_q = np.stack([
    train_both_layers(
        trainable_u0, trainable_v0, trainable_x, trainable_z, trainable_y,
        trainable_eta, trainable_K, quantised=True, rng=trainable_rounding_rng,
    )
    for _ in range(trainable_repetitions)
])
trainable_differences = trainable_q[:, :, 0] - trainable_fp[None, :, 0]
trainable_mean = trainable_differences.mean(axis=0)
trainable_se = trainable_differences.std(axis=0, ddof=1) / np.sqrt(trainable_repetitions)
trainable_steps = np.arange(trainable_K + 1)

# Diagram 3: Conditional-Mean Agreement Failure for V-QV with Squared Loss and ReLU.
# Plot the conditional mean with a pointwise Monte Carlo interval.
trainable_fig, trainable_ax = plt.subplots()
for path in trainable_differences[:visible_paths]:
    trainable_ax.plot(trainable_steps, path, color="tab:blue", alpha=0.2, linewidth=1)
trainable_ax.axhline(0, color="black", linestyle="--", label="FP reference")
trainable_ax.plot(trainable_steps, trainable_mean, "o-", color="tab:red", label="Mean Q discrepancy")
trainable_ax.fill_between(
    trainable_steps, trainable_mean - 1.96 * trainable_se, trainable_mean + 1.96 * trainable_se,
    color="tab:red", alpha=0.2, label="Approximate 95% pointwise interval",
)
trainable_ax.set_xticks(trainable_steps)
trainable_ax.set_xlabel("Training step k")
trainable_ax.set_ylabel("Q prediction - FP prediction")
trainable_ax.set_title(f"V-QV: Fixed initialisation; repeated rounding (M={trainable_M})")
trainable_ax.legend()
trainable_fig.tight_layout()
plt.show()

# Redraw sources and rounding at each width to study finite-horizon agreement.
trainable_widths = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
trainable_width_K = 1
trainable_width_repetitions = 300
trainable_width_source_rng = np.random.default_rng(3000)
trainable_width_rounding_rng = np.random.default_rng(3001)
trainable_maxima = np.empty((len(trainable_widths), trainable_width_repetitions))
trainable_signed = np.empty_like(trainable_maxima)

for i, width in enumerate(trainable_widths):
    for b in range(trainable_width_repetitions):
        width_u0, width_v0 = initialise_sources(width, 2, 2, 1.0, 1.0, trainable_width_source_rng)
        width_fp = train_both_layers(
            width_u0, width_v0, trainable_x, trainable_z, trainable_y, trainable_eta, trainable_width_K
        )
        width_q = train_both_layers(
            width_u0, width_v0, trainable_x, trainable_z, trainable_y, trainable_eta, trainable_width_K,
            quantised=True, rng=trainable_width_rounding_rng,
        )
        difference = width_q - width_fp
        trainable_maxima[i, b] = np.max(np.linalg.norm(difference, axis=1))
        trainable_signed[i, b] = difference[1, 0]

# Show both the maximum discrepancy and the signed first-step counterexample.
trainable_rms = np.sqrt(np.mean(trainable_maxima ** 2, axis=1))
trainable_lower = np.quantile(trainable_maxima, 0.05, axis=1)
trainable_upper = np.quantile(trainable_maxima, 0.95, axis=1)
signed_mean = trainable_signed.mean(axis=1)
signed_se = trainable_signed.std(axis=1, ddof=1) / np.sqrt(trainable_width_repetitions)

# Diagram 4: Finite-Horizon Agreement Failure for V-QV with Squared Loss and ReLU.
trainable_width_fig, trainable_width_ax = plt.subplots()
trainable_width_ax.loglog(trainable_widths, trainable_rms, "o-", label="RMS maximum discrepancy")
trainable_width_ax.fill_between(
    trainable_widths, trainable_lower, trainable_upper, alpha=0.2, label="Middle 90% of runs"
)
trainable_width_ax.set_xlabel("Width M")
trainable_width_ax.set_ylabel("Maximum Q - FP prediction norm over k")
trainable_width_ax.set_title("Trainable inputs; finite-horizon discrepancy")
trainable_width_ax.legend()
trainable_width_fig.tight_layout()

# Diagram 5: First-Step Discrepancy for the V-QV Counterexample.
signed_fig, signed_ax = plt.subplots()
signed_ax.errorbar(
    trainable_widths, signed_mean, yerr=1.96 * signed_se, fmt="o-", capsize=3,
    label="Mean signed discrepancy; approximate 95% interval",
)
signed_ax.axhline(0.0615366, color="tab:red", linestyle="--", label="Predicted limit: 0.0615366")
signed_ax.axhline(0, color="black", linewidth=0.8)
signed_ax.set_xscale("log")
signed_ax.set_xlabel("Width M")
signed_ax.set_ylabel("First-coordinate Q - FP at k=1")
signed_ax.set_title("Trainable inputs; first-step discrepancy")
signed_ax.legend()
signed_fig.tight_layout()
plt.show()
