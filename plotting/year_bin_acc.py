import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Use IEEE style
plt.style.use(['science', 'ieee'])
# Disable the requirement for a system LaTeX installation
plt.rcParams.update({
    "text.usetex": False,
})

# Data
MODEL_TYPE = "mae"

layers_0_acc = [0.52, 0.61, 0.78, 0.81]
layers_1_acc = [0.57, 0.64, 0.78, 0.82]
layers_2_acc = [0.56, 0.62, 0.77, 0.82]
layers_4_acc = [0.56, 0.64, 0.8, 0.83]
layers_8_acc = [0.57, 0.7, 0.81, 0.83]

all_layers = [layers_0_acc, layers_1_acc, layers_2_acc, layers_4_acc, layers_8_acc]
labels = ['K=0', 'K=1', 'K=2', 'K=4', 'K=8']
x_groups = np.arange(len(layers_0_acc))  # 4 groups
width = 0.15  # width of each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure
plt.figure(figsize=(6, 4))

# Plot each layer as a set of bars
for i, layer_acc in enumerate(all_layers):
    plt.bar(x_groups + i*width, layer_acc, width=width, label=labels[i], color=colors[i])

# Axis labels and ticks
plt.xlabel('Group', fontsize=14)  # Replace with actual meaning
plt.ylabel('Validation Accuracy', fontsize=14)
plt.xticks(x_groups + 2*width, ['0-5', '5-20', '20-50', '50+'])
plt.gca().tick_params(axis='both', which='major', labelsize=14)

# axis limits
plt.ylim(0.0, 1.0)

# Grid and legend
plt.grid(axis='y')
plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, ncol=2)

plt.tight_layout()
save_path = "/cluster/home/jiapan/Supervised_Research/plots/" + MODEL_TYPE + "/" + f"{MODEL_TYPE}_year_bin_acc.png"
plt.savefig(save_path, dpi=300)
