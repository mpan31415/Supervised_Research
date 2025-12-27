import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# use IEEE style
plt.style.use(['science', 'ieee'])
# disable the requirement for a system LaTeX installation
plt.rcParams.update({
    "text.usetex": False,
})

x = [0, 1, 2, 4, 8]

# # MAE Data
# MODEL_TYPE = "mae"
# is_military_vec = [0.908, 0.931, 0.932, 0.935, 0.943]
# has_cross_vec = [0.743, 0.825, 0.826, 0.828, 0.819]
# year_rank_vec = [0.715, 0.725, 0.731, 0.73, 0.747]

# DINO Data
MODEL_TYPE = "dino"
is_military_vec = [0.928, 0.935, 0.936, 0.94, 0.955]
has_cross_vec = [0.715, 0.751, 0.784, 0.833, 0.841]
year_rank_vec = [0.752, 0.756, 0.759, 0.768, 0.774]


# create figure
plt.figure(figsize=(6, 4))

# marker size
ms = 8

# plot lines with markers
lines = [
    plt.plot(x, is_military_vec, marker='o', markersize=ms, label='is_military', linewidth=2),
    plt.plot(x, has_cross_vec, marker='s', markersize=ms, label='has_cross', linewidth=2),
    plt.plot(x, year_rank_vec, marker='^', markersize=ms, label='death_year', linewidth=2)
]

# annotate each point with exact value
for xi, yi in zip(x, is_military_vec):
    plt.text(xi, yi + 0.005, f"{yi:.3f}", ha='center', va='bottom', fontsize=12)
for xi, yi in zip(x, has_cross_vec):
    plt.text(xi, yi + 0.005, f"{yi:.3f}", ha='center', va='bottom', fontsize=12)
for xi, yi in zip(x, year_rank_vec):
    plt.text(xi, yi + 0.005, f"{yi:.3f}", ha='center', va='bottom', fontsize=12)

# axis labels
plt.xlabel('Number of Finetuned Layers', fontsize=16)
plt.ylabel('Validation Accuracy', fontsize=16)

# x-axis ticks
plt.xticks(x)
plt.gca().tick_params(axis='both', which='major', labelsize=14)

# axis limits
plt.ylim(0.6, 1.0)

# grid and legend
plt.grid(True)
plt.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=12)

plt.tight_layout()

save_path = "/cluster/home/jiapan/Supervised_Research/plots/" + MODEL_TYPE + "/" + f"{MODEL_TYPE}_val_accuracy.png"
plt.savefig(save_path, dpi=300)
