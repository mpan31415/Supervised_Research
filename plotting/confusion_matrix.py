import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


MODEL_TYPE = "mae"
CKPT_NAME = "epoch_100.pth"

TARGET_LABEL = "is_military"    # OPTIONS: "is_military", "has_cross"

PROBE_TYPE = "partial8"     # OPTIONS: "linear", "nonlinear", "partial(n)"

CONF_MAT_SAVE_NAME = f"{MODEL_TYPE}_{PROBE_TYPE}_{TARGET_LABEL}.png"


################# DATA #################
cm_percent = np.array([
    [18.2, 3.1], 
    [2.5, 76.0]
])

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=["True", "False"],
    yticklabels=["True", "False"],
    cbar=False,
    annot_kws={"size": 40}
)

plt.xlabel("Predicted", fontsize=40)
plt.ylabel("Actual", fontsize=40)

# Move x-axis labels to top
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().tick_params(axis='both', which='major', labelsize=40)

save_path = "/cluster/home/jiapan/Supervised_Research/plots/" + MODEL_TYPE + "/" + CONF_MAT_SAVE_NAME

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Confusion matrix saved to {save_path}")
