from pathlib import Path
import matplotlib.pyplot as plt

def save_fig(folder,name, instance, extension=".png"):
    fold = Path(folder)
    fname = f"{instance.type_graph}_n_{instance.n}_d_{instance.d}_t_{instance.t_limit}_lam_{instance.lambda_}_"
    fname += name + extension
    plt.savefig(fold / fname, bbox_inches="tight")