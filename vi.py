import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generator import generate_system
from methods import precond_kaczmarz

sns.set(style="whitegrid")

# =========================
# Cenários pequenos para visualização
# =========================
cenarios = [
    {"nome": "Sobredeterminado-cod", "m": 10, "n": 2, "sigma_min": 1, "sigma_max": 1.1},
    {"nome": "Sobredeterminado-ncod", "m": 10, "n": 2, "sigma_min": 0.1, "sigma_max": 1},
]

# Parâmetros do Kaczmarz
max_iter = 50
log_interval = 1
metodos = ["uniform", "cyclic"]
fatores_sketch = [0, 4]  # 0 = sem sketch
seed = 32

cores = {0: "black", 4: "green"}
marcadores = {"uniform": "o", "cyclic": "x"}

# =========================
# Loop sobre cenários
# =========================
for cen in cenarios:
    print(f"Executando cenário: {cen['nome']}")
    A, x_true, b, x0 = generate_system(
        cen["m"], cen["n"], cen["sigma_min"], cen["sigma_max"], seed=seed
    )

    fig, axes = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)
    x_vals = np.linspace(-3, 3, 100)
    
    for ax_idx, metodo in enumerate(metodos):
        ax = axes[ax_idx]
        
        # Desenha as linhas do sistema (apenas se n=2)
        if cen["n"] == 2:
            for i in range(cen["m"]):
                if A[i,1] != 0:
                    y_vals = (b[i] - A[i,0]*x_vals)/A[i,1]
                    ax.plot(x_vals, y_vals, 'gray', alpha=0.5)
        
        # Loop fatores de sketch
        for fator in fatores_sketch:
            if fator == 0:
                label = f"sem sketch"
                use_precond = False
                sf = None
                linestyle = '-'  # linha contínua
            else:
                label = f"sketch (fator={fator})"
                use_precond = True
                sf = fator
                linestyle = '--'  # linha tracejada

            res = precond_kaczmarz(
                A, b, x0=x0.copy(), max_iter=max_iter,
                log_interval=log_interval,
                method=metodo,
                use_precond=use_precond,
                sketch_factor=sf,
                seed=seed
            )
            X_hist = np.array(res["conv_x"])
            if cen["n"] == 2:
                ax.plot(
                    X_hist[:,0], X_hist[:,1],
                    color=cores[fator],
                    linestyle=linestyle,
                    marker=marcadores[metodo],
                    label=label
                )

        ax.set_title(f"Kaczmarz {metodo}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.axis('square')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.legend(fontsize=9)

    plt.suptitle(f"Trajetória do Kaczmarz: {cen['nome']}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"trajetoria_{cen['nome'].replace(' ', '_')}.png")
    plt.show()
