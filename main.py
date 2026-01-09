# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generator import generate_system
from methods import precond_kaczmarz

sns.set(style="whitegrid")  # estilo agradável

# =========================
# Configurações dos cenários
# =========================
cenarios = [
    {"nome": "Sobredeterminado, cond=1e5", "m": 250, "n": 30, "sigma_min": 1/1e5, "sigma_max": 1},
    {"nome": "Sobredeterminado, cond~1.1", "m": 250, "n": 30, "sigma_min": 1, "sigma_max": 1 + 1e-1},
    {"nome": "Subdeterminado, cond=1e5", "m": 30, "n": 250, "sigma_min": 1/1e5, "sigma_max": 1},
    {"nome": "Subdeterminado, cond~1.1", "m": 30, "n": 250, "sigma_min": 1, "sigma_max": 1 + 1e-1},
]

# Parâmetros gerais do Kaczmarz
max_iter = 1000
log_interval = 1
metodos = ["uniform", "cyclic"]
fatores_sketch = [0, 2, 4, 8]  # 0 = sem sketch
seed = 42

# Mapas de cores e estilos
cores = {0: "gray", 2: "blue", 4: "green", 8: "red"}  # mesma cor para mesmo fator
estilos = {"uniform": "-", "cyclic": "--"}  # estilos diferentes

# =========================
# Loop sobre cenários
# =========================
resultados = {}

for cen in cenarios:
    print(f"Executando cenário: {cen['nome']}")
    A, x_true, b, x0 = generate_system(
        cen["m"], cen["n"], cen["sigma_min"], cen["sigma_max"], seed=seed
    )

    resultados_cen = {}

    for metodo in metodos:
        for fator in fatores_sketch:
            if fator == 0:
                label = f"{metodo} (sem sketch)"
                print(f"  Método: {label}")
                res = precond_kaczmarz(
                    A, b, x0=x0.copy(), max_iter=max_iter,
                    log_interval=log_interval,
                    method=metodo,
                    use_precond=False,
                    sketch_factor=None,
                    seed=seed
                )
            else:
                label = f"{metodo} + sketch (fator={fator})"
                print(f"  Método: {label}")
                res = precond_kaczmarz(
                    A, b, x0=x0.copy(), max_iter=max_iter,
                    log_interval=log_interval,
                    method=metodo,
                    use_precond=True,
                    sketch_factor=fator,
                    seed=seed
                )

            resultados_cen[label] = {"res": res, "fator": fator, "metodo": metodo}

    resultados[cen["nome"]] = {"A": A, "b": b, "metodos": resultados_cen}

# =========================
# Plotar resultados
# =========================
for cen_nome, cen_dados in resultados.items():
    A_cen = cen_dados["A"]
    b_cen = cen_dados["b"]

    plt.figure(figsize=(10,6))
    for label, info in cen_dados["metodos"].items():
        res = info["res"]
        fator = info["fator"]
        metodo = info["metodo"]
        conv = [np.linalg.norm(A_cen @ x - b_cen) for x in res["conv_x"]]
        plt.semilogy(conv, label=label, color=cores[fator], linestyle=estilos[metodo])

    plt.xlabel("Iterações registradas")
    plt.ylabel("||Ax - b||_2")
    plt.title(f"Convergência: {cen_nome}")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"convergencia_{cen_nome.replace(' ', '_')}.png")
    plt.close()
