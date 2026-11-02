
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. PARAMETROS GENERALES

N = 200000

# Punto inicial
p0 = np.array([0.5, 0.0])

# Colores
color_copa = 'deeppink'
color_tronco = 'saddlebrown'

# 2. ALGORITMO DE BARNSLEY

# La recurrencia 

def barnsley(N, random_numbers, p0):

    X = np.zeros(N)
    Y = np.zeros(N)

    x = p0[0]
    y = p0[1]

    for n in range(N):

        r = random_numbers[n]

        if r < 0.10:
            xn = 0.05*x
            yn = 0.6*y

        elif r < 0.20:
            xn = 0.05*x
            yn = -0.5*y + 1

        elif r < 0.40:
            xn = 0.46*x - 0.15*y
            yn = 0.39*x + 0.38*y + 0.6

        elif r < 0.60:
            xn = 0.47*x - 0.15*y
            yn = 0.17*x + 0.42*y + 1.1

        elif r < 0.80:
            xn = 0.43*x + 0.28*y
            yn = -0.25*x + 0.45*y + 1

        else:
            xn = 0.42*x + 0.26*y
            yn = -0.35*x + 0.31*y + 0.7

        x = xn
        y = yn

        X[n] = x
        Y[n] = y

    return X, Y


# Generacion de numeros aleatorios
random_numbers = np.random.random(N)

# Generacion del fractal
X, Y = barnsley(N, random_numbers, p0)

# 3. CLASIFICACION DE LOS PUNTOS

y_tronco = 0.6

# Mascara booleana vectorizada
tronco = Y < y_tronco

# 4. GRAFICA DEL FRACTAL

fig, ax = plt.subplots(figsize=(6, 9))

ax.scatter(
    X[~tronco], Y[~tronco],
    s=0.1, color=color_copa
)

ax.scatter(
    X[tronco], Y[tronco],
    s=0.1, color=color_tronco
)

ax.text(
    0.03, 0.97,
    rf'$N = {N}$',
    transform=ax.transAxes,
    ha='left', va='top',
    fontsize=12
)

ax.axis('off')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("ferm.pdf", bbox_inches="tight", dpi=800)
plt.show()

# 5. BOX COUNTING

# Cuadrado que contiene el fractal

xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()

L = max(xmax - xmin, ymax - ymin)

xc = 0.5*(xmax + xmin)
yc = 0.5*(ymax + ymin)

x0 = xc - L/2
y0 = yc - L/2

# Escalas de analisis
k_values = np.arange(2, 17)
k_plot = np.arange(2, 10)

n_boxes = []
N_eps = []

# Asignacion vectorizada de puntos a cajas

def box_indices(n):

    ix = np.clip(
        ((X - x0)/L*n).astype(np.int64),
        0, n-1
    )

    iy = np.clip(
        ((Y - y0)/L*n).astype(np.int64),
        0, n-1
    )

    return ix, iy


# Conteo vectorizado de cajas ocupadas

def count_boxes(n):

    ix, iy = box_indices(n)

    # Cada caja recibe un identificador unico
    indices = ix*n + iy

    # Eliminar identificadores repetidos
    return np.unique(indices).size


# Matriz de ocupacion para las graficas

def occupied_grid(n):

    ix, iy = box_indices(n)

    grid = np.zeros((n, n), dtype=bool)

    grid[iy, ix] = True

    return grid


# 6. ANALISIS DE LAS DIFERENTES ESCALAS

for k in k_values:

    n = 2**k

    # Conteo de cajas ocupadas
    Nb = count_boxes(n)

    n_boxes.append(n)
    N_eps.append(Nb)

    # Dimension estimada para cada escala
    Df_k = np.log(Nb)/np.log(n)

    # ---------------------------------------------
    # Graficas de las cajas ocupadas
    # ---------------------------------------------

    if k in k_plot:

        grid = occupied_grid(n)

        fig, ax = plt.subplots(figsize=(6, 6))

        cmap = ListedColormap([
            (0, 0, 0, 0),
            (0.76, 0.87, 0.93, 1)
        ])

        ax.imshow(
            grid,
            origin='lower',
            extent=[x0, x0+L, y0, y0+L],
            cmap=cmap,
            interpolation='nearest',
            zorder=0
        )

        # Cuadricula mediante ticks menores
        ax.set_xticks(
            np.linspace(x0, x0+L, n+1),
            minor=True
        )

        ax.set_yticks(
            np.linspace(y0, y0+L, n+1),
            minor=True
        )

        ax.grid(
            which='minor',
            color='gray',
            linewidth=0.6 if n <= 64 else 0.25,
            alpha=0.5
        )

        # Fractal
        ax.scatter(
            X[~tronco], Y[~tronco],
            s=0.1, color=color_copa,
            linewidths=0, rasterized=True,
            zorder=2
        )

        ax.scatter(
            X[tronco], Y[tronco],
            s=0.1, color=color_tronco,
            linewidths=0, rasterized=True,
            zorder=2
        )

        # Informacion de la escala
        ax.text(
            0.03, 0.97,
            rf'$N = {N}$' + '\n' +
            rf'$n = {n},\ N(\epsilon) = {Nb}$' + '\n' +
            rf'$D_f = {Df_k:.6f}$',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.8
            ),
            zorder=3
        )

        ax.set_xlim(x0, x0+L)
        ax.set_ylim(y0, y0+L)

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            rf'Box counting: $\epsilon = 1/2^{{{k}}} = 1/{n}$'
        )

        plt.tight_layout()

        plt.savefig(
            f"box_counting_k{k}.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()


# =================================================
# 7. REGRESION LINEAL
# =================================================

n_boxes = np.array(n_boxes)
N_eps = np.array(N_eps)

# Variables logaritmicas
ln_inv_eps = np.log(n_boxes)
ln_N_eps = np.log(N_eps)

# Seleccion de escalas para la regresion
mask = N_eps <= 0.05*N

if np.count_nonzero(mask) < 2:
    raise ValueError(
        "No hay suficientes escalas para realizar la regresion."
    )

# Ajuste lineal
slope, intercept = np.polyfit(
    ln_inv_eps[mask],
    ln_N_eps[mask],
    1
)

# =================================================
# 8. GRAFICA DE LA DIMENSION FRACTAL
# =================================================

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(
    ln_inv_eps,
    ln_N_eps,
    'o',
    label='Datos'
)

ax.plot(
    ln_inv_eps[mask],
    slope*ln_inv_eps[mask] + intercept,
    '-',
    label=rf'$D_f = {slope:.4f}$'
)

ax.set_xlabel(r'$\ln(1/\epsilon)$')
ax.set_ylabel(r'$\ln N(\epsilon)$')

ax.legend()

plt.tight_layout()

plt.savefig(
    "box_counting_regresion.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# =================================================
# 9. RESULTADOS
# =================================================

print("--------------------------------")
print("RESULTADOS DEL BOX COUNTING")
print("--------------------------------")

print(f"Numero de puntos: {N}")
print(f"Dimension fractal estimada: {slope:.6f}")
print(f"Intercepto: {intercept:.6f}")
print(f"Escalas utilizadas: {np.count_nonzero(mask)}")
