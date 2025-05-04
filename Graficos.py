import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm

# Crear un gráfico con subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Media para todas las distribuciones
mean = 0
x = np.linspace(-10, 10, 1000)

# 1. Distribuciones normales con diferente desviación estándar
# Normal con desviación estándar baja
std_low = 1
y_low_std = norm.pdf(x, mean, std_low)
sns.lineplot(x=x, y=y_low_std, ax=axs[0, 0])
axs[0, 0].axvline(mean, color='red', linestyle='--', label='Media')  # Dibujar media
axs[0, 0].set_title('Normal - Desv. estándar baja')
axs[0, 0].legend()

# Normal con desviación estándar alta
std_high = 3
y_high_std = norm.pdf(x, mean, std_high)
sns.lineplot(x=x, y=y_high_std, ax=axs[0, 1])
axs[0, 1].axvline(mean, color='red', linestyle='--', label='Media')  # Dibujar media
axs[0, 1].set_title('Normal - Desv. estándar alta')
axs[0, 1].legend()

# 2. Distribución asimétrica (skewness)
skew_param = 5  # Parámetro de asimetría positivo
y_skew = skewnorm.pdf(x, skew_param)
sns.lineplot(x=x, y=y_skew, ax=axs[0, 2])
axs[0, 2].axvline(mean, color='red', linestyle='--', label='Media')  # Dibujar media
axs[0, 2].set_title('Distribución Asimétrica (skewed)')
axs[0, 2].legend()

# 3. Kurtosis: Leptokúrtica (más alta que normal)
leptokurtic_dist = np.random.normal(mean, std_low, 1000)
y_leptokurtic = norm.pdf(x, mean, 0.5)  # Kurtosis mayor con colas pesadas
sns.lineplot(x=x, y=y_leptokurtic, ax=axs[1, 0])
axs[1, 0].axvline(mean, color='red', linestyle='--', label='Media')  # Dibujar media
axs[1, 0].set_title('Leptokúrtica (kurt > 3)')
axs[1, 0].legend()

# 4. Kurtosis: Mesokúrtica (distribución normal estándar)
mesokurtic_dist = np.random.normal(mean, std_low, 1000)
y_mesokurtic = norm.pdf(x, mean, 1)  # Kurtosis normal
sns.lineplot(x=x, y=y_mesokurtic, ax=axs[1, 1])
axs[1, 1].axvline(mean, color='red', linestyle='--', label='Media')  # Dibujar media
axs[1, 1].set_title('Mesokúrtica (kurt = 3)')
axs[1, 1].legend()

# 5. Kurtosis: Platikúrtica (menos pesada que normal)
platykurtic_dist = np.random.uniform(-2, 2, 1000)
y_platykurtic = norm.pdf(x, mean, 2)  # Kurtosis menor con colas ligeras
sns.lineplot(x=x, y=y_platykurtic, ax=axs[1, 2])
axs[1, 2].axvline(mean, color='red', linestyle='--', label='Media')  # Dibujar media
axs[1, 2].set_title('Platikúrtica (kurt < 3)')
axs[1, 2].legend()

# Ajustar los gráficos
plt.tight_layout()
plt.show()

