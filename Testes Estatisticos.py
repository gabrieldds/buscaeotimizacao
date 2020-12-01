from scipy import stats
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt


hc = np.loadtxt('result_ackley_hill_climbing.txt')
sa = np.loadtxt('result_ackley_simulated_annealing.txt')
ga = np.loadtxt('result_genetic_algorithm_ackley.txt')
de = np.loadtxt('resul_de_ackley.txt')

data = [hc, sa, ga, de]
# Friedman de grupo
print(stats.friedmanchisquare(*data))

# Kruskal-Wallis de grupo
print(stats.kruskal(*data))

#Teste de Conover baseado em Kruskal-Wallis
pc = sp.posthoc_conover(data)

#Caso precise mudar os indices e colunas do DataFrame
pc.columns = ['HC', 'SA', 'GA', 'DE']
pc.index = ['HC', 'SA', 'GA', 'DE']

print(pc)

plt.boxplot(data, labels=['Hill Climbing', 'Simulated Annealing', 'GA', 'DE'])
plt.show()

#Heatmap do Teste de Conover
cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(pc, **heatmap_args)
plt.show()

#Exportar Dataframe para Latex
print(pc.to_latex(decimal=",", float_format="%.2f"))