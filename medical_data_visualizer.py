import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importar datos
df = pd.read_csv('medical_examination.csv')

# Añadir la columna 'overweight'
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Normalizar las columnas 'cholesterol' y 'gluc'
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # Crear DataFrame para el gráfico categórico
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Reordenar las etiquetas para que coincidan con el test esperado
    variable_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    
    # Dibujar el gráfico categórico
    fig = sns.catplot(x="variable", hue="value", col="cardio",
                      data=df_cat, kind="count", height=5, aspect=1, order=variable_order)
    
    fig.set_axis_labels("variable", "total")
    fig = fig.fig

    # Guardar y devolver
    fig.savefig('catplot.png')
    return fig

# Función para dibujar el heatmap
def draw_heat_map():
    # Filtrar los datos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calcular la matriz de correlación
    corr = df_heat.corr()

    # Generar la máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Dibujar el heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', 
                square=True, center=0, vmax=0.3, vmin=-0.3,
                cbar_kws={"shrink": 0.5}, cmap="RdBu")
    
    # Guardar y devolver
    fig.savefig('heatmap.png')
    return fig
