import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_csv('../data/Entrenamieto_ECI_2020.csv')
df.sample(10).T


del df['Prod_Category_A']
del df['Actual_Delivery_Date']
del df['ASP_Currency'] # Está en moneda local
del df['ASP'] # Está en moneda local
del df['Last_Activity']
del df['Submitted_for_Approval'] # Llena de 0s


df['Planned_Delivery_Start_Date'] = pd.to_datetime(df['Planned_Delivery_Start_Date'])
df['Planned_Delivery_End_Date'] = pd.to_datetime(df['Planned_Delivery_End_Date'])
df['Account_Created_Date'] = pd.to_datetime(df['Account_Created_Date'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')


df.info()


df.groupby('Stage').size()


df.groupby('Opportunity_Name').size()


df.groupby('Region').size()


df.groupby('Territory').size()


df.groupby('Opportunity_Type').size().sort_values()


df.groupby('Delivery_Terms').size()


df.groupby('Product_Family').size().sort_values()


matriz_correlacion = df.corr()
_ = sns.heatmap(matriz_correlacion, cmap='inferno_r')


stage_ordenado = ['Closed Won', 'Closed Lost', 'Negotiation', 'Proposal', 'Qualification']


df['Region'].value_counts()


region_registros = df.groupby('Region')['ID'].transform(lambda x: x.count())
eci_region = df[region_registros > 1000] # Filtramos regiones con mas de 1000 registros

region_stage = eci_region.groupby(['Region'])['Stage'].value_counts(normalize=True)*100
region_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df['Account_Type'].value_counts()


tipo_cuenta_registros = df.groupby('Account_Type')['ID'].transform(lambda x: x.count())
eci_tipo_cuenta = df[tipo_cuenta_registros > 1000] # Filtramos tipos de cuenta con mas de 1000 registros

tipo_cuenta_stage = eci_tipo_cuenta.groupby(['Account_Type'])['Stage'].value_counts(normalize=True)*100
tipo_cuenta_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df['Opportunity_Type'].value_counts()


tipo_oportunidad_registros = df.groupby('Opportunity_Type')['ID'].transform(lambda x: x.count())
eci_tipo_oportunidad = df[tipo_oportunidad_registros > 1000] # Filtramos tipo de oportunidad con mas de 1000 registros

tipo_oportunidad_stage = eci_tipo_oportunidad.groupby(['Opportunity_Type'])['Stage'].value_counts(normalize=True)*100
tipo_oportunidad_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df.Account_Owner.value_counts().head(10)


vendedor_cliente_registros = df.groupby('Account_Owner')['ID'].transform(lambda x: x.count())
eci_vendedor_cliente = df[vendedor_cliente_registros > 1000] # Filtramos vendedor con mas de 1000 registros

vendedor_cliente_stage = eci_vendedor_cliente.groupby(['Account_Owner'])['Stage'].value_counts(normalize=True)*100
vendedor_cliente_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df.Opportunity_Owner.value_counts().head(10)


vendedor_oportunidad_registros = df.groupby('Opportunity_Owner')['ID'].transform(lambda x: x.count())
eci_vendedor_oportunidad = df[vendedor_oportunidad_registros > 1000] # Filtramos vendedor con mas de 1000 registros

vendedor_oportunidad_stage = eci_vendedor_oportunidad.groupby(['Opportunity_Owner'])['Stage'].value_counts(normalize=True)*100
vendedor_oportunidad_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df['Product_Family'].value_counts()


familia_producto_registros = df.groupby('Product_Family')['ID'].transform(lambda x: x.count())
eci_familia_producto = df[familia_producto_registros > 500] # Filtramos vendedor con mas de 1000 registros

familia_producto_stage = eci_familia_producto.groupby(['Product_Family'])['Stage'].value_counts(normalize=True)*100
familia_producto_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df['Delivery_Terms'].value_counts()


Delivery_Terms_registros = df.groupby('Delivery_Terms')['ID'].transform(lambda x: x.count())
eci_Delivery_Terms = df[Delivery_Terms_registros > 1000] # Filtramos Delivery_Termses con mas de 1000 registros

Delivery_Terms_stage = eci_Delivery_Terms.groupby(['Delivery_Terms'])['Stage'].value_counts(normalize=True)*100
Delivery_Terms_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


df['Billing_Country'].value_counts()


Billing_Country_registros = df.groupby('Billing_Country')['ID'].transform(lambda x: x.count())
eci_Billing_Country = df[Billing_Country_registros > 1000] # Filtramos Billing_Country con mas de 1000 registros

Billing_Country_stage = eci_Billing_Country.groupby(['Billing_Country'])['Stage'].value_counts(normalize=True)*100
Billing_Country_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]





# Porcentaje de Stage para Region y Delivery_Terms

region_delivery_registros = df.groupby(['Region','Delivery_Terms'])['ID'].transform(lambda x: x.count())
eci_region_delivery = df[region_delivery_registros > 500] # Filtramos regiones con mas de 500 registros

region_delivery_stage = eci_region_delivery.groupby(['Region','Delivery_Terms'])['Stage'].value_counts(normalize=True)*100
region_delivery_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


# Porcentaje de Stage para Region y Tipo de Cuenta

region_cuenta_registros = df.groupby(['Region','Account_Type'])['ID'].transform(lambda x: x.count())
eci_region_cuenta = df[region_cuenta_registros > 500] # Filtramos regiones con mas de 500 registros

region_cuenta_stage = eci_region_cuenta.groupby(['Region','Account_Type'])['Stage'].value_counts(normalize=True)*100
region_cuenta_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


# Porcentaje de Stage para Pais de Facturacion y Delivery_Terms

billing_delivery_registros = df.groupby(['Billing_Country','Delivery_Terms'])['ID'].transform(lambda x: x.count())
eci_billing_delivery = df[billing_delivery_registros > 500] # Filtramos Billing_Countryes con mas de 500 registros

billing_delivery_stage = eci_billing_delivery.groupby(['Billing_Country','Delivery_Terms'])['Stage'].value_counts(normalize=True)*100
billing_delivery_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


# Porcentaje de Stage para Region, Tipo de Cuenta y Delivery_Terms

region_cuenta_delivery_registros = df.groupby(['Region','Account_Type','Delivery_Terms'])['ID'].transform(lambda x: x.count())
eci_region_cuenta_delivery = df[region_cuenta_delivery_registros > 500] # Filtramos regiones con mas de 500 registros

region3_stage = eci_region_cuenta_delivery.groupby(['Region','Account_Type','Delivery_Terms'])['Stage'].value_counts(normalize=True)*100
region3_stage.unstack(fill_value=0).sort_values(by='Closed Won', ascending=False)[stage_ordenado]


import math
data_stage_count = df['Stage'].value_counts().sort_values(ascending=False) \
  .to_frame().reset_index().rename(columns={'index': 'Stage', 'Stage': 'Cantidad'})
  
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Cantidad', y='Stage', data=data_stage_count, orient='h', 
                 palette=[(0.118, 0.533, 0.898)])
ax.set_title('Cantidad de oportunidades por Stage')
ax.set_xscale('log')
# plt.xticks([])
for index, row in data_stage_count.iterrows():
    ax.text(row['Cantidad'] * 1.2 , row.name, round(row['Cantidad'], 2), 
            color='black', ha='left', va='center')





oportunidades_por_trimestre = df.groupby(['Delivery_Year', 'Delivery_Quarter'])['ID'].count().reset_index()
oportunidades_por_trimestre['Trimestre'] = oportunidades_por_trimestre['Delivery_Year'].astype('string') + ' ' + oportunidades_por_trimestre['Delivery_Quarter']
del oportunidades_por_trimestre['Delivery_Year']
del oportunidades_por_trimestre['Delivery_Quarter']
oportunidades_por_trimestre.columns = ['Cant. Oportunidades', 'Trimestre']
ax = oportunidades_por_trimestre.plot(x='Trimestre', y='Cant. Oportunidades',
                                 kind='bar', figsize=(18, 5), rot=0,
                                 color=[(0.118, 0.533, 0.898)])


ax.set(xlabel='Año Trimestre', ylabel='Cantidad de oportunidades')
      
ax.set_title('Cantidad de oportunidades ganadas y perdidas por trimestre', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
ax.legend([])
print()


stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_trimestre = stage_reducido.groupby(['Delivery_Year', 'Delivery_Quarter', 'Stage'])['ID'].count().unstack().reset_index()
stage_por_trimestre['Trimestre'] = stage_por_trimestre['Delivery_Year'].astype('string') + ' ' + stage_por_trimestre['Delivery_Quarter']
stage_por_trimestre = stage_por_trimestre[['Trimestre', 'Closed Won', 'Closed Lost']]
stage_por_trimestre.columns = ['Trimestre', 'Cant. Ganados', 'Cant. Perdidos']
ax = stage_por_trimestre.plot(x='Trimestre', y=['Cant. Ganados', 'Cant. Perdidos'],
                              kind='bar', figsize=(18, 5), rot=0, 
                              color=[(0.118, 0.533, 0.898),
                                     (0.847, 0.106, 0.376)]) # o ['darkblue', 'crimson']

ax.set(xlabel='Año Trimestre', ylabel='Cantidad de oportunidades')
      
ax.set_title('Cantidad de oportunidades ganadas y perdidas por trimestre', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print()


# df.plot.plot(x='Month',y='ASP_(converted)', figsize=(60, 5))


# df['Precio']df['TRF']


df['TRF'].plot(kind='kde', figsize=(10, 5))


# Oportunidades ganadas y perdidas por Region

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_region = stage_reducido.groupby(['Region', 'Stage'])['ID'].count().unstack().reset_index()

stage_por_region = stage_por_region[['Region', 'Closed Won', 'Closed Lost']]
stage_por_region.columns = ['Region', 'Cant. Ganados', 'Cant. Perdidos']
ax = stage_por_region.plot(x='Region', y=['Cant. Ganados', 'Cant. Perdidos'], 
                      kind='bar', figsize=(18, 5), rot=0, 
                      color=[(0.118, 0.533, 0.898),
                              (0.847, 0.106, 0.376)]) # o ['darkblue', 'crimson']

ax.set(xlabel='Región', ylabel='Cantidad de oportunidades')
      
ax.set_title('Cantidad de oportunidades ganadas y perdidas por región', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print()


# Oportunidades ganadas y perdidas por Account_Type (con mas de 1000 registros)

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_reducido.loc[:,['has_contract_no']] = (stage_reducido['Sales_Contract_No'] get_ipython().getoutput("= 'None')")
stage_por_Account_Type = stage_reducido.groupby(['has_contract_no', 'Stage'])['ID'].count().unstack().reset_index()


stage_por_Account_Type = stage_por_Account_Type[['has_contract_no', 'Closed Won', 'Closed Lost']]
stage_por_Account_Type.columns = ['has_contract_no', 'Cant. Ganados', 'Cant. Perdidos']

ax = stage_por_Account_Type.plot(x='has_contract_no', y=['Cant. Ganados', 'Cant. Perdidos'], 
                            kind='bar', figsize=(18, 5), rot=0, 
                            color=[(0.118, 0.533, 0.898),
                              (0.847, 0.106, 0.376)])

ax.set(xlabel='Tipo de cuenta', ylabel='Cantidad de oportunidades')
      
ax.set_title('Cantidad de oportunidades ganadas y perdidas por tipo de cuenta', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print(stage_por_Account_Type)


# Oportunidades ganadas y perdidas por Account_Type (con mas de 1000 registros)

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_Account_Type = stage_reducido.groupby(['Account_Type', 'Stage'])['ID'].count().unstack().reset_index()
stage_por_Account_Type = stage_por_Account_Type[(stage_por_Account_Type['Closed Lost'] + stage_por_Account_Type['Closed Won']) > 1000] #filtramos ec de moivre

stage_por_Account_Type = stage_por_Account_Type[['Account_Type', 'Closed Won', 'Closed Lost']]
stage_por_Account_Type.columns = ['Account_Type', 'Cant. Ganados', 'Cant. Perdidos']
ax = stage_por_Account_Type.plot(x='Account_Type', y=['Cant. Ganados', 'Cant. Perdidos'], 
                            kind='bar', figsize=(18, 5), rot=0, 
                            color=[(0.118, 0.533, 0.898),
                              (0.847, 0.106, 0.376)]) # o ['darkblue', 'crimson']

ax.set(xlabel='Tipo de cuenta', ylabel='Cantidad de oportunidades')
      
ax.set_title('Cantidad de oportunidades ganadas y perdidas por tipo de cuenta', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print()


# Oportunidades ganadas y perdidas por Opportunity_Type (con mas de 1000 registros)

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_Opportunity_Type = stage_reducido.groupby(['Opportunity_Type', 'Stage'])['ID'].count().unstack().reset_index()
stage_por_Opportunity_Type = stage_por_Opportunity_Type[(stage_por_Opportunity_Type['Closed Lost'] + stage_por_Opportunity_Type['Closed Won']) > 1000] #filtramos ec de moivre

stage_por_Opportunity_Type = stage_por_Opportunity_Type[['Opportunity_Type', 'Closed Won', 'Closed Lost']]
stage_por_Opportunity_Type.columns = ['Opportunity_Type', 'Cant. Ganados', 'Cant. Perdidos']
ax = stage_por_Opportunity_Type.plot(x='Opportunity_Type', y=['Cant. Ganados', 'Cant. Perdidos'], kind='bar', figsize=(18, 5), rot=0, color=['blue', 'tomato']) # o ['darkblue', 'crimson']


# Oportunidades ganadas y perdidas por Account_Owner (con mas de 1000 registros)

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_Account_Owner = stage_reducido.groupby(['Account_Owner', 'Stage'])['ID'].count().unstack().reset_index()
stage_por_Account_Owner = stage_por_Account_Owner[(stage_por_Account_Owner['Closed Lost'] + stage_por_Account_Owner['Closed Won']) > 1000] #filtramos ec de moivre

stage_por_Account_Owner = stage_por_Account_Owner[['Account_Owner', 'Closed Won', 'Closed Lost']]
stage_por_Account_Owner.columns = ['Account_Owner', 'Cant. Ganados', 'Cant. Perdidos']
stage_por_Account_Owner.plot(x='Account_Owner', y=['Cant. Ganados', 'Cant. Perdidos'], kind='bar', figsize=(18, 5), rot=0, color=['blue', 'tomato']) # o ['darkblue', 'crimson']


# Oportunidades ganadas y perdidas por Bureaucratic_Code (con mas de 1000 registros)

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_Bureaucratic_Code = stage_reducido.groupby(['Bureaucratic_Code', 'Stage'])['ID'].count().unstack().reset_index()
stage_por_Bureaucratic_Code = stage_por_Bureaucratic_Code[(stage_por_Bureaucratic_Code['Closed Lost'] + stage_por_Bureaucratic_Code['Closed Won']) > 1000] #filtramos ec de moivre

stage_por_Bureaucratic_Code = stage_por_Bureaucratic_Code[['Bureaucratic_Code', 'Closed Won', 'Closed Lost']]
stage_por_Bureaucratic_Code.columns = ['Bureaucratic_Code', 'Cant. Ganados', 'Cant. Perdidos']
stage_por_Bureaucratic_Code.plot(x='Bureaucratic_Code', y=['Cant. Ganados', 'Cant. Perdidos'], kind='bar', figsize=(18, 5), rot=0, color=['blue', 'tomato']) # o ['darkblue', 'crimson']


df[['Month', 'Delivery_Year', 'Delivery_Quarter']].info()


# Oportunidades ganadas y perdidas por Delivery_Terms (con mas de 1000 registros)

stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_Delivery_Terms = (stage_reducido.groupby(['Delivery_Terms', 
                                                    'Stage'])['ID']
                            .count().unstack().reset_index())
stage_por_Delivery_Terms = \
  stage_por_Delivery_Terms[(stage_por_Delivery_Terms['Closed Lost'] 
                            + stage_por_Delivery_Terms['Closed Won']) > 500] #filtramos ec de moivre

stage_por_Delivery_Terms = stage_por_Delivery_Terms[['Delivery_Terms', 'Closed Won', 'Closed Lost']]
stage_por_Delivery_Terms.columns = ['Delivery_Terms', 'Cant. Ganados', 'Cant. Perdidos']
ax = stage_por_Delivery_Terms.plot(x='Delivery_Terms', 
                                   y=['Cant. Ganados', 'Cant. Perdidos'], 
                                   kind='bar', figsize=(18, 5), rot=0, 
                                   color=[(0.118, 0.533, 0.898),
                                          (0.847, 0.106, 0.376)]) # o ['darkblue', 'crimson']

ax.set(xlabel='Término de entrega', ylabel='Cantidad de oportunidades')
      
ax.set_title('Cantidad de oportunidades ganadas y perdidas por término de entrega', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print()


df['Delivery_Interval'] = (df['Planned_Delivery_End_Date'] - 
                           df['Planned_Delivery_Start_Date']).dt.days
casos_cerrados_filtrados_por_intervalo = df[(df['Delivery_Interval'] < 2000) & 
                                    (df['Stage'] == 'Closed Won')]

df[(df['Delivery_Interval'] >= 2000)]


promedio_intervalos_por_region = casos_cerrados_filtrados_por_intervalo \
  .groupby('Region').agg({
      'Delivery_Interval': ['count', 'mean']
  })[['Delivery_Interval']]

promedio_intervalos_por_region = promedio_intervalos_por_region[
  promedio_intervalos_por_region[('Delivery_Interval', 'count')] > 200
][[('Delivery_Interval', 'mean')]]

promedio_intervalos_por_region.columns = ['Delivery_Interval']
promedio_intervalos_por_region = promedio_intervalos_por_region \
  .sort_values(by='Delivery_Interval').reset_index()

axes = sns.barplot(data=promedio_intervalos_por_region,
            x='Delivery_Interval', y='Region',
            color='#1E88E5')

axes.set(xlabel='Días que duró la entrega', ylabel='Región')
axes.set_title('Días que duró la entrega por región', fontdict={
                    'fontsize': 15,
                    'fontweight': 2,
                  }, pad=10)

regiones_espanol = {
    'Japan': 'Japón',
    'Americas': 'América',
    'APAC': 'Asia-Pacífico',
    'EMEA': 'Europa, \nOriente Medio y \nÁfrica',
    'Middle East': 'Medio Oriente',
}
_ = axes.set_yticklabels([regiones_espanol[label.get_text()] 
                      for label in axes.get_yticklabels()])


import matplotlib.ticker as mticker


promedio_intervalos_por_region = df.groupby(['Total_Amount_Currency', 'Delivery_Year'])\
  .agg({
      'Total_Amount': 'mean'
  })[['Total_Amount']].unstack()

promedio_intervalos_por_region.columns = [anio for _, anio in 
                                          promedio_intervalos_por_region.columns]

axes = promedio_intervalos_por_region.T \
  .plot.line(figsize=(15,8),
  color=[
      '#5EA9DB',
      '#9469DB',
      '#54DB76',
      '#DB9B3D',
      '#DBD665',
  ])
axes.xaxis.set_major_locator(mticker.MultipleLocator(1))

axes.set(xlabel='Año', ylabel='Monto Total')
axes.set_title('Evolución de la monto total por moneda', fontdict={
                    'fontsize': 15,
                    'fontweight': 2, }, pad=10)

#_ = axes.legend([regiones_espanol.get(label, label) 
             # for label in axes.get_legend_handles_labels()[1]])


# TODO: Sacar middle east, no sé si está incluido en EMEA
# TODO: Filtrar los que tienen pocos registros
promedio_intervalos_por_region = casos_cerrados_filtrados_por_intervalo \
  .groupby(['Region', 'Delivery_Year'])\
  .agg({
      'Delivery_Interval': 'mean'
  })[['Delivery_Interval']].unstack()

promedio_intervalos_por_region.columns = [anio for _, anio in 
                                          promedio_intervalos_por_region.columns]

axes = promedio_intervalos_por_region.T \
  .plot.line(figsize=(15,8),
  color=[
      '#5EA9DB',
      '#9469DB',
      '#54DB76',
      '#DB9B3D',
      '#DBD665',
  ])
axes.xaxis.set_major_locator(mticker.MultipleLocator(1))

axes.set(xlabel='Año', ylabel='Días que duró la entrega')
axes.set_title('Evolución de la duración (en días) de entregas por región', fontdict={
                    'fontsize': 15,
                    'fontweight': 2,
                  }, pad=10)

_ = axes.legend([regiones_espanol.get(label, label) 
             for label in axes.get_legend_handles_labels()[1]])


import random
regions = ['EMEA', 'Japan']
promedio_intervalos_por_region = casos_cerrados_filtrados_por_intervalo[
  (casos_cerrados_filtrados_por_intervalo['Region'] == 'Japan') | \
  (casos_cerrados_filtrados_por_intervalo['Region'] == 'EMEA')
] \
  .groupby(['Region', 'Delivery_Year'])\
  .agg({
      'Delivery_Interval': 'mean'
  })[['Delivery_Interval']].unstack()

promedio_intervalos_por_region.columns = [anio for _, anio in 
                                          promedio_intervalos_por_region.columns]
promedio_intervalos_por_region.T.head()

fig, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
for i, region in enumerate(regions):
  promedio_intervalos_por_region.iloc[i, :].T \
    .plot.line(figsize=(15,8),
    color=([
        '#9469DB',
        '#DB9B3D',
    ][i]), ax=axes[i])
  axes[i].xaxis.set_major_locator(mticker.MultipleLocator(1))

  axes[i].set(xlabel='Año', ylabel='Días que duró la entrega')

  axes[i].legend([regiones_espanol.get(label, label) 
              for label in axes[i].get_legend_handles_labels()[1]])
_ = fig.suptitle('Evolución de la duración (en días) de entregas por región', fontsize=15)


df_filtrados_por_intervalo = df[(df['Delivery_Interval'] < 500) & 
                                df['Stage'].str.startswith('Closed')]
df_filtrados_por_intervalo[['TRF', 'Delivery_Interval']] \
  .describe().T


import matplotlib.ticker as ticker






# plt.figure(figsize=(30, 30))
# axes = 
def plot_scatter_trf_vs_delivery(data, name, ax):
  sns.scatterplot(x='Delivery_Interval', y='TRF', 
                  data=data \
                  [data['TRF'] > 0],
                  hue='Stage', linewidth=0, style='Stage', markers={
                      'Closed Won': 'P',
                      'Closed Lost': 'o',
                  }, palette={
                      'Closed Won': (0.118, 0.533, 0.898, 0.8),
                      'Closed Lost': (0.847, 0.106, 0.376, 0.2),
                  }, ax=ax)
  ax.set(xlabel='Duración del envío', 
          ylabel='TRF')
  ax.set_title(name, 
              fontdict={
                  'fontsize': 10,
                  'fontweight': 2,
                })
  ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

fig, ax = plt.subplots(1, 1, figsize=(15, 7))
plot_scatter_trf_vs_delivery(df_filtrados_por_intervalo, 
                             'Op. ganadas y perdidas según TRF vs Duración (días) del envío', ax)
ax.set_title('Op. ganadas y perdidas según TRF vs Duración (días) del envío', 
              fontdict={
                  'fontsize': 14,
                  'fontweight': 2,
                })
print()



n_cols = 2
fig, axes = plt.subplots(5, n_cols, figsize=(15, 15))
plt.subplots_adjust(hspace=0.6, wspace=0.4)
zoom = {
    'APAC': (80, 50),
    'Americas': (100, 70),
    'EMEA': (60, 60),
    'Japan': (100, 30),
    'Middle East': (70, 40),
}
for i, (name, group) in enumerate(df_filtrados_por_intervalo.groupby('Region')):
  plot_scatter_trf_vs_delivery(group, name, axes[i][0])
  plot_scatter_trf_vs_delivery(group[(group['Delivery_Interval'] < zoom[name][0]) & \
                                     (group['TRF'] < zoom[name][1])], name + " Zoomed", axes[i][1])

fig.suptitle('Op. ganadas y perdidas según TRF vs Duración (días) del envío por región', 
  fontsize=15)
print()


fig, ax = plt.subplots(figsize=(15, 7))

sns.violinplot(y="Region", x="Delivery_Interval", hue="Stage",
               data=df_filtrados_por_intervalo \
               [df_filtrados_por_intervalo['Delivery_Interval'] < 200], 
               split=True, ax=ax, scale="width", cut=0,
               hue_order=['Closed Won', 'Closed Lost'], palette={
                  'Closed Won': (0.118, 0.533, 0.898, 0.8),
                  'Closed Lost': (0.847, 0.106, 0.376, 0.2),
               })
# legend = ax._legend

ax.set(xlabel='Duración de la entrega (días)', ylabel='Región')
ax.set_title('Distribución de la duración de la entrega \n'
             'por oportunidades ganadas y perdidas', fontdict={
                  'fontsize': 15,
                  'fontweight': 2,
                })
tipo_oportunidad_espanol = {
    'Closed Won': 'Op. Ganada',
    'Closed Lost': 'Op. Perdida'
}
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, list(tipo_oportunidad_espanol.values()),
          loc='lower right')
print()



ax = sns.barplot(data=df['Delivery_Terms'].value_counts().to_frame().reset_index(), 
                 y='index', x='Delivery_Terms', palette=[(0.118, 0.533, 0.898)])
ax.set_xscale('log')

ax.set(xlabel='Cantidad de oportunidades', ylabel='Término de entrega')
ax.set_title('Cantidad de oportunidades ganadas y \nperdidas por término de entrega', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print()


productos_won_lost_por_product_family = df.groupby('Product_Family').agg({
    'Stage': [('Won', lambda x: x[x == 'Closed Won'].count()),
              ('Lost', lambda x: x[x == 'Closed Lost'].count()),
              ('Count', 'count')],
    'Delivery_Interval': [('Delivery_Interval_Mean', 'mean')]
})
productos_won_lost_por_product_family.columns = ['Won', 'Lost', 'Count', 'Delivery_Interval_Mean']

productos_won_lost_por_product_family['Won_Ratio'] = \
  productos_won_lost_por_product_family['Won'] / \
  productos_won_lost_por_product_family['Count'] # Desprecio los otros posibles
                                                # Stages porque son poquitos
display(productos_won_lost_por_product_family.describe().T)
productos_won_lost_por_product_family.head(5)


productos_won_lost_por_product_family = productos_won_lost_por_product_family \
  [productos_won_lost_por_product_family['Count'] > 200] # Ecuación de de Moivre


top_7_productos_won_lost_por_product_family = \
  productos_won_lost_por_product_family.nlargest(7, columns=['Won_Ratio']) \
  .iloc[::-1]


axes = top_7_productos_won_lost_por_product_family[['Won', 'Lost']] \
  .plot.barh(stacked=True, 
             color={
      'Won': '#1E88E5',
      'Lost': '#D81B60',
  }, linewidth=0, width=0.7, figsize=(15, 7))

axes.set_title('Top 7 familias de productos con más porcentaje de op. de ganadas', 
               fontdict={
                  'fontsize': 15,
                  'fontweight': 2,
                })

tipo_oportunidad_espanol = {
    'Won': 'Op. Ganada',
    'Lost': 'Op. Perdida'
}
axes.legend([tipo_oportunidad_espanol.get(label, label) 
             for label in axes.get_legend_handles_labels()[1]])
axes.set(xlabel='Cantidad de oportunidades', ylabel='Familia de producto')
print()



top_7_worst_productos_won_lost_por_product_family = \
  productos_won_lost_por_product_family.nsmallest(7, columns=['Won_Ratio']) \
  .iloc[::-1]


axes = top_7_worst_productos_won_lost_por_product_family[['Won', 'Lost']] \
  .plot.barh(stacked=True, 
             color={
      'Won': '#1E88E5',
      'Lost': '#D81B60',
  }, linewidth=0, width=0.7, figsize=(15, 7))

axes.set_title('Top 7 familias de productos con menos porcentaje de op. de ganadas', 
               fontdict={
                  'fontsize': 15,
                  'fontweight': 2,
                })

tipo_oportunidad_espanol = {
    'Won': 'Op. Ganada',
    'Lost': 'Op. Perdida'
}
axes.legend([tipo_oportunidad_espanol.get(label, label) 
             for label in axes.get_legend_handles_labels()[1]])
axes.set(xlabel='Cantidad de oportunidades', ylabel='Familia de producto')
print()


df_ordenados_por_won_ratio = productos_won_lost_por_product_family \
  [(productos_won_lost_por_product_family['Count'] > 100)] \
  .sort_values(by='Won_Ratio')
plt.figure(figsize=(10, 12))
plt.plot(df_ordenados_por_won_ratio['Won_Ratio'] * 100,
         df_ordenados_por_won_ratio.index, marker='o', color='black', 
         linestyle='')
plt.hlines(df_ordenados_por_won_ratio.index, xmin=0, 
            xmax=df_ordenados_por_won_ratio['Won_Ratio'] * 100)
axes = plt.gca()
axes.set_title('Porcentaje de op. de ganadas por familia de producto', 
               fontdict={
                  'fontsize': 15,
                  'fontweight': 2,
                })
axes.set(xlabel='Porcentaje de op. ganadas', ylabel='Familia de producto')
print()


plt.style.use('fivethirtyeight')
regiones = df['Region'].unique()
fig, ax = plt.subplots()
colores = ['red', 'blue', 'green', 'purple', 'yellow']
index_color = 0
for region in regiones:
  df[df['Region'] == region].plot.scatter(x='TRF', y='Total_Amount', label=region, ax=ax, c=colores[index_color], figsize=(12,4))
  index_color += 1


regiones = df['Region'].unique()
fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
colores = ['red', 'blue', 'green', 'purple', 'yellow']
index_color = 0
for region in regiones:
  df[df['Region'] == region].plot.scatter(x='TRF', y='Total_Amount', label=region, ax=ax[index_color], c=colores[index_color], figsize=(15,3))
  index_color += 1


regiones = df['Total_Amount_Currency'].unique()
fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
colores = ['red', 'blue', 'green', 'purple', 'yellow']
index_color = 0
for region in regiones:
  df[df['Total_Amount_Currency'] == region].plot.scatter(x='TRF', y='Total_Amount', label=region, ax=ax[index_color], c=colores[index_color], figsize=(15,3))
  index_color += 1


import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.io import show 
from bokeh.sampledata.les_mis import data
import numpy as np

def generate_chord_for_owners(data):
  nodes = hv.Dataset(pd.DataFrame(pd.concat([data['Account_Owner'], 
                                  data['Opportunity_Owner']]).unique()).reset_index().rename(columns={
                                      0: 'Name'
                                  }), 'index')
  links = data[['Account_Owner',	'Opportunity_Owner', 'Stage']]
  links = links[links['Account_Owner'] get_ipython().getoutput("= links['Opportunity_Owner']]")
  # display(links)
  # display(links.count())

  # display(links.groupby(['Account_Owner',	'Opportunity_Owner']).size().reset_index())

  # display(links.count())
  links['value'] = links.groupby(['Account_Owner',	'Opportunity_Owner']).transform('count')\
    .reset_index().iloc[:,1]

  
  links = links[links['value'] > 5]
  links = links \
          .merge(nodes.data, right_on='Name',
          left_on='Account_Owner') \
          .merge(nodes.data, right_on='Name',
          left_on='Opportunity_Owner')[['index_x', 'index_y', 'value', 'Stage']]
  
  if links.empty:
    return
  hv.extension('bokeh')
  hv.output(size=300)
  chord = hv.Chord((links, nodes)) #.select(value=(5, None))
  return chord.opts(
      opts.Chord(
          cmap='Category10', 
          edge_cmap=['#d81b60', '#1e88e5'],
          # colors={'red', 'green'},
          edge_color=dim('Stage').str(),
          labels='Name',
          # node_color=dim('index').str()
      )
  )

(generate_chord_for_owners(df))


for name, group in df.groupby('Region'):
  print(name)
  # display(group.head(10))
  display(generate_chord_for_owners(group))


import plotly.express as px
import plotly.graph_objects as go

def create_parallel_owners(data):
  df_for_parallel = data.copy()
  df_for_parallel['count_1'] = df_for_parallel.groupby(['Account_Owner']) \
    ['Opportunity_Owner'].transform('size')
  df_for_parallel = df_for_parallel[df_for_parallel['count_1'] > 100]

  df_for_parallel['count_2'] = df_for_parallel.groupby(['Opportunity_Owner']) \
    ['Account_Owner'].transform('size')
  df_for_parallel = df_for_parallel[df_for_parallel['count_2'] > 100]

  df_for_parallel.loc[:,'Won'] = (df_for_parallel['Stage'] == 'Closed Won').astype(int)
  colorscale = [[0, 'rgb(0.847, 0.106, 0.376)'], [1, 'rgb(0.118, 0.533, 0.898)']]

  account_owner = go.parcats.Dimension(
      values=df_for_parallel['Account_Owner'],
      categoryorder='category ascending', label="Dueño de \nla cuenta"
  )
  opportunity_owner = go.parcats.Dimension(
      values=df_for_parallel['Opportunity_Owner'],
      categoryorder='category ascending', label="Dueño de \nla oportunidad"
  )
  fig = go.Figure(data = [go.Parcats(dimensions=[account_owner, opportunity_owner],
          line={'color': df_for_parallel['Won'], 'colorscale': colorscale},
          labelfont={'size': 18, 'family': 'Helvetica'},
          tickfont={'size': 12, 'family': 'Helvetica'},
          arrangement='freeform')])

  fig.update_layout(
          height=800, width=500, xaxis={'title': 'Horsepower'},
          yaxis={'title': 'MPG', 'domain': [0.6, 1]},
          dragmode='lasso', hovermode='closest')
  fig.show()


for name, group in df.groupby('Region'):
  print(name)
  create_parallel_owners(group)


create_parallel_owners(df)


df_product_owner = df.copy()

df_product_owner = (df_product_owner.groupby(['Region', 'Account_Owner']).agg({
    'Stage': [('Won', lambda x: x[x == 'Closed Won'].count()),
              ('Lost', lambda x: x[x == 'Closed Lost'].count()),
              ('Count', 'count')]
  }).reset_index()
)
df_product_owner.columns = ['Region', 'Account_Owner', 'Won', 'Lost', 'Count']
df_product_owner['Win_Ratio_Opp_Owner'] = df_product_owner['Won'] / df_product_owner['Count']
df_product_owner.head(80)



fig, ax = plt.subplots(figsize=(15, 7))

sns.violinplot(x="Region", y="Win_Ratio_Opp_Owner",
               data=(df_product_owner[(df_product_owner['Count'] > 10) & 
                                     (df_product_owner['Region'] get_ipython().getoutput("= 'Middle East')]),")
               split=True, ax=ax, scale="width", cut=0,
               palette=[(0.118, 0.533, 0.898),])

ax.set(xlabel='Región', ylabel='Win-ratio')
ax.set_title('Distribución del win-ratio \n'
             'de los dueños de oportunidad por región', fontdict={
                  'fontsize': 15,
                  'fontweight': 2,
                })
print()



fig, ax = plt.subplots(figsize=(15, 7))
sns.boxplot(y="Region", x="Count",
               data=(df_product_owner[(df_product_owner['Count'] > 10) & (df_product_owner['Count'] < 1500) &
                                     (df_product_owner['Region'] get_ipython().getoutput("= 'Middle East')]),")
               ax=ax, palette=[(0.118, 0.533, 0.898),])

ax.set(xlabel='Cantidad de ventas', ylabel='Región')
ax.set_title('Distribución de la cantidad de ventas \n'
             'de los dueños de oportunidad por región', fontdict={
                  'fontsize': 15,
                  'fontweight': 2,
                })
print()


df_product_owner_g = df.copy()

# df_product_owner_g = 
ax = (df_product_owner_g.groupby('Region').size().sort_values().plot.barh(color=[(0.118, 0.533, 0.898),]))
ax.set(xlabel='Cantidad de personas', ylabel='Región')
ax.set_title('Cantidad de dueños de oportunidad por región',
             fontdict={
              'fontsize': 15,
              'fontweight': 2,
            })
print()




stage_reducido = df[(df['Stage'] == 'Closed Won') | (df['Stage'] == 'Closed Lost')]
stage_por_region = stage_reducido.groupby(['Region', 'Stage'])['ID'].count().unstack().reset_index()

stage_por_region = stage_por_region[['Region', 'Closed Won', 'Closed Lost']]
stage_por_region.columns = ['Region', 'Cant. Ganados', 'Cant. Perdidos']

fig, ax = plt.subplots(1, 2)
stage_por_region.plot(x='Region', y=['Cant. Ganados', 'Cant. Perdidos'], 
                      kind='bar', figsize=(18, 5), rot=0, ax=ax[1],
                      color=[(0.118, 0.533, 0.898),
                              (0.847, 0.106, 0.376)]) # o ['darkblue', 'crimson']

ax[1].set(xlabel='Región', ylabel='Cantidad de oportunidades')
      
ax[1].set_title('Cantidad de oportunidades ganadas y perdidas por región\nAlternativo-Red-Green', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)

stage_por_region.plot(x='Region', y=['Cant. Ganados', 'Cant. Perdidos'], 
                      kind='bar', figsize=(18, 5), rot=0, ax=ax[0],
                      color=['red', 'green']) 
ax[0].set(xlabel='Región', ylabel='Cantidad de oportunidades')
      
ax[0].set_title('Cantidad de oportunidades ganadas y perdidas por región\nRed-Green', fontdict={
                    'fontsize': 18,
                    'fontweight': 2,
                  }, pad=10)
print()
