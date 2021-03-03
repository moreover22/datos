import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
import numpy as np

get_ipython().run_line_magic("matplotlib", " inline")
sns.set()


@lru_cache()
def get_data(path):
    df = pd.read_csv(path)
    return df


def clean_df_column_names(df):
    df.columns = [(col.lower()
                      .rstrip()
                      .lstrip()
                      .replace(' ', '_')
                      .replace(',', '')
                      .replace('(', '')
                      .replace(')', ''))
                    for col in df.columns]
    return df


def clean_unused_columns(dataset, cols=None):
    cols_to_remove = ['prod_category_a',
                          'actual_delivery_date',
                          'asp_currency',  # Está en moneda local
                          'asp', # Está en moneda local
                          'last_activity',
                          'submitted_for_approval', # Llena de 0s
                          'sales_contract_no', # Revela el target, se explica 
                                               # en su correspondiente sección
                          'asp_converted_currency']
    if cols is not None:
        cols_to_remove = cols
    return dataset.drop(cols_to_remove, axis=1, errors='ignore')


def convert_columns_to_correct_format(df):
    df['Planned_Delivery_Start_Date'] = pd.to_datetime(df['Planned_Delivery_Start_Date'])
    df['Planned_Delivery_End_Date'] = pd.to_datetime(df['Planned_Delivery_End_Date'])
    df['Account_Created_Date'] = pd.to_datetime(df['Account_Created_Date'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    return df


stages = {
  'Closed Lost': 0,
  'Closed Won': 1,
}
def get_palette(mode='rgb_p', category=None):
    palette = {
        'rgb_p': {
            0: (0.847, 0.106, 0.376),
            1: (0.118, 0.533, 0.898),
        },
        'hex': {
            0: '#D81B60',
            1: '#1E88E5',
        }
    }

    if category == None:
        return palette[mode]
    return palette[mode][category]

def get_palette_neutro(mode='rgb_p'):
    return get_palette(mode, 1)

def get_title_style(category=None):
    return {
        'fontdict': {
            'fontsize': 18,
            'fontweight': 2,
        },
        'pad': 10,
    }


df = get_data('../data/Train_TP2_Datos_2020-2C.csv')
df = clean_df_column_names(df)


df_submit = get_data('../data/Test_TP2_Datos_2020-2C.csv')
df_submit = clean_df_column_names(df_submit)


has_contract_no_vs_stage_won = df.groupby('opportunity_id').agg({
    'sales_contract_no': [('has_contract_no', lambda x: any(c get_ipython().getoutput("= 'None' for c in x))],")
    'stage': [('has_won', lambda x: any(c == 'Closed Won' for c in x))],
})
has_contract_no_vs_stage_won.columns = [col[1] for col in has_contract_no_vs_stage_won.columns]
has_contract_no_vs_stage_won = (has_contract_no_vs_stage_won
                                .reset_index()
                                .groupby(['has_contract_no', 'has_won'])['opportunity_id']
                                .count().unstack().reset_index())
has_contract_no_vs_stage_won.columns = ['Tiene número de contrato', 'Cant. perdidas', 'Cant. ganadas']
ax = has_contract_no_vs_stage_won.plot(x='Tiene número de contrato', y=['Cant. perdidas', 'Cant. ganadas'], 
                            kind='bar', figsize=(18, 5), rot=0, 
                            color=get_palette().values())
ax.set_ylabel('Cant. de oportunidades')
_ = ax.set_title('Tener número de contrato delata la probabilidad de éxito', **get_title_style())


df.head().T


df.info()


df.describe().T


df.shape


len(df.opportunity_id.unique())


df.prod_category_a.value_counts()


df.actual_delivery_date.value_counts()


df.last_activity.value_counts()


df.submitted_for_approval.value_counts()


df.asp_converted_currency.value_counts()


df.asp.value_counts()


df.asp_currency.value_counts()


df_filtered = df[['asp', 'asp_currency', 'asp_converted']].copy()
df_filtered = df_filtered[~df_filtered.asp.isna() & ~df_filtered.asp_converted.isna()]
df_filtered['asp_ratio'] = df_filtered.asp / df_filtered.asp_converted
df_filtered[['asp_currency', 'asp_ratio']].groupby('asp_currency').std()



df_filtered_japan = df_filtered[df_filtered.asp_currency.eq('JPY')][['asp', 'asp_converted']]

fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
_ = sns.scatterplot(data=df_filtered_japan, x='asp', y='asp_converted', ax=ax)


df = clean_unused_columns(df, ['asp_ratio']) # TODO: Se podría utilizar para 
                                             # convertir las otras columnas de precios


df = clean_unused_columns(df)


import math
data_stage_count = (
    df.stage.value_counts()
               .sort_values(ascending=False)
               .to_frame()
               .reset_index()
               .rename(columns={
                           'index': 'Stage', 
                           'stage': 'Cantidad'
                       }))
  
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Cantidad', y='Stage', 
                 data=data_stage_count, orient='h', 
                 palette=[get_palette_neutro()])
ax.set_title('Cantidad de oportunidades por Stage')
ax.set_xscale('log')
for index, row in data_stage_count.iterrows():
    ax.text(row['Cantidad'] * 1.2 , row.name, 
            round(row['Cantidad'], 2), 
            color='black', ha='left', 
            va='center')


df = df[df.stage.eq('Closed Won') | df.stage.eq('Closed Lost')]


df['stage'] = df.loc[:,'stage'].eq('Closed Won').astype(int)


df.head().T


brand_counts = df.brand.value_counts()
percentage_brand_none = brand_counts['None'] /  brand_counts.sum()
print(percentage_brand_none)
brand_counts


# TODO: Esto es una primera transformación que se deberá hacer sobre el set de train también
df['has_brand'] = df.brand.ne('None').astype(int)
df = clean_unused_columns(df, ['brand'])


df.head().T


# TODO: reducir la cardinalidad de territory
register_per_territory = df.territory.value_counts()
register_per_territory_lower = register_per_territory[register_per_territory < 200]
_ = register_per_territory[register_per_territory < 800].hist(bins=5)
print("Territorios distintos", len(register_per_territory_lower))
register_per_territory_lower


dataset['price'] = pd.to_numeric(dataset['price'], errors='coerce')









dataset['planned_delivery_start_date'] = \
  pd.to_datetime(dataset['planned_delivery_start_date'])
dataset['planned_delivery_end_date'] = \
  pd.to_datetime(dataset['planned_delivery_end_date'])
dataset['account_created_date'] = \
  pd.to_datetime(dataset['account_created_date'])
dataset['price'] = pd.to_numeric(dataset['price'], errors='coerce')


dataset = dataset[dataset['stage'].isin(stages.keys())]


import datetime as df
dataset['delivery_interval'] = (dataset['planned_delivery_end_date'] - 
                           dataset['planned_delivery_start_date']).dt.days


dataset['stage'] = dataset.stage.replace(stages) 


dataset.columns


dataset.shape


dataset['id'].unique().size


del dataset['id']


dataset['region'].head()


dataset['region'].value_counts().plot.bar(color=get_palette_neutro())
print()


(dataset.groupby(['region', 'stage'])
  .size().unstack()
  .plot.bar(color=get_palette('rgb_p')))

print()


get_ipython().getoutput("pip install squarify")


dataset.groupby(['region', 'territory']) \
  .size()


# import squarify
# squarify.plot(sizes=)
# _df = dataset.groupby(['region', 'territory']) \
#   .count()   #.unstack()
# ax = (_df[_df > 100]
#   .plot.barh(figsize=(15, 15)))

# ax.legend([])
# print()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset[['trf', 'delivery_interval']].replace({np.nan: 0}), dataset['stage'], 
    stratify=dataset['stage'], random_state=42
)


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier().fit(X_train, y_train)
(tree.score(X_train, y_train),
 tree.score(X_test, y_test))


from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
(gbrt.score(X_train, y_train),
 gbrt.score(X_test, y_test))


cat_op = ['pricing_delivery_terms_quote_appr',
'pricing_delivery_terms_approved',
'bureaucratic_code_0_approval',
'bureaucratic_code_0_approved',]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset[cat_op].replace({np.nan: 0}), dataset['stage'], 
    stratify=dataset['stage'], random_state=42
)


from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train, y_train)
(gbrt.score(X_train, y_train),
 gbrt.score(X_test, y_test))


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier().fit(X_train, y_train)
(tree.score(X_train, y_train),
 tree.score(X_test, y_test))


print(dataset.columns)
dataset_dummies = pd.get_dummies(dataset)
print(dataset_dummies.columns)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset_dummies.drop(columns=['planned_delivery_end_date', 
                                  'planned_delivery_start_date',
                                  'account_created_date', 'stage']).replace({np.nan: 0}), dataset_dummies['stage'], 
    stratify=dataset_dummies['stage'], random_state=42
)


from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)


X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
(X_train.shape,
X_train_selected.shape)


from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train_selected, y_train)
(gbrt.score(X_train_selected, y_train),
 gbrt.score(X_test_selected, y_test))
