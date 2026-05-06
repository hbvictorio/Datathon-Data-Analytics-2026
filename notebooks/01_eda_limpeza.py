#!/usr/bin/env python3
"""
=============================================================================
DATATHON PASSOS MÁGICOS - BLOCO 2: ANÁLISE EXPLORATÓRIA E LIMPEZA (2020-2024)
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 11, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 150, })

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PEDRA_COLORS = {'Quartzo': '#d62728', 'Ágata': '#ff7f0e', 'Ametista': '#9467bd', 'Topázio': '#2ca02c'}
PEDRA_ORDER = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
FIG_DIR = '/home/ubuntu/datathon_passos/figures/'

print("Carregando dados 2020-2024...")
df = pd.read_csv('/home/ubuntu/datathon_passos/data/dataset_completo_2020_2024.csv')
numeric_cols = ['INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['PEDRA'] = pd.Categorical(df['PEDRA'], categories=PEDRA_ORDER, ordered=True)
print(f"Dataset: {df.shape[0]} registros, Anos: {sorted(df['ANO'].unique())}")
print(df.groupby('ANO')['NOME'].count())

# P1: IAN
ian_by_year = df.groupby('ANO')['IAN'].agg(['mean','std']).round(2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(ian_by_year.index, ian_by_year['mean'], 'o-', color=COLORS[0], lw=2, ms=8)
axes[0].fill_between(ian_by_year.index, ian_by_year['mean']-ian_by_year['std'], ian_by_year['mean']+ian_by_year['std'], alpha=0.2)
axes[0].set_title('Evolução do IAN Médio (2020-2024)'); axes[0].set_xlabel('Ano'); axes[0].set_ylabel('IAN')
axes[0].set_xticks(sorted(df['ANO'].unique()))
for i, year in enumerate(sorted(df['ANO'].unique())):
    axes[1].hist(df[df['ANO']==year]['IAN'].dropna(), bins=20, alpha=0.5, label=str(int(year)), color=COLORS[i%len(COLORS)])
axes[1].set_title('Distribuição do IAN por Ano'); axes[1].legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p1_ian_defasagem.png'); plt.close()

# P2: IDA
ida_by_year = df.groupby('ANO')['IDA'].agg(['mean','std']).round(2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(ida_by_year.index, ida_by_year['mean'], color=COLORS[1], alpha=0.8, width=0.6)
for yr, row in ida_by_year.iterrows():
    axes[0].text(yr, row['mean']+0.1, f"{row['mean']:.2f}", ha='center', fontweight='bold')
axes[0].set_title('IDA Médio por Ano'); axes[0].set_xticks(sorted(df['ANO'].unique())); axes[0].set_ylim(0,10)
data_box = [df[df['ANO']==y]['IDA'].dropna().values for y in sorted(df['ANO'].unique())]
bp = axes[1].boxplot(data_box, labels=[str(int(y)) for y in sorted(df['ANO'].unique())], patch_artist=True)
for patch, c in zip(bp['boxes'], COLORS[:5]): patch.set_facecolor(c); patch.set_alpha(0.6)
axes[1].set_title('Distribuição do IDA por Ano')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p2_ida_desempenho.png'); plt.close()

# P3: IEG vs IDA e IPV
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, year in enumerate(sorted(df['ANO'].unique())):
    s = df[df['ANO']==year].dropna(subset=['IEG','IDA'])
    axes[0].scatter(s['IEG'], s['IDA'], alpha=0.3, s=15, color=COLORS[i%len(COLORS)], label=str(int(year)))
    s2 = df[df['ANO']==year].dropna(subset=['IEG','IPV'])
    axes[1].scatter(s2['IEG'], s2['IPV'], alpha=0.3, s=15, color=COLORS[i%len(COLORS)], label=str(int(year)))
axes[0].set_title('IEG vs IDA'); axes[0].legend(); axes[1].set_title('IEG vs IPV'); axes[1].legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p3_ieg_ida_ipv.png'); plt.close()

# P4: IAA coerência
fig, ax = plt.subplots(figsize=(12, 5))
vies_data = [(df[df['ANO']==y]['IAA']-df[df['ANO']==y]['IDA']).dropna().values for y in sorted(df['ANO'].unique())]
bp = ax.boxplot(vies_data, labels=[str(int(y)) for y in sorted(df['ANO'].unique())], patch_artist=True)
for patch, c in zip(bp['boxes'], COLORS[:5]): patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(y=0, color='red', ls='--', alpha=0.7); ax.set_title('Viés IAA-IDA por Ano')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p4_iaa_coerencia.png'); plt.close()

# P5: IPS antes de quedas
results = []
for y1, y2 in [(2020,2021),(2021,2022),(2022,2023),(2023,2024)]:
    d1 = df[df['ANO']==y1][['NOME','IPS','INDE']].rename(columns={'IPS':f'IPS','INDE':'INDE1'})
    d2 = df[df['ANO']==y2][['NOME','INDE']].rename(columns={'INDE':'INDE2'})
    m = d1.merge(d2, on='NOME', how='inner')
    m['queda'] = m['INDE2'] < m['INDE1']
    results.append({'Per': f'{y1}→{y2}', 'Queda': m[m['queda']]['IPS'].mean(), 'Sem': m[~m['queda']]['IPS'].mean()})
res_df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(res_df)); w = 0.35
ax.bar(x-w/2, res_df['Queda'], w, label='Com Queda', color=COLORS[3])
ax.bar(x+w/2, res_df['Sem'], w, label='Sem Queda', color=COLORS[2])
ax.set_xticks(x); ax.set_xticklabels(res_df['Per']); ax.set_title('IPS: Queda vs Sem Queda no INDE'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p5_ips_quedas.png'); plt.close()

# P6: IPP vs IAN
fig, ax = plt.subplots(figsize=(10, 5))
for i, year in enumerate(sorted(df['ANO'].unique())):
    s = df[df['ANO']==year].dropna(subset=['IPP','IAN'])
    ax.scatter(s['IPP'], s['IAN'], alpha=0.3, s=15, color=COLORS[i%len(COLORS)], label=str(int(year)))
ax.set_title('IPP vs IAN (2020-2024)'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p6_ipp_ian.png'); plt.close()

# P7: Influências no IPV
predictors = ['IDA','IEG','IAA','IPS','IPP','IAN']
corr_results = {}
for year in sorted(df['ANO'].unique()):
    s = df[df['ANO']==year].dropna(subset=['IPV']+predictors)
    corrs = {}
    for p in predictors:
        if len(s) > 2: r, _ = stats.pearsonr(s[p], s['IPV']); corrs[p] = r
    corr_results[int(year)] = corrs
fig, ax = plt.subplots(figsize=(12, 5))
pd.DataFrame(corr_results).T.plot(kind='bar', ax=ax, width=0.8, color=COLORS[:6])
ax.set_title('Correlação com IPV por Ano'); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p7_ipv_influencias.png'); plt.close()

# P8: Combinações que elevam INDE
corr_inde = {}
for p in numeric_cols:
    if p != 'INDE':
        s = df.dropna(subset=['INDE',p]); r, _ = stats.pearsonr(s[p], s['INDE']); corr_inde[p] = r
fig, ax = plt.subplots(figsize=(10, 5))
cs = pd.Series(corr_inde).sort_values(ascending=True)
colors_bar = [COLORS[2] if v>0.5 else COLORS[1] if v>0.3 else COLORS[0] for v in cs.values]
cs.plot(kind='barh', ax=ax, color=colors_bar)
ax.set_title('Correlação com INDE (2020-2024)'); ax.axvline(x=0.5, color='red', ls='--', alpha=0.5)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p8_inde_combinacoes.png'); plt.close()

# P9: Alunos em risco
risco = df[df['PEDRA']=='Quartzo']; nao_risco = df[df['PEDRA']!='Quartzo']
indicators = [c for c in numeric_cols if c != 'INDE']
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(indicators)); w = 0.35
ax.bar(x-w/2, [risco[c].mean() for c in indicators], w, label='Quartzo (Risco)', color=COLORS[3])
ax.bar(x+w/2, [nao_risco[c].mean() for c in indicators], w, label='Demais', color=COLORS[2])
ax.set_xticks(x); ax.set_xticklabels(indicators); ax.set_title('Perfil: Risco vs Demais'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p9_alunos_risco.png'); plt.close()

# P10: Impacto nas fases
all_trans = []
for y1, y2 in [(2020,2021),(2021,2022),(2022,2023),(2023,2024)]:
    d1 = df[df['ANO']==y1][['NOME','PEDRA']].rename(columns={'PEDRA':'P1'})
    d2 = df[df['ANO']==y2][['NOME','PEDRA']].rename(columns={'PEDRA':'P2'})
    m = d1.merge(d2, on='NOME', how='inner'); m['Per'] = f'{y1}→{y2}'
    all_trans.append(m)
trans = pd.concat(all_trans)
order_map = {'Quartzo':0,'Ágata':1,'Ametista':2,'Topázio':3}
trans['T'] = trans.apply(lambda r: 'Subiu' if order_map.get(str(r['P2']),-1)>order_map.get(str(r['P1']),-1) else ('Desceu' if order_map.get(str(r['P2']),-1)<order_map.get(str(r['P1']),-1) else 'Manteve'), axis=1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tp = trans.groupby(['Per','T']).size().unstack(fill_value=0)
tpp = tp.div(tp.sum(axis=1), axis=0)*100
tc = {'Subiu':COLORS[2],'Manteve':COLORS[0],'Desceu':COLORS[3]}
tpp.plot(kind='bar', stacked=True, ax=axes[0], color=[tc.get(c,'gray') for c in tpp.columns])
axes[0].set_title('Transições de Pedra'); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
inde_yr = df.groupby('ANO')['INDE'].mean()
axes[1].plot(inde_yr.index, inde_yr.values, 'o-', color=COLORS[4], lw=2, ms=8)
for x_v, y_v in zip(inde_yr.index, inde_yr.values): axes[1].text(x_v, y_v+0.1, f"{y_v:.2f}", ha='center', fontweight='bold')
axes[1].set_title('INDE Médio (2020-2024)'); axes[1].set_xticks(sorted(df['ANO'].unique()))
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p10_impacto_fases.png'); plt.close()

# P11: Distribuição de Pedras
pedra_yr = df.groupby(['ANO','PEDRA']).size().unstack(fill_value=0)
pedra_pct = pedra_yr.div(pedra_yr.sum(axis=1), axis=0)*100
fig, ax = plt.subplots(figsize=(12, 5))
cols_ord = [p for p in PEDRA_ORDER if p in pedra_pct.columns]
pedra_pct[cols_ord].plot(kind='bar', stacked=True, ax=ax, color=[PEDRA_COLORS[p] for p in cols_ord])
ax.set_title('Distribuição de Pedras por Ano (2020-2024)'); ax.set_xticklabels([str(int(y)) for y in pedra_pct.index], rotation=0)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}p11_progressao_pedra.png'); plt.close()

print("\nEDA COMPLETA - Gráficos salvos em /figures/")
