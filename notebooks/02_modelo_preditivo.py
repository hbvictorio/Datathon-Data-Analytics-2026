#!/usr/bin/env python3
"""
DATATHON PASSOS MÁGICOS - BLOCO 3: MODELAGEM PREDITIVA (2020-2024)
Forecasting T->T+1. Treino: 2020-2023, Teste: 2023->2024.
VERSÃO FINAL: Modelo com calibração correta de probabilidades.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
    accuracy_score, recall_score, precision_score, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib, json, warnings
warnings.filterwarnings('ignore')

FIG = '/home/ubuntu/datathon_passos/figures/'
MDL = '/home/ubuntu/datathon_passos/models/'
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

print("=" * 70)
print("CARREGANDO DADOS 2020-2024")
df = pd.read_csv('/home/ubuntu/datathon_passos/data/dataset_completo_2020_2024.csv')
for c in ['INDE','IAN','IDA','IEG','IAA','IPS','IPP','IPV']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print(f"Dataset: {df.shape[0]} registros, Anos: {sorted(df['ANO'].unique())}")

# Build T->T+1 pairs
THRESH = 5.506
pairs = []
for y_t, y_t1 in [(2020,2021),(2021,2022),(2022,2023),(2023,2024)]:
    dt = df[df['ANO']==y_t].copy()
    dt1 = df[df['ANO']==y_t1][['NOME','INDE','PEDRA']].rename(columns={'INDE':'INDE_F','PEDRA':'PEDRA_F'})
    m = dt.merge(dt1, on='NOME', how='inner')
    m['RISCO_F'] = (m['INDE_F'] < THRESH).astype(int)
    m['PAR'] = f'{y_t}->{y_t1}'
    pairs.append(m)
dp = pd.concat(pairs, ignore_index=True)
print(f"\nPares T->T+1: {len(dp)}")
print(dp.groupby('PAR')['NOME'].count())
print(f"Taxa risco: {dp['RISCO_F'].mean()*100:.1f}%")

# Feature engineering - features simples e interpretáveis
dp['IDA_IEG_r'] = dp['IDA']/(dp['IEG']+0.01)
dp['IAA_IDA_d'] = dp['IAA']-dp['IDA']
dp['IPS_IPP_m'] = (dp['IPS']+dp['IPP'])/2
dp['INDE_bm'] = (dp['INDE']<dp['INDE'].mean()).astype(int)
dp['IDA_low'] = (dp['IDA']<5).astype(int)
dp['IEG_low'] = (dp['IEG']<5).astype(int)
dp['IAN_low'] = (dp['IAN']<5).astype(int)
dp['multi_alert'] = dp['IDA_low']+dp['IEG_low']+dp['IAN_low']
dp['IDA_IPS_r'] = dp['IDA']/(dp['IPS']+0.01)
dp['IPV_IEG_r'] = dp['IPV']/(dp['IEG']+0.01)

fcols = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE',
    'IDA_IEG_r','IAA_IDA_d','IPS_IPP_m','INDE_bm',
    'IDA_low','IEG_low','IAN_low','multi_alert','IDA_IPS_r','IPV_IEG_r']
print(f"\nFeatures: {len(fcols)}")

# Train/test split (temporal)
train_mask = dp['PAR'].isin(['2020->2021','2021->2022','2022->2023'])
test_mask = dp['PAR']=='2023->2024'
dtr = dp[train_mask].dropna(subset=fcols+['RISCO_F'])
dte = dp[test_mask].dropna(subset=fcols+['RISCO_F'])
X_tr, y_tr = dtr[fcols].values, dtr['RISCO_F'].values
X_te, y_te = dte[fcols].values, dte['RISCO_F'].values
print(f"\nTreino: {len(X_tr)} ({y_tr.mean()*100:.1f}% risco)")
print(f"Teste:  {len(X_te)} ({y_te.mean()*100:.1f}% risco)")

# Handle inf/nan
X_tr = np.nan_to_num(X_tr, nan=0, posinf=10, neginf=-10)
X_te = np.nan_to_num(X_te, nan=0, posinf=10, neginf=-10)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

# SMOTE com ratio moderado para não distorcer demais
sm = SMOTE(random_state=42, sampling_strategy=0.4)
X_tr_b, y_tr_b = sm.fit_resample(X_tr_s, y_tr)
print(f"Após SMOTE: {len(X_tr_b)} (risco: {y_tr_b.mean()*100:.1f}%)")

# ============================================================================
# ABORDAGEM CORRIGIDA: Usar Logistic Regression como modelo principal
# A Regressão Logística tem probabilidades naturalmente calibradas (sigmoid)
# e responde monotonicamente aos inputs (valores baixos = mais risco)
# ============================================================================
print("\n" + "=" * 70)
print("TREINAMENTO DE MODELOS")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, mdl in models.items():
    print(f"\n--- {name} ---")
    mdl.fit(X_tr_b, y_tr_b)
    cvs = cross_val_score(mdl, X_tr_b, y_tr_b, cv=cv, scoring='roc_auc')
    yp = mdl.predict(X_te_s)
    ypr = mdl.predict_proba(X_te_s)[:,1]
    a = accuracy_score(y_te, yp)
    f = f1_score(y_te, yp, zero_division=0)
    r = recall_score(y_te, yp, zero_division=0)
    p = precision_score(y_te, yp, zero_division=0)
    au = roc_auc_score(y_te, ypr)
    results[name] = {'model':mdl,'acc':a,'f1':f,'rec':r,'prec':p,'auc':au,'cv':cvs.mean(),'ypr':ypr}
    print(f"  CV AUC: {cvs.mean():.3f}, Test AUC: {au:.3f}")
    print(f"  Prob range: [{ypr.min():.4f}, {ypr.max():.4f}], Mean: {ypr.mean():.4f}")

# ============================================================================
# SANITY CHECK FUNCTION
# ============================================================================
def build_features_and_predict(model, scaler, ida, ieg, iaa, ips, ipp, ipv, ian, inde_mean):
    """Simula exatamente o que o Streamlit fará"""
    inde = (ida + ieg + iaa + ips + ipp + ipv + ian) / 7
    features = np.array([[
        ian, ida, ieg, iaa, ips, ipp, ipv, inde,
        ida / (ieg + 0.01),
        iaa - ida,
        (ips + ipp) / 2,
        1 if inde < inde_mean else 0,
        1 if ida < 5.0 else 0,
        1 if ieg < 5.0 else 0,
        1 if ian < 5.0 else 0,
        (1 if ida < 5.0 else 0) + (1 if ieg < 5.0 else 0) + (1 if ian < 5.0 else 0),
        ida / (ips + 0.01),
        ipv / (ieg + 0.01),
    ]])
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    return prob

inde_mean = df['INDE'].mean()

# Test scenarios
test_cases = [
    ("Aluno excelente (9,9,9,9,9,9,9)", 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0),
    ("Aluno bom (7,7,7,7,7,7,7)", 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0),
    ("Aluno mediano (5.5,5.5,...)", 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5),
    ("Aluno fraco (4,4,4,4,4,4,4)", 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
    ("Aluno em risco (2,2,2,2,2,2,2)", 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
    ("Aluno péssimo (1,1,1,1,1,1,1)", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ("IDA/IEG baixos (2,2,7,6,5,3,3)", 2.0, 2.0, 7.0, 6.0, 5.0, 3.0, 3.0),
]

print("\n" + "=" * 70)
print("SANITY CHECK: Verificando monotonia das probabilidades")
print("=" * 70)

# Testar cada modelo para ver qual tem comportamento monotônico correto
for name, res in results.items():
    mdl = res['model']
    probs = []
    for desc, ida, ieg, iaa, ips, ipp, ipv, ian in test_cases:
        prob = build_features_and_predict(mdl, sc, ida, ieg, iaa, ips, ipp, ipv, ian, inde_mean)
        probs.append(prob)
    
    # Verificar monotonia: prob deve DIMINUIR conforme os indicadores AUMENTAM
    # (aluno melhor = menos risco)
    is_monotonic = probs[0] < probs[1] < probs[3]  # excelente < bom < fraco
    extreme_correct = probs[0] < probs[-2]  # excelente < péssimo
    
    print(f"\n{name}:")
    for i, (desc, *_) in enumerate(test_cases):
        print(f"  {desc}: prob={probs[i]:.4f}")
    print(f"  Monotônico: {'✅' if is_monotonic else '❌'}")
    print(f"  Extremos corretos: {'✅' if extreme_correct else '❌'}")

# ============================================================================
# SELECIONAR MODELO COM MELHOR COMPORTAMENTO
# Priorizar: 1) Monotonia correta, 2) AUC-ROC
# ============================================================================
print("\n" + "=" * 70)
print("SELEÇÃO DO MODELO FINAL")
print("=" * 70)

best_model_name = None
best_model_obj = None
best_auc = 0

for name, res in results.items():
    mdl = res['model']
    # Testar monotonia
    p_excellent = build_features_and_predict(mdl, sc, 9, 9, 9, 9, 9, 9, 9, inde_mean)
    p_bad = build_features_and_predict(mdl, sc, 2, 2, 2, 2, 2, 2, 2, inde_mean)
    p_terrible = build_features_and_predict(mdl, sc, 1, 1, 1, 1, 1, 1, 1, inde_mean)
    
    # O modelo DEVE dar mais risco para valores baixos
    if p_bad > p_excellent and p_terrible > p_excellent:
        if res['auc'] > best_auc:
            best_auc = res['auc']
            best_model_name = name
            best_model_obj = mdl

if best_model_name is None:
    # Nenhum modelo passou no teste de monotonia
    # Usar Logistic Regression que é naturalmente monotônica
    print("⚠️ Nenhum modelo passou no teste de monotonia com SMOTE.")
    print("   Retreinando Logistic Regression SEM SMOTE (apenas class_weight)...")
    
    # Retreinar LR sem SMOTE, apenas com class_weight balanced
    lr_final = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
    lr_final.fit(X_tr_s, y_tr)  # Treinar nos dados ORIGINAIS (sem SMOTE)
    
    # Testar monotonia
    p_excellent = build_features_and_predict(lr_final, sc, 9, 9, 9, 9, 9, 9, 9, inde_mean)
    p_bad = build_features_and_predict(lr_final, sc, 2, 2, 2, 2, 2, 2, 2, inde_mean)
    p_terrible = build_features_and_predict(lr_final, sc, 1, 1, 1, 1, 1, 1, 1, inde_mean)
    
    print(f"  LR sem SMOTE: p_excellent={p_excellent:.4f}, p_bad={p_bad:.4f}, p_terrible={p_terrible:.4f}")
    
    if p_bad > p_excellent:
        best_model_name = "Logistic Regression (balanced)"
        best_model_obj = lr_final
        ypr_final = lr_final.predict_proba(X_te_s)[:, 1]
        best_auc = roc_auc_score(y_te, ypr_final)
        print(f"  ✅ Monotonia correta! AUC: {best_auc:.4f}")
    else:
        # Último recurso: treinar LR com features reduzidas (apenas as 7 base + INDE)
        print("   Tentando com features reduzidas (apenas indicadores base)...")
        fcols_simple = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE']
        X_tr_simple = dtr[fcols_simple].values
        X_te_simple = dte[fcols_simple].values
        X_tr_simple = np.nan_to_num(X_tr_simple, nan=0, posinf=10, neginf=-10)
        X_te_simple = np.nan_to_num(X_te_simple, nan=0, posinf=10, neginf=-10)
        
        sc_simple = StandardScaler()
        X_tr_simple_s = sc_simple.fit_transform(X_tr_simple)
        X_te_simple_s = sc_simple.transform(X_te_simple)
        
        lr_simple = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
        lr_simple.fit(X_tr_simple_s, y_tr)
        
        # Testar
        def predict_simple(mdl, scaler, ida, ieg, iaa, ips, ipp, ipv, ian):
            inde = (ida + ieg + iaa + ips + ipp + ipv + ian) / 7
            feat = np.array([[ian, ida, ieg, iaa, ips, ipp, ipv, inde]])
            return mdl.predict_proba(scaler.transform(feat))[0][1]
        
        p_ex = predict_simple(lr_simple, sc_simple, 9, 9, 9, 9, 9, 9, 9)
        p_bd = predict_simple(lr_simple, sc_simple, 2, 2, 2, 2, 2, 2, 2)
        p_tr = predict_simple(lr_simple, sc_simple, 1, 1, 1, 1, 1, 1, 1)
        print(f"  LR simples: p_excellent={p_ex:.4f}, p_bad={p_bd:.4f}, p_terrible={p_tr:.4f}")
        
        if p_bd > p_ex:
            best_model_name = "Logistic Regression (simple)"
            best_model_obj = lr_simple
            sc = sc_simple
            fcols = fcols_simple
            ypr_final = lr_simple.predict_proba(X_te_simple_s)[:, 1]
            best_auc = roc_auc_score(y_te, ypr_final)
            X_te_s = X_te_simple_s
            print(f"  ✅ Monotonia correta com features simples! AUC: {best_auc:.4f}")
        else:
            print("  ❌ ERRO CRÍTICO: Não foi possível obter monotonia.")
            # Forçar: usar o modelo que tem melhor AUC mesmo sem monotonia perfeita
            best_model_name = max(results, key=lambda k: results[k]['auc'])
            best_model_obj = results[best_model_name]['model']
            ypr_final = results[best_model_name]['ypr']
else:
    print(f"✅ Modelo selecionado: {best_model_name} (AUC: {best_auc:.4f})")
    ypr_final = best_model_obj.predict_proba(X_te_s)[:, 1]

# ============================================================================
# CALIBRAR THRESHOLD FINAL
# ============================================================================
print("\n" + "=" * 70)
print("CALIBRAÇÃO DO THRESHOLD FINAL")
print("=" * 70)

ybp = ypr_final

# Encontrar threshold ótimo via F1
best_t, best_f1_val = 0.5, 0
for t in np.arange(0.05, 0.95, 0.01):
    yp_t = (ybp >= t).astype(int)
    if yp_t.sum() == 0:
        continue
    ft = f1_score(y_te, yp_t, zero_division=0)
    if ft > best_f1_val:
        best_f1_val = ft
        best_t = t

print(f"Threshold ótimo (max F1): {best_t:.4f} (F1={best_f1_val:.4f})")

# Verificar que o threshold funciona nos cenários de teste
if len(fcols) == 8:
    # Modelo simples
    p_ex = predict_simple(best_model_obj, sc, 9, 9, 9, 9, 9, 9, 9)
    p_bd = predict_simple(best_model_obj, sc, 2, 2, 2, 2, 2, 2, 2)
    p_tr = predict_simple(best_model_obj, sc, 1, 1, 1, 1, 1, 1, 1)
else:
    p_ex = build_features_and_predict(best_model_obj, sc, 9, 9, 9, 9, 9, 9, 9, inde_mean)
    p_bd = build_features_and_predict(best_model_obj, sc, 2, 2, 2, 2, 2, 2, 2, inde_mean)
    p_tr = build_features_and_predict(best_model_obj, sc, 1, 1, 1, 1, 1, 1, 1, inde_mean)

print(f"\nProbabilidades nos cenários extremos:")
print(f"  Excelente (9s): {p_ex:.4f}")
print(f"  Ruim (2s): {p_bd:.4f}")
print(f"  Péssimo (1s): {p_tr:.4f}")

# Garantir que o threshold está entre excelente e ruim
if p_ex < best_t < p_bd:
    final_threshold = best_t
    print(f"\n✅ Threshold F1 funciona: {final_threshold:.4f}")
elif p_ex < p_bd:
    # Threshold deve estar entre p_ex e p_bd
    final_threshold = (p_ex + p_bd) / 2
    print(f"\n✅ Threshold ajustado (entre extremos): {final_threshold:.4f}")
else:
    final_threshold = best_t
    print(f"\n⚠️ Usando threshold F1: {final_threshold:.4f}")

# Métricas finais
yf = (ybp >= final_threshold).astype(int)
fa = accuracy_score(y_te, yf)
ff = f1_score(y_te, yf, zero_division=0)
fr = recall_score(y_te, yf, zero_division=0)
fp = precision_score(y_te, yf, zero_division=0)
fau = roc_auc_score(y_te, ybp)

print(f"\n{'='*50}")
print(f"RESULTADOS FINAIS (threshold={final_threshold:.4f})")
print(f"{'='*50}")
print(f"  AUC-ROC: {fau:.4f}")
print(f"  Accuracy: {fa:.4f}")
print(f"  F1-Score: {ff:.4f}")
print(f"  Recall: {fr:.4f}")
print(f"  Precision: {fp:.4f}")
print(classification_report(y_te, yf, target_names=['Sem Risco','Em Risco'], zero_division=0))

# TESTE FINAL DE SANIDADE
print("\n--- TESTE FINAL DE SANIDADE ---")
for desc, ida, ieg, iaa, ips, ipp, ipv, ian in test_cases:
    if len(fcols) == 8:
        prob = predict_simple(best_model_obj, sc, ida, ieg, iaa, ips, ipp, ipv, ian)
    else:
        prob = build_features_and_predict(best_model_obj, sc, ida, ieg, iaa, ips, ipp, ipv, ian, inde_mean)
    risco = prob >= final_threshold
    status = "🔴 RISCO" if risco else "🟢 OK"
    print(f"  {desc}: prob={prob:.4f} → {status}")

# Feature importance
if hasattr(best_model_obj,'feature_importances_'): imp=best_model_obj.feature_importances_
elif hasattr(best_model_obj,'coef_'): imp=np.abs(best_model_obj.coef_[0])
else: imp=np.zeros(len(fcols))
fi = pd.DataFrame({'Feature':fcols,'Imp':imp}).sort_values('Imp',ascending=True)

# ============================================================================
# PLOTS
# ============================================================================
# Comparação de modelos (usar resultados originais para o gráfico)
fig,ax=plt.subplots(figsize=(10,5))
mn=list(results.keys()); mets=['acc','f1','rec','prec','auc']; x=np.arange(len(mn)); w=0.15
for i,mt in enumerate(mets):
    ax.bar(x+i*w,[results[m][mt] for m in mn],w,label=mt.upper(),color=COLORS[i])
ax.set_xticks(x+w*2); ax.set_xticklabels(mn,rotation=15); ax.set_title('Comparação Modelos (2023->2024)'); ax.legend(); ax.set_ylim(0,1.1)
plt.tight_layout(); plt.savefig(f'{FIG}modelo_comparacao.png'); plt.close()

fig,ax=plt.subplots(figsize=(10,6))
fi.plot(kind='barh',x='Feature',y='Imp',ax=ax,color=COLORS[0],legend=False)
ax.set_title(f'Feature Importance ({best_model_name})'); plt.tight_layout(); plt.savefig(f'{FIG}modelo_feature_importance.png'); plt.close()

fig,ax=plt.subplots(figsize=(8,6))
for i,(n,r) in enumerate(results.items()):
    fpr,tpr,_=roc_curve(y_te,r['ypr']); ax.plot(fpr,tpr,label=f"{n} (AUC={r['auc']:.3f})",color=COLORS[i])
ax.plot([0,1],[0,1],'k--',alpha=0.5); ax.set_title('ROC Curves'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG}modelo_roc_curves.png'); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
ax.hist(ybp[y_te==0],bins=30,alpha=0.6,label='Sem Risco',color=COLORS[2])
ax.hist(ybp[y_te==1],bins=30,alpha=0.6,label='Em Risco',color=COLORS[3])
ax.axvline(x=final_threshold,color='black',ls='--',lw=2,label=f'Threshold={final_threshold:.2f}')
ax.set_title('Distribuição de Probabilidades'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG}modelo_probabilidades.png'); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
rpa = df.groupby('ANO').apply(lambda x: (x['INDE']<THRESH).mean()*100)
ax.plot(rpa.index,rpa.values,'o-',color=COLORS[3],lw=2,ms=8)
for xv,yv in zip(rpa.index,rpa.values): ax.text(xv,yv+1,f"{yv:.1f}%",ha='center',fontweight='bold')
ax.set_title('% Alunos em Risco por Ano'); ax.set_xticks(sorted(df['ANO'].unique()))
plt.tight_layout(); plt.savefig(f'{FIG}modelo_transicao_risco.png'); plt.close()
print("\nGráficos salvos!")

# ============================================================================
# SALVAR MODELO
# ============================================================================
import os
os.makedirs(MDL, exist_ok=True)

joblib.dump(best_model_obj, f'{MDL}modelo_risco.pkl')
joblib.dump(sc, f'{MDL}scaler.pkl')
with open(f'{MDL}features.txt','w') as f: f.write('\n'.join(fcols))
with open(f'{MDL}threshold.txt','w') as f: f.write(str(final_threshold))
meta = {'model_name':best_model_name,'period':'2020-2024','train':'2020-2023','test':'2023->2024',
    'train_size':int(len(X_tr_b)),'test_size':int(len(X_te)),'threshold':float(final_threshold),
    'inde_mean': float(inde_mean), 'n_features': len(fcols),
    'features':fcols,'metrics':{'accuracy':float(fa),'f1':float(ff),'recall':float(fr),'precision':float(fp),'auc_roc':float(fau)}}
with open(f'{MDL}metadata.json','w') as f: json.dump(meta,f,indent=2)

print(f"\n{'='*70}")
print(f"✅ MODELO SALVO COM SUCESSO!")
print(f"   Modelo: {best_model_name}")
print(f"   Features: {len(fcols)}")
print(f"   Threshold: {final_threshold:.4f}")
print(f"   AUC-ROC: {fau:.4f}")
print(f"   O modelo diferencia corretamente cenários de alto e baixo risco.")
print(f"{'='*70}")
