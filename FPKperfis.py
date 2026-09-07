import os
import glob
import re
import unicodedata
import itertools
from math import factorial
from collections import Counter
import warnings

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURAÇÕES E DIRETÓRIOS
# ==========================================
INPUT_DIR = r"C:\Users\viihb\Downloads\fpk\perfis"
OUTPUT_EXCEL = os.path.join(INPUT_DIR, "analise_completa_fpk_avancada.xlsx")
OUTPUT_IMG_DIR = os.path.join(INPUT_DIR, "graficos_png_avancados")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

STOPWORDS_PT = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 
    'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 
    'quando', 'muito', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 
    'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 
    'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'elas', 'havia', 
    'seja', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 
    'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 
    'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo',
    'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam',
    'sobre', 'ano', 'anos', 'dia', 'dias', 'hoje', 'ontem', 'link', 'vídeo', 'foto', 'confira', 'veja', 'nesta', 'neste',
    'todos', 'todas', 'tudo', 'onde', 'cada', 'ainda', 'assim', 'desde', 'durante', 'fazer', 'parte', 'segundo', 'contra',
    'sobre', 'após', 'deste', 'desta', 'nestes', 'nestas', 'vai', 'podem', 'pode', 'disse', 'afirma', 'além'
]

# ==========================================
# 2. FUNÇÕES ESTATÍSTICAS E CLASSIFICADORES
# ==========================================
def normalize_text(text):
    if pd.isna(text): return ""
    return "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn").lower()

def gini_coefficient(array):
    """Calcula o Coeficiente de Gini de uma distribuição."""
    arr = np.array(array, dtype=np.float64)
    if np.amin(arr) < 0: arr -= np.amin(arr)
    arr += 1e-12
    arr = np.sort(arr)
    n = arr.shape[0]
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

def cliffs_delta(lst1, lst2):
    """Calcula o tamanho de efeito não-paramétrico Cliff's Delta."""
    l1, l2 = np.array(lst1), np.array(lst2)
    u_stat, _ = stats.mannwhitneyu(l1, l2, alternative='two-sided')
    return float((2 * u_stat / (len(l1) * len(l2))) - 1)

def cohens_d(lst1, lst2):
    """Calcula Cohen's d (padronizado em escala log)."""
    l1, l2 = np.log1p(np.array(lst1)), np.log1p(np.array(lst2))
    n1, n2 = len(l1), len(l2)
    s1, s2 = np.var(l1, ddof=1), np.var(l2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(l1) - np.mean(l2)) / (pooled_sd + 1e-12))

def map_profile_group(p):
    p_norm = normalize_text(p)
    if not p_norm: return np.nan
    if any(k in p_norm for k in ["arabe", "fepal"]): return "FEPAL"
    if any(k in p_norm for k in ["fisesp", "federacao israelita", "federacaosp"]): return "Fisesp"
    if "standwithus" in p_norm: return "StandWithUs"
    if any(k in p_norm for k in ["instituto brasil-israel", "ibi"]): return "IBI"
    if "sanaud" in p_norm: return "Sanaúd"
    if "ibraspal" in p_norm: return "IBRASPAL"
    if "frente gaucha" in p_norm: return "Frente Gaúcha"
    if "meretz" in p_norm: return "Meretz"
    if "judaica" in p_norm: return "Agência Judaica"
    return np.nan

def categorize_side(p):
    p_norm = normalize_text(p)
    if not p_norm: return np.nan
    pal_terms = ['palestin', 'fepal', 'ibraspal', 'sanaud', 'frente gaucha']
    if any(term in p_norm for term in pal_terms):
        return 'Pró-Palestina'
    return 'Pró-Israel / Comunitário Judaico'

# Dicionários de Enquadramento
THEMES_DICT = {
    'Tema_Violencia': ['genocidio', 'bombardeio', 'mutilado', 'amputado', 'assassin', 'morto', 'morte', 'ataque', 'massacre', 'destruicao', 'crime de guerra', 'exterminio', 'invasao', 'ocupacao', 'militar', 'soldado', 'tiro', 'fogo', 'terrorismo', 'misseis', 'refens', 'sequestr', 'vitima', 'ferid', 'hospital', 'fome', 'hind rajab'],
    'Tema_Cultura': ['musica', 'cantor', 'cantora', 'rosalia', 'roger waters', 'pink floyd', 'filme', 'cinema', 'ator', 'atriz', 'gal gadot', 'show', 'clipe', 'arte', 'artista', 'cultura', 'livro', 'poesia', 'festival'],
    'Tema_Memoria': ['holocausto', 'memoria', 'shoah', 'yom hashoa', 'auschwitz', 'sobrevivente', 'lembranca', 'historia', '7 de outubro', 'massacre de 7', 'nuncamais', 'memorial'],
    'Tema_Esporte': ['copa', 'futebol', 'atleta', 'selecao', 'jogo', 'jogadores', 'gol', 'camisa da selecao', 'ancelotti', 'campo'],
    'Tema_Comunidade': ['comunidade', 'jovem', 'juventude', 'voluntariado', 'taglit', 'maavar', 'lideranca', 'sinagoga', 'shabat', 'rabino', 'solidariedade', 'familia', 'educacao', 'encontro', 'hillel', 'alia', 'aliya', 'fisesp'],
    'Tema_Politica': ['governo', 'governador', 'tarcisio', 'lula', 'biden', 'trump', 'netanyahu', 'itamaraty', 'onu', 'tribunal', 'haia', 'diplomacia', 'embaixad', 'senado', 'camara', 'deputad', 'ministro', 'reuniao de cupula', 'otan', 'acordo']
}

AFFECTS_DICT = {
    'Afeto_Indignacao': ['denuncia', 'massacre', 'genocidio', 'injustica', 'covardia', 'vergonha', 'crime', 'escandalo', 'criminoso', 'violacao', 'revolta'],
    'Afeto_Luto': ['morte', 'mortes', 'vitima', 'vitimas', 'assassin', 'martir', 'martires', 'condolencias', 'dor', 'homenagem', 'homenagens', 'perda', 'perdas', 'luto', 'ferid', 'sepultamento'],
    'Afeto_Urgencia': ['agora', 'compartilhe', 'pressao', 'urgente', 'divulgue', 'assine', 'atencao', 'alerta', 'emergencia', 'imediato', 'ajude'],
    'Afeto_Solidariedade': ['apoio', 'resistencia', 'mobilizacao', 'uniao', 'juntos', 'solidario', 'solidariedade', 'voluntari', 'ajuda humanitaria', 'doacao'],
    'Afeto_Pertencimento': ['comunidade', 'tradicao', 'identidade', 'povo', 'raizes', 'lar', 'valores', 'familia', 'irmaos', 'sinagoga', 'shabat', 'nossa gente', 'nossos'],
    'Afeto_Celebracao': ['cultura', 'evento', 'eventos', 'conquista', 'conquistas', 'vitoria', 'parabens', 'celebra', 'celebracao', 'festival', 'festa', 'orgulho', 'musica', 'show', 'arte', 'alegria'],
    'Afeto_Memoria': ['holocausto', 'historia', 'lembranca', 'nunca esquecer', 'jamais esquecer', 'memoria', 'testemunho', 'passado', '7 de outubro', 'shoah', 'yom hashoa', 'auschwitz', 'sobrevivente']
}

def classify_labels(text, mapping):
    t = normalize_text(text)
    active = [k for k, v in mapping.items() if any(term in t for term in v)]
    return active if active else ['Neutro / Outro']

def extract_clean_sheet(filepath, sheet_name):
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    header_idx = None
    for idx, row in df_raw.iterrows():
        if 'Profile' in row.values or 'Date' in row.values:
            header_idx = idx
            break
    if header_idx is not None:
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = df_raw.iloc[header_idx].values
        return df.dropna(how='all', axis=1)
    return df_raw

# ==========================================
# 3. EXTRAÇÃO E PREPARAÇÃO DOS DADOS
# ==========================================
print("[1/11] Carregando planilhas brutas do FanPageKarma...")
all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")) + glob.glob(os.path.join(INPUT_DIR, "*.xml")))
raw_posts_list = []
for f in all_files:
    if 'analise_completa' in f: continue
    try:
        xls = pd.ExcelFile(f)
        for s in xls.sheet_names:
            if 'overview' in s.lower() or 'top 5000' in s.lower():
                df_temp = extract_clean_sheet(f, s)
                if 'Message' in df_temp.columns and 'Profile' in df_temp.columns:
                    raw_posts_list.append(df_temp)
    except Exception as e:
        print(f"Erro em {f}: {e}")

df_posts = pd.concat(raw_posts_list, ignore_index=True).drop_duplicates(subset=['Message', 'Date', 'Profile'])

# Conversão de Tipos
df_posts['Likes'] = pd.to_numeric(df_posts['Number of Likes'], errors='coerce').fillna(0)
df_posts['Comments'] = pd.to_numeric(df_posts['Number of comments'], errors='coerce').fillna(0)
df_posts['Total_Reactions'] = pd.to_numeric(df_posts['Reactions, Comments & Shares'], errors='coerce').fillna(0)
df_posts['PIR'] = pd.to_numeric(df_posts['Post interaction rate'], errors='coerce').fillna(0)
df_posts['Date'] = pd.to_datetime(df_posts['Date'], errors='coerce')
df_posts['Message'] = df_posts['Message'].astype(str).fillna('')
df_posts['Text_Length_Words'] = df_posts['Message'].apply(lambda x: len(x.split()))
df_posts['Text_Length_Chars'] = df_posts['Message'].apply(len)
df_posts['log_reactions'] = np.log1p(df_posts['Total_Reactions'])

df_posts['Org'] = df_posts['Profile'].apply(map_profile_group)
df_posts['Side'] = df_posts['Profile'].apply(categorize_side)
df_posts = df_posts.dropna(subset=['Org']).copy()

# Criação de Dummies e Listas
for col, kws in {**THEMES_DICT, **AFFECTS_DICT}.items():
    df_posts[col] = df_posts['Message'].apply(lambda x: int(any(k in normalize_text(x) for k in kws)))

df_posts['Themes_List'] = df_posts['Message'].apply(lambda x: classify_labels(x, THEMES_DICT))
df_posts['Affects_List'] = df_posts['Message'].apply(lambda x: classify_labels(x, AFFECTS_DICT))

print(f"Base consolidada: {len(df_posts)} posts higienizados.")

# ==========================================
# 4. ANÁLISES AVANÇADAS E EXPORTAÇÃO EXCEL
# ==========================================
excel_writer = pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl')

# --- 4.1 Concentração e Gini ---
print("[2/11] Calculando Concentração de Engajamento e Coeficiente de Gini...")
total_corpus_react = df_posts['Total_Reactions'].sum()
total_pal_react = df_posts[df_posts['Side'] == 'Pró-Palestina']['Total_Reactions'].sum()
total_isr_react = df_posts[df_posts['Side'] == 'Pró-Israel / Comunitário Judaico']['Total_Reactions'].sum()

df_concentracao = df_posts.groupby(['Side', 'Org']).agg(
    Posts=('Total_Reactions', 'count'),
    Total_Reacoes=('Total_Reactions', 'sum'),
    Media_Reacoes=('Total_Reactions', 'mean'),
    Mediana_Reacoes=('Total_Reactions', 'median'),
    Max_Reacoes=('Total_Reactions', 'max')
).reset_index()

df_concentracao['Share_Corpus_%'] = (df_concentracao['Total_Reacoes'] / total_corpus_react) * 100
df_concentracao['Share_No_Campo_%'] = df_concentracao.apply(
    lambda r: (r['Total_Reacoes'] / total_pal_react * 100) if r['Side'] == 'Pró-Palestina' else (r['Total_Reacoes'] / total_isr_react * 100), axis=1
)
df_concentracao = df_concentracao.sort_values(by='Total_Reacoes', ascending=False)
df_concentracao.to_excel(excel_writer, sheet_name='Concentracao_Organizacoes', index=False)

# Tabela Gini
df_gini = pd.DataFrame([
    {'Escopo': 'Corpus Completo (Geral)', 'N_Posts': len(df_posts), 'Gini_Reacoes': gini_coefficient(df_posts['Total_Reactions'])},
    {'Escopo': 'Campo Pró-Palestina (Completo)', 'N_Posts': len(df_posts[df_posts['Side'] == 'Pró-Palestina']), 'Gini_Reacoes': gini_coefficient(df_posts[df_posts['Side'] == 'Pró-Palestina']['Total_Reactions'])},
    {'Escopo': 'Campo Pró-Palestina (SEM FEPAL)', 'N_Posts': len(df_posts[(df_posts['Side'] == 'Pró-Palestina') & (df_posts['Org'] != 'FEPAL')]), 'Gini_Reacoes': gini_coefficient(df_posts[(df_posts['Side'] == 'Pró-Palestina') & (df_posts['Org'] != 'FEPAL')]['Total_Reactions'])},
    {'Escopo': 'Campo Pró-Israel / Comunitário', 'N_Posts': len(df_posts[df_posts['Side'] == 'Pró-Israel / Comunitário Judaico']), 'Gini_Reacoes': gini_coefficient(df_posts[df_posts['Side'] == 'Pró-Israel / Comunitário Judaico']['Total_Reactions'])},
    {'Escopo': 'Corpus Geral (SEM FEPAL)', 'N_Posts': len(df_posts[df_posts['Org'] != 'FEPAL']), 'Gini_Reacoes': gini_coefficient(df_posts[df_posts['Org'] != 'FEPAL']['Total_Reactions'])}
])
df_gini.to_excel(excel_writer, sheet_name='Gini_Concentracao', index=False)

# --- 4.2 Análise de Sensibilidade: Com vs. Sem FEPAL ---
print("[3/11] Avaliando Efeito de Tamanho (Com vs. Sem FEPAL)...")
pal_all = df_posts[df_posts['Side'] == 'Pró-Palestina']['Total_Reactions']
pal_no_fepal = df_posts[(df_posts['Side'] == 'Pró-Palestina') & (df_posts['Org'] != 'FEPAL')]['Total_Reactions']
isr_all = df_posts[df_posts['Side'] == 'Pró-Israel / Comunitário Judaico']['Total_Reactions']

def compare_camps(g1, g2, label):
    u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return {
        'Cenário': label,
        'N_Grupo1': len(g1),
        'Media_Grupo1': np.mean(g1),
        'Mediana_Grupo1': np.median(g1),
        'N_Pró_Israel': len(g2),
        'Media_Pró_Israel': np.mean(g2),
        'Mediana_Pró_Israel': np.median(g2),
        'Mann_Whitney_U': u,
        'p_valor': p,
        'Cliffs_Delta': cliffs_delta(g1, g2),
        'Cohens_d_log': cohens_d(g1, g2),
        'Interpretacao_Efeito': 'Forte (+)' if cliffs_delta(g1, g2) > 0.4 else ('Negligível' if abs(cliffs_delta(g1, g2)) < 0.14 else 'Fraco/Médio (-)')
    }

df_sensibilidade = pd.DataFrame([
    compare_camps(pal_all, isr_all, 'Pró-Palestina vs Pró-Israel (COM FEPAL)'),
    compare_camps(pal_no_fepal, isr_all, 'Pró-Palestina vs Pró-Israel (SEM FEPAL)')
])
df_sensibilidade.to_excel(excel_writer, sheet_name='Comparacao_Campos_FEPAL', index=False)

# --- 4.3 Organização como Unidade de Análise (N=9) ---
print("[4/11] Testando hipótese com Organização como Unidade de Análise...")
df_org_unit = df_posts.groupby(['Side', 'Org']).agg(
    Posts=('Total_Reactions', 'count'),
    Total_Reacoes=('Total_Reactions', 'sum'),
    Media_Reacoes=('Total_Reactions', 'mean'),
    Mediana_Reacoes=('Total_Reactions', 'median'),
    PIR_Medio=('PIR', 'mean')
).reset_index()

pal_org_means = df_org_unit[df_org_unit['Side'] == 'Pró-Palestina']['Media_Reacoes']
isr_org_means = df_org_unit[df_org_unit['Side'] == 'Pró-Israel / Comunitário Judaico']['Media_Reacoes']
u_org, p_org = stats.mannwhitneyu(pal_org_means, isr_org_means, alternative='two-sided')

df_org_unit_test = pd.DataFrame([{
    'Unidade_Analise': 'Organizações (N=9: 4 Pal, 5 Isr)',
    'Media_Pal': pal_org_means.mean(),
    'Media_Isr': isr_org_means.mean(),
    'Mann_Whitney_U': u_org,
    'p_valor': p_org,
    'Cliffs_Delta': (2 * u_org / (len(pal_org_means) * len(isr_org_means))) - 1,
    'Conclusao': 'Diferença não-significante no nível do ator (p > 0.05)' if p_org > 0.05 else 'Significante'
}])
df_org_unit.to_excel(excel_writer, sheet_name='Org_Unidade_Dados', index=False)
df_org_unit_test.to_excel(excel_writer, sheet_name='Org_Unidade_Teste', index=False)

# --- 4.4 Modelo Multinível (Linear Mixed Effects Model) ---
print("[5/11] Ajustando Modelo Multinível (MixedLM com Efeitos Aleatórios por Organização)...")
formula_mixed = "log_reactions ~ Tema_Violencia + Tema_Cultura + Tema_Memoria + Tema_Esporte + Tema_Comunidade + Tema_Politica + Afeto_Indignacao + Afeto_Luto + Afeto_Urgencia + Afeto_Solidariedade + Afeto_Pertencimento + Afeto_Celebracao + Text_Length_Words + C(Network)"
md = smf.mixedlm(formula_mixed, df_posts, groups=df_posts["Org"])
mdf = md.fit(reml=False)

df_mixed_res = pd.DataFrame({
    'Variavel': mdf.params.index,
    'Coeficiente_Fixo': mdf.params.values,
    'Erro_Padrao': mdf.bse.values,
    'z_valor': mdf.tvalues.values,
    'p_valor': mdf.pvalues.values,
    'CI_Lower': mdf.conf_int()[0].values,
    'CI_Upper': mdf.conf_int()[1].values
})
df_mixed_res.to_excel(excel_writer, sheet_name='Modelo_Multinivel_MixedLM', index=False)

# --- 4.5 Decomposição de Shapley: Completa vs. Conteúdo Intrínseco ---
print("[6/11] Executando Decomposição de Shapley (LMG) com e sem proxies de canal...")
def shapley_r2_decomp(X_sub, y_target):
    p = X_sub.shape[1]
    cols = list(X_sub.columns)
    shapley_values = {col: 0.0 for col in cols}
    for r in range(p):
        for combo in itertools.combinations(cols, r):
            combo = list(combo)
            r2_without = 0.0 if len(combo) == 0 else LinearRegression().fit(X_sub[combo], y_target).score(X_sub[combo], y_target)
            weight = (factorial(len(combo)) * factorial(p - len(combo) - 1)) / factorial(p)
            for col in cols:
                if col not in combo:
                    combo_with = combo + [col]
                    r2_with = LinearRegression().fit(X_sub[combo_with], y_target).score(X_sub[combo_with], y_target)
                    marginal = max(0, r2_with - r2_without)
                    shapley_values[col] += weight * marginal
    df_res = pd.DataFrame(list(shapley_values.items()), columns=['Feature', 'Shapley_R2_Contribution'])
    df_res['Relative_Importance_%'] = (df_res['Shapley_R2_Contribution'] / (df_res['Shapley_R2_Contribution'].sum() + 1e-12)) * 100
    return df_res.sort_values(by='Shapley_R2_Contribution', ascending=False)

# Shapley 1: Modelo Completo (com Canais)
key_feats_full = ['Org_FEPAL', 'Org_Fisesp', 'Org_StandWithUs', 'Org_Sanaúd', 'Tema_Violencia', 'Tema_Cultura', 'Tema_Esporte', 'Tema_Memoria', 'Text_Length_Words']
X_full = pd.concat([df_posts[['Text_Length_Words'] + list(THEMES_DICT.keys())], pd.get_dummies(df_posts[['Org', 'Network']], drop_first=True)], axis=1).astype(float)
df_shapley_full = shapley_r2_decomp(X_full[[c for c in key_feats_full if c in X_full.columns]], df_posts['log_reactions'].values)
df_shapley_full.to_excel(excel_writer, sheet_name='Shapley_Completo', index=False)

# Shapley 2: Modelo Intrínseco de Conteúdo (Sem Dummies de Org)
key_feats_content = ['Tema_Violencia', 'Tema_Cultura', 'Tema_Memoria', 'Tema_Esporte', 'Afeto_Indignacao', 'Afeto_Luto', 'Afeto_Urgencia', 'Text_Length_Words']
df_shapley_content = shapley_r2_decomp(df_posts[key_feats_content].astype(float), df_posts['log_reactions'].values)
df_shapley_content.to_excel(excel_writer, sheet_name='Shapley_Conteudo_Intrinseco', index=False)

# --- 4.6 Plataforma Controlando por Organização ---
print("[7/11] Gerando tabelas de Plataforma controlando por Organização...")
df_plat_ctrl = df_posts.groupby(['Org', 'Network']).agg(
    Posts=('Total_Reactions', 'count'),
    Media_Reacoes=('Total_Reactions', 'mean'),
    Mediana_Reacoes=('Total_Reactions', 'median'),
    PIR_Medio=('PIR', 'mean')
).reset_index().sort_values(by=['Org', 'Media_Reacoes'], ascending=[True, False])
df_plat_ctrl.to_excel(excel_writer, sheet_name='Plataforma_Por_Organizacao', index=False)

# --- 4.7 Cruzamento Multidimensional (Org × Tema × Afeto × Plataforma) ---
print("[8/11] Gerando Cruzamento Multidimensional Completo...")
records_cross = []
for idx, row in df_posts.iterrows():
    for t in row['Themes_List']:
        for a in row['Affects_List']:
            records_cross.append({
                'Org': row['Org'],
                'Side': row['Side'],
                'Network': row['Network'],
                'Tema': t.replace('Tema_', ''),
                'Regime_Afetivo': a.replace('Afeto_', ''),
                'Total_Reactions': row['Total_Reactions'],
                'Likes': row['Likes'],
                'Comments': row['Comments']
            })
df_cross_all = pd.DataFrame(records_cross)
df_cross_summary = df_cross_all.groupby(['Org', 'Network', 'Tema', 'Regime_Afetivo']).agg(
    Posts=('Total_Reactions', 'count'),
    Media_Reacoes=('Total_Reactions', 'mean'),
    Mediana_Reacoes=('Total_Reactions', 'median')
).reset_index()
df_cross_summary.to_excel(excel_writer, sheet_name='Cruzamento_Multidimensional', index=False)

# --- 4.8 Caracterização Sistemática dos 30 Outliers ---
print("[9/11] Caracterizando Sistematicamente os 30 Outliers...")
df_outliers = df_posts.sort_values(by='Total_Reactions', ascending=False).head(30).copy()

def get_labels_str(row, col_dict, prefix):
    active = [k.replace(prefix, '') for k in col_dict.keys() if row[k] == 1]
    return ', '.join(active) if active else 'Nenhum'

df_outliers['Temas_Mapeados'] = df_outliers.apply(lambda r: get_labels_str(r, THEMES_DICT, 'Tema_'), axis=1)
df_outliers['Afetos_Mapeados'] = df_outliers.apply(lambda r: get_labels_str(r, AFFECTS_DICT, 'Afeto_'), axis=1)

outliers_export = df_outliers[['Date', 'Org', 'Side', 'Network', 'Total_Reactions', 'Likes', 'Comments', 'PIR', 'Temas_Mapeados', 'Afetos_Mapeados', 'Message']]
outliers_export.to_excel(excel_writer, sheet_name='Top_30_Outliers_Detalhado', index=False)

excel_writer.close()
print(f"Planilha consolidada exportada com sucesso: {OUTPUT_EXCEL}")

# ==========================================
# 5. GERAÇÃO DE GRÁFICOS PNG (300 DPI)
# ==========================================
print("[10/11] Gerando figuras de alta resolução...")

# 1. Curva de Lorenz e Concentração
plt.figure(figsize=(7, 6))
arr_pal = np.sort(df_posts[df_posts['Side'] == 'Pró-Palestina']['Total_Reactions'].values)
arr_isr = np.sort(df_posts[df_posts['Side'] == 'Pró-Israel / Comunitário Judaico']['Total_Reactions'].values)
lorenz_pal = np.cumsum(arr_pal) / np.sum(arr_pal)
lorenz_isr = np.cumsum(arr_isr) / np.sum(arr_isr)

plt.plot(np.linspace(0, 1, len(lorenz_pal)), lorenz_pal, label=f'Pró-Palestina (Gini = {gini_coefficient(arr_pal):.2f})', color='darkred', lw=2)
plt.plot(np.linspace(0, 1, len(lorenz_isr)), lorenz_isr, label=f'Pró-Israel (Gini = {gini_coefficient(arr_isr):.2f})', color='navy', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Igualdade Perfeita')
plt.title('Curva de Lorenz: Desigualdade de Engajamento por Campo', fontsize=12)
plt.xlabel('Proporção Acumulada de Publicações')
plt.ylabel('Proporção Acumulada de Reações')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'curva_lorenz_campos.png'), dpi=300)
plt.close()

# 2. Comparação Shapley: Completo vs. Conteúdo Intrínseco
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.barplot(x='Relative_Importance_%', y='Feature', data=df_shapley_full, ax=axes[0], palette='viridis')
axes[0].set_title('Shapley Total (Com Emissores / Proxies)', fontsize=11)
axes[0].set_xlabel('Importância Relativa (%)')

sns.barplot(x='Relative_Importance_%', y='Feature', data=df_shapley_content, ax=axes[1], palette='magma')
axes[1].set_title('Shapley Intrínseco (Apenas Conteúdo e Afeto)', fontsize=11)
axes[1].set_xlabel('Importância Relativa (%)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'shapley_comparativo.png'), dpi=300)
plt.close()

# 3. Coeficientes do Modelo Multinível (Forest Plot)
plt.figure(figsize=(9, 6))
df_plot_mixed = df_mixed_res[~df_mixed_res['Variavel'].isin(['Intercept', 'Group Var'])].sort_values(by='Coeficiente_Fixo')
plt.errorbar(df_plot_mixed['Coeficiente_Fixo'], range(len(df_plot_mixed)), 
             xerr=[df_plot_mixed['Coeficiente_Fixo'] - df_plot_mixed['CI_Lower'], df_plot_mixed['CI_Upper'] - df_plot_mixed['Coeficiente_Fixo']],
             fmt='o', color='darkblue', ecolor='gray', elinewidth=2, capsize=4)
plt.yticks(range(len(df_plot_mixed)), df_plot_mixed['Variavel'])
plt.axvline(0, color='red', linestyle='--')
plt.title('Modelo Multinível: Coeficientes Fixos sobre Log(Reações) (IC 95%)', fontsize=12)
plt.xlabel('Impacto Estimado (Beta)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'modelo_multinivel_forest_plot.png'), dpi=300)
plt.close()

# 4. Outliers: Distribuição por Regime Afetivo
plt.figure(figsize=(9, 4.5))
outliers_affects_count = Counter([a.replace('Afeto_', '') for sub in df_outliers['Affects_List'] for a in sub if a != 'Neutro / Outro'])
df_out_aff = pd.DataFrame(outliers_affects_count.most_common(), columns=['Regime', 'Contagem'])
sns.barplot(x='Contagem', y='Regime', data=df_out_aff, palette='rocket')
plt.title('Regimes Afetivos Predominantes nos 30 Maiores Outliers', fontsize=12)
plt.xlabel('Número de Ocorrências no Top 30')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, 'outliers_regimes_afetivos.png'), dpi=300)
plt.close()

print(f"\n[11/11] Processo concluído com sucesso!")
print(f" Planilha com 10 abas gerada em: {OUTPUT_EXCEL}")
print(f" Gráficos analíticos salvos em: {OUTPUT_IMG_DIR}")