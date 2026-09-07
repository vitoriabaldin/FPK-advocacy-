import os
import glob
import re
import unicodedata
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# 1. DIRETÓRIOS E CONFIGURAÇÕES
# ==========================================
INPUT_DIR = r"C:\Users\viihb\Downloads\fpk\perfis"
OUTPUT_EXCEL = os.path.join(INPUT_DIR, "Cruzamento_Multidimensional_Detalhado.xlsx")
OUTPUT_IMG_DIR = os.path.join(INPUT_DIR, "png_cruzamentos")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ==========================================
# 2. DICIONÁRIOS E CLASSIFICADORES
# ==========================================
THEMES_DICT = {
    'Violência & Conflito': ['genocidio', 'bombardeio', 'mutilado', 'amputado', 'assassin', 'morto', 'morte', 'ataque', 'massacre', 'destruicao', 'crime de guerra', 'exterminio', 'invasao', 'ocupacao', 'militar', 'soldado', 'tiro', 'fogo', 'terrorismo', 'misseis', 'refens', 'sequestr', 'vitima', 'ferid', 'hospital', 'fome', 'hind rajab'],
    'Cultura & Celebridades': ['musica', 'cantor', 'cantora', 'rosalia', 'roger waters', 'pink floyd', 'filme', 'cinema', 'ator', 'atriz', 'gal gadot', 'show', 'clipe', 'arte', 'artista', 'cultura', 'livro', 'poesia', 'festival'],
    'Memória & História': ['holocausto', 'memoria', 'shoah', 'yom hashoa', 'auschwitz', 'sobrevivente', 'lembranca', 'historia', '7 de outubro', 'massacre de 7', 'nuncamais', 'memorial'],
    'Esporte': ['copa', 'futebol', 'atleta', 'selecao', 'jogo', 'jogadores', 'gol', 'camisa da selecao', 'ancelotti', 'campo'],
    'Comunidade & Identidade': ['comunidade', 'jovem', 'juventude', 'voluntariado', 'taglit', 'maavar', 'lideranca', 'sinagoga', 'shabat', 'rabino', 'solidariedade', 'familia', 'educacao', 'encontro', 'hillel', 'alia', 'aliya', 'fisesp'],
    'Política Institucional & Diplomacia': ['governo', 'governador', 'tarcisio', 'lula', 'biden', 'trump', 'netanyahu', 'itamaraty', 'onu', 'tribunal', 'haia', 'diplomacia', 'embaixad', 'senado', 'camara', 'deputad', 'ministro', 'reuniao de cupula', 'otan', 'acordo']
}

AFFECTS_DICT = {
    'Indignação': ['denuncia', 'massacre', 'genocidio', 'injustica', 'covardia', 'vergonha', 'crime', 'escandalo', 'criminoso', 'violacao', 'revolta'],
    'Luto': ['morte', 'mortes', 'vitima', 'vitimas', 'assassin', 'martir', 'martires', 'condolencias', 'dor', 'homenagem', 'homenagens', 'perda', 'perdas', 'luto', 'ferid', 'sepultamento'],
    'Urgência': ['agora', 'compartilhe', 'pressao', 'urgente', 'divulgue', 'assine', 'atencao', 'alerta', 'emergencia', 'imediato', 'ajude'],
    'Solidariedade': ['apoio', 'resistencia', 'mobilizacao', 'uniao', 'juntos', 'solidario', 'solidariedade', 'voluntari', 'ajuda humanitaria', 'doacao'],
    'Pertencimento': ['comunidade', 'tradicao', 'identidade', 'povo', 'raizes', 'lar', 'valores', 'familia', 'irmaos', 'sinagoga', 'shabat', 'nossa gente', 'nossos'],
    'Celebração': ['cultura', 'evento', 'eventos', 'conquista', 'conquistas', 'vitoria', 'parabens', 'celebra', 'celebracao', 'festival', 'festa', 'orgulho', 'musica', 'show', 'arte', 'alegria'],
    'Memória': ['holocausto', 'historia', 'lembranca', 'nunca esquecer', 'jamais esquecer', 'memoria', 'testemunho', 'passado', '7 de outubro', 'shoah', 'yom hashoa', 'auschwitz', 'sobrevivente']
}

def normalize_text(text):
    if pd.isna(text): return ""
    return "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn").lower()

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

def classify_labels(text, mapping):
    t = normalize_text(text)
    active = [k for k, v in mapping.items() if any(term in t for term in v)]
    return active if active else ['Neutro / Geral']

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
# 3. EXTRAÇÃO E CONSOLIDAÇÃO DOS DADOS
# ==========================================
print("[1/5] Lendo e higienizando planilhas brutas...")
all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")) + glob.glob(os.path.join(INPUT_DIR, "*.xml")))
raw_posts_list = []
for f in all_files:
    if 'analise_completa' in f or 'cruzamento' in f.lower(): continue
    try:
        xls = pd.ExcelFile(f)
        for s in xls.sheet_names:
            if 'overview' in s.lower() or 'top 5000' in s.lower():
                df_temp = extract_clean_sheet(f, s)
                if 'Message' in df_temp.columns and 'Profile' in df_temp.columns:
                    raw_posts_list.append(df_temp)
    except Exception as e:
        print(f"Aviso em {f}: {e}")

df_posts = pd.concat(raw_posts_list, ignore_index=True).drop_duplicates(subset=['Message', 'Date', 'Profile'])

# Tratamento Numérico
df_posts['Likes'] = pd.to_numeric(df_posts['Number of Likes'], errors='coerce').fillna(0)
df_posts['Comments'] = pd.to_numeric(df_posts['Number of comments'], errors='coerce').fillna(0)
df_posts['Total_Reactions'] = pd.to_numeric(df_posts['Reactions, Comments & Shares'], errors='coerce').fillna(0)
df_posts['PIR'] = pd.to_numeric(df_posts['Post interaction rate'], errors='coerce').fillna(0)
df_posts['Date'] = pd.to_datetime(df_posts['Date'], errors='coerce')
df_posts['Message'] = df_posts['Message'].astype(str).fillna('')
df_posts['Text_Length_Words'] = df_posts['Message'].apply(lambda x: len(x.split()))

df_posts['Org'] = df_posts['Profile'].apply(map_profile_group)
df_posts['Side'] = df_posts['Profile'].apply(categorize_side)
df_posts = df_posts.dropna(subset=['Org']).copy()

# Classificação por post
df_posts['Temas_Lista'] = df_posts['Message'].apply(lambda x: classify_labels(x, THEMES_DICT))
df_posts['Afetos_Lista'] = df_posts['Message'].apply(lambda x: classify_labels(x, AFFECTS_DICT))

df_posts['Temas_Classificados'] = df_posts['Temas_Lista'].apply(lambda l: "; ".join(l))
df_posts['Afetos_Classificados'] = df_posts['Afetos_Lista'].apply(lambda l: "; ".join(l))

print(f"Base consolidada com {len(df_posts)} publicações válidas.")

# ==========================================
# 4. PROCESSAMENTO DO CRUZAMENTO MULTIDIMENSIONAL
# ==========================================
print("[2/5] Processando tabela analítica de cruzamento...")
records_cross = []
for idx, row in df_posts.iterrows():
    for t in row['Temas_Lista']:
        for a in row['Afetos_Lista']:
            records_cross.append({
                'Org': row['Org'],
                'Network': row['Network'],
                'Tema': t,
                'Regime_Afetivo': a,
                'Total_Reactions': row['Total_Reactions']
            })

df_cross = pd.DataFrame(records_cross)

# Agrupamento com as colunas exatas solicitadas
agg_cross = df_cross.groupby(['Org', 'Network', 'Tema', 'Regime_Afetivo']).agg(
    Posts=('Total_Reactions', 'count'),
    Media_Reacoes=('Total_Reactions', 'mean'),
    Mediana_Reacoes=('Total_Reactions', 'median')
).reset_index().sort_values(by=['Org', 'Network', 'Media_Reacoes'], ascending=[True, True, False])

# ==========================================
# 5. EXPORTAÇÃO EXCEL (DUAS ABAS)
# ==========================================
print("[3/5] Exportando Excel com detalhes e cruzamentos...")
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    # Página 1: Cruzamento Multidimensional
    agg_cross.to_excel(writer, sheet_name='Cruzamento_Multidimensional', index=False)
    
    # Página 2: Detalhes de Classificação de Cada Publicação
    cols_detalhes = [
        'Date', 'Profile', 'Org', 'Side', 'Network', 
        'Total_Reactions', 'Likes', 'Comments', 'PIR', 
        'Temas_Classificados', 'Afetos_Classificados', 'Message'
    ]
    df_detalhes_export = df_posts[cols_detalhes].sort_values(by='Date', ascending=False)
    df_detalhes_export.to_excel(writer, sheet_name='Classificacao_Publicacoes', index=False)

print(f"Planilha exportada com sucesso em: {OUTPUT_EXCEL}")

# ==========================================
# 6. GERAÇÃO DE PNG PARA CADA CRUZAMENTO
# ==========================================
print("[4/5] Gerando heatmaps e gráficos em PNG para cada cruzamento...")

# Lista única de pares Organização x Plataforma
combos = agg_cross[['Org', 'Network']].drop_duplicates().values

for org, net in combos:
    sub = agg_cross[(agg_cross['Org'] == org) & (agg_cross['Network'] == net)]
    if len(sub) == 0: continue
    
    # Matriz Pivô para Heatmap (Tema x Regime Afetivo com Media de Reacoes)
    pivot_mean = sub.pivot(index='Tema', columns='Regime_Afetivo', values='Media_Reacoes')
    pivot_posts = sub.pivot(index='Tema', columns='Regime_Afetivo', values='Posts').fillna(0).astype(int)
    
    plt.figure(figsize=(11, 6.5))
    sns.heatmap(
        pivot_mean, 
        annot=pivot_posts, 
        fmt="d", 
        cmap="YlOrRd", 
        linewidths=0.6, 
        cbar_kws={'label': 'Média de Reações'},
        annot_kws={"size": 9}
    )
    plt.title(f"Cruzamento: {org} [{net}]\n(Cores: Média de Reações | Números: Qtd de Posts)", fontsize=12, pad=12)
    plt.xlabel("Regime Afetivo", fontsize=10, labelpad=8)
    plt.ylabel("Tema", fontsize=10, labelpad=8)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    safe_org = re.sub(r'[^\w\-]', '_', org)
    filename = f"cruzamento_{safe_org}_{net}.png"
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, filename), dpi=300)
    plt.close()

# Gráfico comparativo geral de Médias por Tema x Afeto
print("[5/5] Gerando gráfico geral consolidado...")
pivot_geral = df_cross.groupby(['Tema', 'Regime_Afetivo'])['Total_Reactions'].mean().unstack()
plt.figure(figsize=(12, 7))
sns.heatmap(pivot_geral, annot=True, fmt=".1f", cmap="magma", linewidths=0.5, cbar_kws={'label': 'Média de Reações Global'})
plt.title("Cruzamento Geral: Média de Reações por Tema e Regime Afetivo (Todo o Corpus)", fontsize=13, pad=14)
plt.xlabel("Regime Afetivo", fontsize=11)
plt.ylabel("Tema", fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, "cruzamento_geral_tema_afeto.png"), dpi=300)
plt.close()

print(f"\n Processo finalizado com sucesso!")
print(f" Arquivo Excel: {OUTPUT_EXCEL}")
print(f" Imagens salvas em: {OUTPUT_IMG_DIR}")