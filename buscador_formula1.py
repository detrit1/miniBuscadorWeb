import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    {
        "id": "A",
        "titulo": "Portal F1 Brasil",
        "texto": "portal de notícias sobre fórmula 1 corridas pilotos equipes campeonato temporada resultados",
        "links": ["B", "C", "D", "E"]
    },
    {
        "id": "B",
        "titulo": "Red Bull Racing",
        "texto": "equipe red bull racing campeonato construtores verstappen título vencedor temporada dominante",
        "links": ["C", "F"]
    },
    {
        "id": "C",
        "titulo": "Max Verstappen",
        "texto": "max verstappen piloto campeão mundial fórmula 1 red bull vitórias título holandês pole position",
        "links": ["B", "F"]
    },
    {
        "id": "D",
        "titulo": "Ferrari Scuderia",
        "texto": "scuderia ferrari equipe italiana fórmula 1 leclerc sainz campeonato construtores resultado corrida",
        "links": ["C", "E", "G"]
    },
    {
        "id": "E",
        "titulo": "Lewis Hamilton",
        "texto": "lewis hamilton piloto mercedes sete vezes campeão mundial fórmula 1 recordes vitórias pole position ferrari",
        "links": ["D", "F", "H"]
    },
    {
        "id": "F",
        "titulo": "Grande Prêmio de Mônaco",
        "texto": "grande prêmio de mônaco corrida fórmula 1 circuito histórico street circuit pit stop estratégia pole",
        "links": ["C", "G", "I"]
    },
    {
        "id": "G",
        "titulo": "Calendário F1 2024",
        "texto": "calendário temporada fórmula 1 2024 corridas grandes prêmios países circuitos datas programação",
        "links": ["A", "F", "H", "J"]
    },
    {
        "id": "H",
        "titulo": "Classificação Pilotos",
        "texto": "classificação campeonato mundial pilotos fórmula 1 pontuação verstappen hamilton leclerc posição temporada",
        "links": ["C", "E", "I"]
    },
    {
        "id": "I",
        "titulo": "Tecnologia dos Carros F1",
        "texto": "tecnologia carros fórmula 1 motor híbrido aerodinâmica downforce pit stop pneus estratégia corrida",
        "links": ["B", "D", "J"]
    },
    {
        "id": "J",
        "titulo": "Página Spam F1",
        "texto": "fórmula 1 grátis assistir ao vivo fórmula 1 fórmula 1 fórmula 1 clique agora promoção melhor site",
        "links": ["A", "C", "H"]
    }
]

df_paginas = pd.DataFrame(docs)

print("=" * 65)
print("SISTEMA DE BUSCA — FÓRMULA 1")
print("=" * 65)

print("\n1) PÁGINAS DA MINI-WEB")
print(df_paginas[["id", "titulo", "texto", "links"]].to_string(index=False))

consultas = [
    {
        "texto": "campeão mundial piloto fórmula 1",
        "relevantes": {"C", "E", "H"}
    },
    {
        "texto": "corridas circuito calendário fórmula 1",
        "relevantes": {"F", "G", "H"}
    }
]

print("\n2) CONSULTAS TESTADAS")
for i, c in enumerate(consultas, 1):
    print(f"   Consulta {i}: \"{c['texto']}\"")
    print(f"   Páginas relevantes: {sorted(c['relevantes'])}")

indice = {}
for doc in docs:
    for palavra in doc["texto"].lower().split():
        indice.setdefault(palavra, set()).add(doc["id"])

print("\n3) ÍNDICE INVERTIDO (palavras-chave selecionadas)")
for palavra in ["campeão", "piloto", "corrida", "equipe", "fórmula"]:
    print(f"   {palavra}: {sorted(indice.get(palavra, []))}")

def normalizar_coluna(df, coluna):
    maior = df[coluna].max()
    return df[coluna] / maior if maior != 0 else 0


def precision_k(tabela, k, relevantes, coluna_id="id"):
    topk = tabela.head(k)[coluna_id].tolist()
    acertos = sum(p in relevantes for p in topk)
    return topk, acertos, acertos / k


def executar_busca(consulta_texto, relevantes, label=""):
    print("\n" + "=" * 65)
    print(f"CONSULTA: \"{consulta_texto}\"")
    print(f"Relevantes definidos: {sorted(relevantes)}")
    print("=" * 65)

    textos = [doc["texto"] for doc in docs]
    vectorizer = TfidfVectorizer(lowercase=True)
    X = vectorizer.fit_transform(textos)
    q = vectorizer.transform([consulta_texto])
    scores_texto = cosine_similarity(q, X).flatten()

    ranking_texto = pd.DataFrame({
        "id": [doc["id"] for doc in docs],
        "titulo": [doc["titulo"] for doc in docs],
        "score_texto": scores_texto
    }).sort_values("score_texto", ascending=False).reset_index(drop=True)

    ranking_texto["posicao_texto"] = ranking_texto.index + 1
    ranking_texto["top3_texto"] = ranking_texto["posicao_texto"] <= 3
    ranking_texto["relevante"] = ranking_texto["id"].isin(relevantes)

    print(f"\n4) RANKING TEXTUAL")
    print(ranking_texto[["posicao_texto", "id", "titulo", "score_texto", "top3_texto", "relevante"]].round(6).to_string(index=False))

    G = nx.DiGraph()
    for doc in docs:
        G.add_node(doc["id"], titulo=doc["titulo"])
        for destino in doc["links"]:
            G.add_edge(doc["id"], destino)

    df_links = pd.DataFrame(list(G.edges()), columns=["origem", "destino"])
    print(f"\n5) LINKS DA MINI-WEB")
    print(df_links.to_string(index=False))

    # links recebidos
    df_links_rec = pd.DataFrame({
        "id": list(dict(G.in_degree()).keys()),
        "links_recebidos": list(dict(G.in_degree()).values())
    }).merge(df_paginas[["id", "titulo"]], on="id")
    df_links_rec = df_links_rec[["id", "titulo", "links_recebidos"]].sort_values("links_recebidos", ascending=False).reset_index(drop=True)
    print(f"\n6) LINKS RECEBIDOS POR PÁGINA")
    print(df_links_rec.to_string(index=False))

    pagerank = nx.pagerank(G, alpha=0.85)
    df_pr = pd.DataFrame({
        "id": list(pagerank.keys()),
        "pagerank": list(pagerank.values())
    }).merge(df_paginas[["id", "titulo"]], on="id")
    df_pr = df_pr[["id", "titulo", "pagerank"]].sort_values("pagerank", ascending=False).reset_index(drop=True)
    df_pr["posicao_pagerank"] = df_pr.index + 1

    print(f"\n7) RANKING POR PAGERANK")
    print(df_pr[["posicao_pagerank", "id", "titulo", "pagerank"]].round(6).to_string(index=False))

    hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
    df_hits = pd.DataFrame({
        "id": [doc["id"] for doc in docs],
        "titulo": [doc["titulo"] for doc in docs],
        "hub": [hubs[doc["id"]] for doc in docs],
        "authority": [authorities[doc["id"]] for doc in docs]
    })
    for col in ["hub", "authority"]:
        df_hits[col] = df_hits[col].apply(lambda x: 0 if abs(x) < 1e-7 else x)

    print(f"\n8) HITS — AUTHORITIES")
    aut = df_hits.sort_values("authority", ascending=False).reset_index(drop=True)
    aut["pos"] = aut.index + 1
    print(aut[["pos", "id", "titulo", "authority"]].round(6).to_string(index=False))

    print(f"\n9) HITS — HUBS")
    hub_rank = df_hits.sort_values("hub", ascending=False).reset_index(drop=True)
    hub_rank["pos"] = hub_rank.index + 1
    print(hub_rank[["pos", "id", "titulo", "hub"]].round(6).to_string(index=False))

    df_rank = ranking_texto[["id", "titulo", "score_texto"]].merge(
        df_pr[["id", "pagerank"]], on="id"
    ).merge(df_hits[["id", "hub", "authority"]], on="id")

    for col in ["score_texto", "pagerank", "authority"]:
        df_rank[col + "_norm"] = normalizar_coluna(df_rank, col)

    peso_texto     = 0.60
    peso_pagerank  = 0.25
    peso_authority = 0.15

    df_rank["score_final"] = (
        peso_texto     * df_rank["score_texto_norm"] +
        peso_pagerank  * df_rank["pagerank_norm"] +
        peso_authority * df_rank["authority_norm"]
    )

    ranking_antes = df_rank.sort_values("score_final", ascending=False).reset_index(drop=True)
    ranking_antes["posicao_antes"] = ranking_antes.index + 1
    ranking_antes["top3_antes"] = ranking_antes["posicao_antes"] <= 3
    ranking_antes["relevante"] = ranking_antes["id"].isin(relevantes)

    print(f"\n10) RANKING FINAL — ANTES DA MELHORIA")
    print("    obs: pesos -> texto=60%, pagerank=25%, authority=15%")
    cols = ["posicao_antes", "id", "titulo", "score_texto_norm",
            "pagerank_norm", "authority_norm", "score_final", "top3_antes", "relevante"]
    print(ranking_antes[cols].round(6).to_string(index=False))

    def detectar_spam(texto):
        palavras = texto.lower().split()
        from collections import Counter
        contagem = Counter(palavras)
        # considera spam se qualquer palavra aparecer 3 ou mais vezes
        return int(any(v >= 3 for v in contagem.values()))

    df_rank["spam_suspeito"] = df_rank["id"].apply(
        lambda pid: detectar_spam(next(d["texto"] for d in docs if d["id"] == pid))
    )

    media_authority = df_rank["authority_norm"].mean()
    df_rank["bonus_relevancia"] = (
        (df_rank["authority_norm"] > media_authority) & (~df_rank["spam_suspeito"].astype(bool))
    ).astype(float) * 0.15

    df_rank["score_final_melhorado"] = (
        df_rank["score_final"]
        * (1 - 0.70 * df_rank["spam_suspeito"])
        + df_rank["bonus_relevancia"]
    )

    ranking_depois = df_rank.sort_values("score_final_melhorado", ascending=False).reset_index(drop=True)
    ranking_depois["posicao_depois"] = ranking_depois.index + 1
    ranking_depois["top3_depois"] = ranking_depois["posicao_depois"] <= 3
    ranking_depois["relevante"] = ranking_depois["id"].isin(relevantes)

    print(f"\n11) RANKING FINAL — DEPOIS DA MELHORIA (anti-spam + bônus authority)")
    cols2 = ["posicao_depois", "id", "titulo", "score_final",
             "spam_suspeito", "bonus_relevancia", "score_final_melhorado",
             "top3_depois", "relevante"]
    print(ranking_depois[cols2].round(6).to_string(index=False))

    comparacao = ranking_antes[["id", "titulo", "posicao_antes"]].merge(
        ranking_depois[["id", "posicao_depois"]], on="id"
    ).merge(
        df_rank[["id", "score_final", "spam_suspeito", "bonus_relevancia", "score_final_melhorado"]], on="id"
    )
    comparacao["mudanca"] = comparacao["posicao_antes"] - comparacao["posicao_depois"]
    comparacao = comparacao.sort_values("posicao_depois").reset_index(drop=True)

    print(f"\n12) COMPARAÇÃO ANTES vs DEPOIS")
    print(comparacao.round(6).to_string(index=False))

    top3_txt, ac3_txt, p3_txt = precision_k(ranking_texto, 3, relevantes)
    top3_ant, ac3_ant, p3_ant = precision_k(ranking_antes, 3, relevantes)
    top3_dep, ac3_dep, p3_dep = precision_k(ranking_depois, 3, relevantes)

    linhas = [
        {"ranking": "somente texto",          "top3": ", ".join(top3_txt), "acertos": ac3_txt, "precision@3": p3_txt},
        {"ranking": "texto + pagerank/hits",   "top3": ", ".join(top3_ant), "acertos": ac3_ant, "precision@3": p3_ant},
        {"ranking": "texto + links + melhoria","top3": ", ".join(top3_dep), "acertos": ac3_dep, "precision@3": p3_dep},
    ]

    tabela_precision = pd.DataFrame(linhas)
    print(f"\n13) AVALIAÇÃO — PRECISION@3")
    print(tabela_precision.round(4).to_string(index=False))

    print(f"\n14) RESUMO RÁPIDO")
    print(f"    Top 3 só texto:             {top3_txt}  | Precision@3 = {p3_txt:.4f}")
    print(f"    Top 3 antes da melhoria:    {top3_ant}  | Precision@3 = {p3_ant:.4f}")
    print(f"    Top 3 depois da melhoria:   {top3_dep}  | Precision@3 = {p3_dep:.4f}")

    return G  # retorna o grafo para plotar ao final

for i, c in enumerate(consultas, 1):
    print(f"\n\n{'#'*65}")
    print(f"# CONSULTA {i}")
    print(f"{'#'*65}")
    G = executar_busca(c["texto"], c["relevantes"])

print("\n\n[gerando figura do grafo...]")

print("\nRESUMO DAS CONEXOES (pagina -> paginas referenciadas)")
for doc in docs:
    setas = " -> ".join(doc["links"]) if doc["links"] else "(nenhuma)"
    print(f"   {doc['id']} ({doc['titulo']}): {doc['id']} -> {setas}")

G_plot = nx.DiGraph()
for doc in docs:
    G_plot.add_node(doc["id"], titulo=doc["titulo"])
    for destino in doc["links"]:
        G_plot.add_edge(doc["id"], destino)

fig, ax = plt.subplots(figsize=(14, 10))

pos = nx.spring_layout(G_plot, seed=42, k=2.5)

cores = {
    "A": "#4fc3f7",  # portal
    "B": "#ef5350",  # red bull
    "C": "#ef5350",  # verstappen
    "D": "#ef5350",  # ferrari
    "E": "#ef5350",  # hamilton
    "F": "#66bb6a",  # gp monaco
    "G": "#66bb6a",  # calendário
    "H": "#66bb6a",  # classificação
    "I": "#ffa726",  # tecnologia
    "J": "#9e9e9e",  # spam
}

node_colors = [cores[n] for n in G_plot.nodes()]

nx.draw_networkx_nodes(G_plot, pos, node_color=node_colors, node_size=1800,
                       edgecolors="#333333", linewidths=1.5, ax=ax)

nx.draw_networkx_edges(G_plot, pos, edge_color="#333333", arrows=True,
                       arrowsize=30, arrowstyle="->",
                       width=2.2, min_target_margin=20,
                       connectionstyle="arc3,rad=0.12", ax=ax)

nx.draw_networkx_labels(G_plot, pos, font_size=11, font_weight="bold",
                        font_color="white", ax=ax)

titulos_curtos = {doc["id"]: doc["titulo"] for doc in docs}
label_offset = {k: (v[0], v[1] - 0.15) for k, v in pos.items()}
nx.draw_networkx_labels(G_plot, label_offset,
                        labels=titulos_curtos, font_size=7, ax=ax)

legenda = [
    mpatches.Patch(color="#4fc3f7", label="Portal (A)"),
    mpatches.Patch(color="#ef5350", label="Equipes / Pilotos (B,C,D,E)"),
    mpatches.Patch(color="#66bb6a", label="Corridas / Rankings (F,G,H)"),
    mpatches.Patch(color="#ffa726", label="Tecnologia (I)"),
    mpatches.Patch(color="#9e9e9e", label="Spam (J)"),
]
ax.legend(handles=legenda, loc="upper left", fontsize=9, title="Legenda",
          title_fontsize=10)

ax.set_title("Grafo Direcionado da Mini-Web — Fórmula 1\n(setas apontam para as páginas referenciadas)",
             fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("grafo_formula1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafo salvo em: grafo_formula1.png (na mesma pasta do script)")
