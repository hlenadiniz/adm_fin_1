#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
TRABALHO 1 - ADMINISTRAÇÃO FINANCEIRA (Sistemas de Informação)
Tema: RISCO E RETORNO

"O machine learning consegue prever RISCO melhor do que prever RETORNO?
 E isso ajuda a montar uma carteira melhor?"

O que o programa faz (passo a passo):
  1. CAPTURA DE DADOS  : baixa preços reais de ações da B3 + Ibovespa
                          (Yahoo Finance) e a taxa Selic (Banco Central).
  2. RISCO E RETORNO   : retorno, volatilidade, correlação, beta e CAPM.
  3. MACHINE LEARNING  : prevê a volatilidade (risco) dos próximos 21 dias
                          com 4 modelos e compara com um modelo ingênuo.
  4. CARTEIRA          : compara carteira de pesos iguais x carteira que
                          dá menos peso a quem tem risco previsto maior.
  5. RELATÓRIO         : gera um PDF com tabelas, gráficos e conclusões.

Como executar:
    pip install yfinance pandas numpy scikit-learn matplotlib reportlab arch
    python trabalho1_risco_retorno.py            (modo interativo)
    python trabalho1_risco_retorno.py --yes      (usa as opções padrão)
    python trabalho1_risco_retorno.py --demo     (dados SIMULADOS, p/ testes)
=====================================================================
"""

# ---------------------------------------------------------------------
# 0. IMPORTAÇÕES
# ---------------------------------------------------------------------
import argparse                 # ler opções da linha de comando
import json                     # guardar a Selic em cache
import os                       # caminhos de arquivos
import sys
import urllib.request           # chamar a API do Banco Central
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # gera imagens sem abrir janela
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. PARÂMETROS GERAIS (podem ser alterados)
# ---------------------------------------------------------------------
TICKERS_PADRAO = ["PETR4", "VALE3", "ITUB4", "WEGE3"]   # ações (sem o ".SA")
BENCHMARK = "^BVSP"          # Ibovespa = "carteira de mercado"
DATA_INICIO = "2015-01-01"   # início do histórico
H = 21                       # horizonte de previsão: 21 dias úteis (~1 mês)
DIAS_ANO = 252               # dias úteis por ano (para anualizar)
PCT_TREINO = 0.70            # 70% mais antigos = treino; 30% recentes = teste
SELIC_PADRAO = 15.0          # % a.a. usada SÓ se a API do BCB falhar
                             # (atualize se necessário!)
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQ_PRECOS = os.path.join(PASTA, "dados_precos.csv")    # cache dos preços
ARQ_SELIC = os.path.join(PASTA, "selic_cache.json")     # cache da Selic
PASTA_SAIDA = os.path.join(PASTA, "saida")              # onde vão os gráficos/relatório
os.makedirs(PASTA_SAIDA, exist_ok=True)


# ---------------------------------------------------------------------
# 2. CAPTURA DE DADOS
# ---------------------------------------------------------------------
def baixar_precos(tickers, inicio):
    """Baixa preços de fechamento AJUSTADOS (já descontam dividendos e
    desdobramentos) no Yahoo Finance. Se falhar, tenta o cache em CSV."""
    simbolos = [t + ".SA" for t in tickers] + [BENCHMARK]
    try:
        import yfinance as yf
    except ImportError:
        print("[ERRO] yfinance não instalado. Rode: pip install -U yfinance")
        yf = None

    bruto = None
    if yf is not None:
        # Tentativa 1: baixar todos os ativos de uma vez
        try:
            bruto = yf.download(simbolos, start=inicio, auto_adjust=True,
                                progress=False, threads=False)["Close"]
            bruto = bruto.dropna(how="all")
            if bruto.empty:
                raise RuntimeError("o Yahoo devolveu uma tabela vazia")
        except Exception as erro:
            print(f"[ERRO] download em lote falhou: {type(erro).__name__}: {erro}")
            bruto = None
        # Tentativa 2: baixar um ativo por vez (mais tolerante a limites do Yahoo)
        if bruto is None:
            import time
            partes = {}
            for s in simbolos:
                for tentativa in range(3):
                    try:
                        h = yf.Ticker(s).history(start=inicio, auto_adjust=True)["Close"]
                        if len(h) == 0:
                            raise RuntimeError("sem dados")
                        h.index = h.index.tz_localize(None).normalize()
                        partes[s] = h
                        break
                    except Exception as erro:
                        print(f"[ERRO] {s} (tentativa {tentativa+1}/3): "
                              f"{type(erro).__name__}: {erro}")
                        time.sleep(2)
            if partes:
                bruto = pd.DataFrame(partes)
        if bruto is not None and len(bruto) > 0:
            bruto.columns = [c.replace(".SA", "") for c in bruto.columns]
            bruto = bruto.rename(columns={BENCHMARK: "IBOV"})
            faltando = [t for t in tickers + ["IBOV"] if t not in bruto.columns]
            if faltando:
                print(f"[ERRO] Sem dados para: {faltando} (ticker digitado errado?)")
            else:
                bruto = bruto.dropna()
                if len(bruto) >= 500:
                    bruto.to_csv(ARQ_PRECOS)         # salva cache p/ uso offline
                    print(f"[dados] Preços baixados do Yahoo Finance ({len(bruto)} pregões).")
                    return bruto, False
                print(f"[ERRO] Histórico curto demais ({len(bruto)} pregões).")

    print("[aviso] Não foi possível baixar os preços.")
    print("        Dicas: 1) pip install -U yfinance   2) confira a internet/VPN"
          "   3) tente de novo em alguns minutos (limite do Yahoo)")
    if os.path.exists(ARQ_PRECOS):
        print("[dados] Usando o cache local dados_precos.csv.")
        cache = pd.read_csv(ARQ_PRECOS, index_col=0, parse_dates=True)
        return cache.dropna(), False
    return None, True


def baixar_selic():
    """Busca a meta Selic (% a.a.) na API do Banco Central (série SGS 432).
    Usada como taxa livre de risco no CAPM e no índice de Sharpe."""
    url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/"
           "ultimos/1?formato=json")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            valor = float(json.loads(resp.read())[0]["valor"])
        json.dump({"selic": valor}, open(ARQ_SELIC, "w"))
        print(f"[dados] Selic obtida do Banco Central: {valor:.2f}% a.a.")
        return valor / 100, "Banco Central (SGS 432)"
    except Exception:
        if os.path.exists(ARQ_SELIC):
            valor = json.load(open(ARQ_SELIC))["selic"]
            print(f"[dados] Selic do cache: {valor:.2f}% a.a.")
            return valor / 100, "cache local"
        print(f"[aviso] Selic indisponível; usando padrão {SELIC_PADRAO}% a.a.")
        return SELIC_PADRAO / 100, "valor padrão (API indisponível)"


def gerar_dados_simulados(tickers, inicio, semente=42):
    """APENAS PARA TESTE: cria preços artificiais com volatilidade que muda
    no tempo (processo GARCH). NÃO são dados reais."""
    rng = np.random.default_rng(semente)
    datas = pd.bdate_range(inicio, datetime.today())
    n = len(datas)

    def garch(omega, alpha, beta):
        r, var = np.zeros(n), omega / (1 - alpha - beta)
        for i in range(n):
            r[i] = np.sqrt(var) * rng.standard_normal()
            var = omega + alpha * r[i] ** 2 + beta * var
        return r

    mercado = garch(2e-6, 0.09, 0.89)
    precos = {}
    for k, t in enumerate(tickers):
        beta = 0.8 + 0.25 * k
        r = beta * mercado + garch(3e-6, 0.07, 0.90) + 0.0003
        precos[t] = 20 * np.exp(np.cumsum(r))
    precos["IBOV"] = 100000 * np.exp(np.cumsum(mercado + 0.0003))
    return pd.DataFrame(precos, index=datas)


# ---------------------------------------------------------------------
# 3. RISCO E RETORNO (conceitos da disciplina)
# ---------------------------------------------------------------------
def tabela_risco_retorno(precos, rf):
    """Para cada ativo calcula: retorno anual, risco (desvio-padrão anual),
    beta e retorno esperado pelo CAPM."""
    ret = np.log(precos / precos.shift(1)).dropna()      # retornos diários
    ret_anual = ret.mean() * DIAS_ANO                    # retorno médio anualizado
    vol_anual = ret.std() * np.sqrt(DIAS_ANO)            # risco anualizado
    # BETA = Cov(Ra, Rm) / Var(Rm): sensibilidade da ação ao mercado
    beta = ret.apply(lambda s: s.cov(ret["IBOV"]) / ret["IBOV"].var())
    # CAPM: E(Ra) = Rf + Beta * (E(Rm) - Rf)
    capm = rf + beta * (ret_anual["IBOV"] - rf)
    tab = pd.DataFrame({"Retorno a.a.": ret_anual, "Risco a.a.": vol_anual,
                        "Beta": beta, "Retorno CAPM": capm})
    tab["Sharpe"] = (tab["Retorno a.a."] - rf) / tab["Risco a.a."]
    return tab, ret


# ---------------------------------------------------------------------
# 4. MACHINE LEARNING: PREVER A VOLATILIDADE
# ---------------------------------------------------------------------
def montar_base(ret, ativos):
    """Cria a base de dados do ML (uma linha por ativo e por dia).
    VARIÁVEIS EXPLICATIVAS (só usam informação do passado):
      vol5, vol21, vol63 : volatilidade passada em 1 sem., 1 mês, 3 meses
      ret21              : retorno acumulado do último mês
      queda21            : risco apenas dos dias de queda (downside)
      vol_mercado        : volatilidade do Ibovespa no último mês
    ALVO: volatilidade REALIZADA nos próximos H dias."""
    raiz = np.sqrt(DIAS_ANO)
    vol_mercado = ret["IBOV"].rolling(21).std() * raiz
    blocos = []
    for a in ativos:
        r = ret[a]
        d = pd.DataFrame({
            "vol5": r.rolling(5).std() * raiz,
            "vol21": r.rolling(21).std() * raiz,
            "vol63": r.rolling(63).std() * raiz,
            "ret21": r.rolling(21).sum(),
            "queda21": r.clip(upper=0).rolling(21).std() * raiz,
            "vol_mercado": vol_mercado,
        })
        # rolling(H).std() em t usa t-H+1..t; shift(-H) traz o valor de t+H,
        # ou seja, a volatilidade dos dias t+1..t+H (o FUTURO de t).
        d["alvo"] = r.rolling(H).std().shift(-H) * raiz
        d["ativo"] = a
        d["ret_diario"] = r
        blocos.append(d)
    base = pd.concat(blocos)
    base.index.name = "data"
    return base


def previsao_garch(ret, ativos, data_corte):
    """Modelo GARCH(1,1): modelo clássico de finanças para volatilidade.
    Estima os parâmetros SÓ com dados de treino e depois atualiza a
    variância dia a dia. Devolve a volatilidade prevista p/ os próximos H dias."""
    try:
        from arch import arch_model
    except ImportError:
        print("[aviso] Biblioteca 'arch' ausente: GARCH será ignorado.")
        return None
    saida = {}
    for a in ativos:
        r = ret[a] * 100                                   # escala em %
        res = arch_model(r[:data_corte], vol="GARCH", p=1, q=1,
                         mean="Constant").fit(disp="off")
        om, al, be = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
        mu, pers = res.params["mu"], al + be
        var_inc = om / (1 - pers)                          # variância de longo prazo
        var, prev = r[:data_corte].var(), []
        for x in r:                                        # recursão da variância
            var = om + al * (x - mu) ** 2 + be * var       # variância de amanhã
            # média da variância nos próximos H dias (reversão à média)
            media = var_inc + (var - var_inc) * (1 - pers ** H) / (H * (1 - pers))
            prev.append(np.sqrt(max(media, 1e-12)) / 100 * np.sqrt(DIAS_ANO))
        saida[a] = pd.Series(prev, index=ret.index)
    return saida


def avaliar_modelos(base, ret, ativos):
    """Separa treino/teste NO TEMPO, treina os modelos e compara o erro."""
    datas = base.index.unique().sort_values()
    corte = datas[int(len(datas) * PCT_TREINO)]            # data de separação
    pos_corte = datas.get_loc(corte)
    # Treino termina H dias antes do corte: o alvo de uma linha de treino
    # olha H dias à frente e NÃO pode invadir o período de teste (vazamento).
    limite_treino = datas[pos_corte - H]
    dados = base.dropna(subset=["vol5", "vol21", "vol63", "ret21", "queda21",
                                "vol_mercado"])
    treino = dados[(dados.index <= limite_treino)].dropna(subset=["alvo"])
    teste = dados[(dados.index >= corte)].dropna(subset=["alvo"])

    cols = ["vol5", "vol21", "vol63", "ret21", "queda21", "vol_mercado"]
    cols_har = ["vol5", "vol21", "vol63"]
    # Trabalhamos com log da volatilidade (estabiliza a escala)
    def tf(df, c):
        X = df[c].copy()
        for k in X.columns:
            if k != "ret21":
                X[k] = np.log(X[k].clip(lower=1e-4))
        return X

    previsoes = pd.DataFrame(index=teste.index)
    previsoes["ativo"] = teste["ativo"]
    previsoes["real"] = teste["alvo"]
    # Modelo 0 (REFERÊNCIA): "o risco do próximo mês será igual ao do último mês"
    previsoes["Ingênuo (vol 21d)"] = teste["vol21"]
    # Modelo 1: regressão linear em volatilidades passadas (tipo HAR)
    lin = LinearRegression().fit(tf(treino, cols_har), np.log(treino["alvo"]))
    previsoes["Regressão linear"] = np.exp(lin.predict(tf(teste, cols_har)))
    # Modelo 2: Random Forest (aprende relações não-lineares)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                               max_features=0.7, random_state=1, n_jobs=-1)
    rf.fit(tf(treino, cols), np.log(treino["alvo"]))
    previsoes["Random Forest"] = np.exp(rf.predict(tf(teste, cols)))
    # Modelo 3: GARCH(1,1)
    garch = previsao_garch(ret, ativos, limite_treino)
    if garch is not None:
        previsoes["GARCH(1,1)"] = [garch[a].get(d, np.nan)
                                   for a, d in zip(teste["ativo"], teste.index)]

    previsoes = previsoes.dropna()
    nomes = [c for c in previsoes.columns if c not in ("ativo", "real")]
    linhas = []
    for n in nomes:
        linhas.append({"Modelo": n,
                       "RMSE": np.sqrt(mean_squared_error(previsoes["real"], previsoes[n])),
                       "MAE": mean_absolute_error(previsoes["real"], previsoes[n]),
                       "R²": r2_score(previsoes["real"], previsoes[n])})
    metricas = pd.DataFrame(linhas).set_index("Modelo")
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values()
    return previsoes, metricas, imp, corte, limite_treino


# ---------------------------------------------------------------------
# 5. CARTEIRAS: pesos iguais x pesos pelo risco previsto
# ---------------------------------------------------------------------
def simular_carteiras(precos, previsoes, ativos, corte, rf):
    """A cada H dias (rebalanceamento mensal) define os pesos:
      - Pesos iguais (1/N)
      - ML : peso proporcional a 1/(volatilidade prevista pelo Random Forest)
      - Ingênua: peso proporcional a 1/(volatilidade dos últimos 21 dias)
    Entre rebalanceamentos mantém a carteira (buy and hold)."""
    ret_simples = precos[ativos].pct_change()
    teste_datas = precos.index[precos.index >= corte]
    prev_rf = previsoes.pivot_table(index=previsoes.index, columns="ativo",
                                    values="Random Forest")
    prev_ing = previsoes.pivot_table(index=previsoes.index, columns="ativo",
                                     values="Ingênuo (vol 21d)")
    estrategias = {"Pesos iguais": None, "Inv. vol. prevista (ML)": prev_rf,
                   "Inv. vol. passada (ingênua)": prev_ing}
    curvas = {k: [] for k in estrategias}
    pesos_hist = []
    valor = {k: 1.0 for k in estrategias}
    i = 0
    while i + 1 < len(teste_datas):
        d0 = teste_datas[i]
        fim = min(i + H, len(teste_datas) - 1)
        janela = teste_datas[i + 1: fim + 1]             # dias do período
        if len(janela) == 0:
            break
        for nome, prev in estrategias.items():
            if prev is None or d0 not in prev.index:
                w = np.repeat(1 / len(ativos), len(ativos))
            else:
                inv = 1 / prev.loc[d0, ativos].values
                w = inv / inv.sum()
            if nome == "Inv. vol. prevista (ML)":
                pesos_hist.append(pd.Series(w, index=ativos, name=d0))
            cresc = (1 + ret_simples.loc[janela, ativos]).cumprod()
            caminho = valor[nome] * (cresc * w).sum(axis=1)
            curvas[nome].append(caminho)
            valor[nome] = caminho.iloc[-1]
        i = fim
    curvas = {k: pd.concat(v) for k, v in curvas.items()}
    cart = pd.DataFrame(curvas)
    cart["Ibovespa"] = precos["IBOV"].reindex(cart.index) / precos["IBOV"].reindex(cart.index).iloc[0]
    # Métricas de desempenho
    linhas = []
    for c in cart.columns:
        v = cart[c] / cart[c].iloc[0]
        r = v.pct_change().dropna()
        anos = len(r) / DIAS_ANO
        ret_a = v.iloc[-1] ** (1 / anos) - 1
        vol_a = r.std() * np.sqrt(DIAS_ANO)
        dd = (v / v.cummax() - 1).min()                  # drawdown máximo
        linhas.append({"Carteira": c, "Retorno a.a.": ret_a, "Risco a.a.": vol_a,
                       "Sharpe": (ret_a - rf) / vol_a, "Queda máx.": dd})
    return cart, pd.DataFrame(linhas).set_index("Carteira"), pd.DataFrame(pesos_hist)


# ---------------------------------------------------------------------
# 6. GRÁFICOS
# ---------------------------------------------------------------------
PALETA = ["#2a6f97", "#e07a1f", "#3a9d5d", "#9b4f96", "#777777", "#c8383a"]


def estilo(ax, titulo, xl="", yl=""):
    ax.set_title(titulo, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def gerar_graficos(precos, ret, tab, previsoes, metricas, imp, cart, ativos):
    arq = {}
    # 1) Preços normalizados
    fig, ax = plt.subplots(figsize=(8, 3.6))
    (precos / precos.iloc[0] * 100).plot(ax=ax, color=PALETA, lw=1.2)
    estilo(ax, "Evolução dos preços (base 100)")
    ax.legend(ncol=5, fontsize=8, frameon=False)
    arq["precos"] = os.path.join(PASTA_SAIDA, "g1_precos.png")
    fig.tight_layout(); fig.savefig(arq["precos"], dpi=150); plt.close(fig)

    # 2) Risco x Retorno (+ reta do CAPM)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, a in enumerate(tab.index):
        ax.scatter(tab.loc[a, "Risco a.a."], tab.loc[a, "Retorno a.a."],
                   s=70, color=PALETA[i % 6])
        ax.annotate(a, (tab.loc[a, "Risco a.a."], tab.loc[a, "Retorno a.a."]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    estilo(ax, "Risco x Retorno (anualizado)", "Risco (desvio-padrão)", "Retorno médio")
    arq["riscoret"] = os.path.join(PASTA_SAIDA, "g2_risco_retorno.png")
    fig.tight_layout(); fig.savefig(arq["riscoret"], dpi=150); plt.close(fig)

    # 3) Correlação
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = ret.corr()
    im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if corr.iloc[i, j] > .6 else "black")
    ax.set_title("Correlação dos retornos", fontsize=11, fontweight="bold", loc="left")
    arq["corr"] = os.path.join(PASTA_SAIDA, "g3_correlacao.png")
    fig.tight_layout(); fig.savefig(arq["corr"], dpi=150); plt.close(fig)

    # 4) Volatilidade prevista x realizada (1º ativo)
    a0 = ativos[0]
    p = previsoes[previsoes["ativo"] == a0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(p.index, p["real"], color="#222", lw=1.4, label="Realizada")
    ax.plot(p.index, p["Random Forest"], color=PALETA[1], lw=1.1, label="Random Forest")
    ax.plot(p.index, p["Ingênuo (vol 21d)"], color=PALETA[0], lw=1, alpha=.7, label="Ingênuo")
    estilo(ax, f"{a0}: volatilidade dos próximos 21 dias, prevista x realizada (período de teste)",
           "", "Volatilidade anualizada")
    ax.legend(fontsize=8, frameon=False)
    arq["prev"] = os.path.join(PASTA_SAIDA, "g4_previsao.png")
    fig.tight_layout(); fig.savefig(arq["prev"], dpi=150); plt.close(fig)

    # 5) Erro dos modelos
    fig, ax = plt.subplots(figsize=(6, 3.4))
    m = metricas["RMSE"].sort_values()
    ax.barh(m.index, m.values, color=[PALETA[1] if "Forest" in i else "#9aa5b1" for i in m.index])
    estilo(ax, "Erro de previsão do risco (RMSE, menor = melhor)")
    arq["rmse"] = os.path.join(PASTA_SAIDA, "g5_rmse.png")
    fig.tight_layout(); fig.savefig(arq["rmse"], dpi=150); plt.close(fig)

    # 6) Importância das variáveis
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(imp.index, imp.values, color=PALETA[0])
    estilo(ax, "O que o Random Forest mais usa para prever o risco")
    arq["imp"] = os.path.join(PASTA_SAIDA, "g6_importancia.png")
    fig.tight_layout(); fig.savefig(arq["imp"], dpi=150); plt.close(fig)

    # 7) Carteiras
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cart.plot(ax=ax, color=PALETA, lw=1.4)
    estilo(ax, "Valor de R$ 1 investido no início do teste")
    ax.legend(fontsize=8, frameon=False)
    arq["cart"] = os.path.join(PASTA_SAIDA, "g7_carteiras.png")
    fig.tight_layout(); fig.savefig(arq["cart"], dpi=150); plt.close(fig)
    return arq


# ---------------------------------------------------------------------
# 7. RELATÓRIO EM PDF
# ---------------------------------------------------------------------
def pct(x): return f"{x * 100:.1f}%"


def gerar_pdf(info, tab, metricas, cart_met, graf, ativos, conclusoes):
    arquivo = os.path.join(PASTA_SAIDA, "relatorio_risco_retorno.pdf")
    doc = SimpleDocTemplate(arquivo, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title="Relatório - Risco e Retorno")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10,
                        textColor=colors.HexColor("#1f4e79"))
    corpo = ParagraphStyle("c", parent=ss["BodyText"], fontSize=9.5, leading=13.5)
    peq = ParagraphStyle("p", parent=corpo, fontSize=8, textColor=colors.HexColor("#555"))
    alerta = ParagraphStyle("a", parent=corpo, backColor=colors.HexColor("#fde8e8"),
                            borderPadding=6, textColor=colors.HexColor("#8a1c1c"))

    def tabela(df, fmt):
        dados = [[df.index.name or ""] + list(df.columns)]
        for idx, lin in df.iterrows():
            dados.append([str(idx)] + [fmt.get(c, str)(lin[c]) for c in df.columns])
        t = Table(dados, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
            ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#c5ccd3")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        return t

    def img(k, w=16.5):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(graf[k]).getSize()
        return Image(graf[k], width=w*cm, height=w*cm*ih/iw)

    H_ = []
    H_ += [Paragraph("Risco e Retorno: dá para prever o risco com Machine Learning?", h1),
           Paragraph(f"Administração Financeira – Sistemas de Informação · gerado em "
                     f"{datetime.now():%d/%m/%Y %H:%M}", peq), Spacer(1, 6)]
    if info["simulado"]:
        H_.append(Paragraph("<b>ATENÇÃO: este relatório usa DADOS SIMULADOS (modo de teste), "
                            "não preços reais da B3.</b>", alerta))
        H_.append(Spacer(1, 6))
    H_.append(Paragraph(
        f"<b>Ativos:</b> {', '.join(ativos)} e Ibovespa · <b>Período:</b> {info['ini']} a {info['fim']} · "
        f"<b>Taxa livre de risco (Selic):</b> {info['selic']*100:.2f}% a.a. ({info['fonte_selic']}) · "
        f"<b>Horizonte de previsão:</b> {H} dias úteis · <b>Treino:</b> até {info['corte']} "
        f"· <b>Teste:</b> a partir de {info['corte']}.", corpo))

    H_.append(Paragraph("1. Risco e retorno dos ativos", h2))
    H_.append(Paragraph(
        "Retorno = média dos retornos diários anualizada. Risco = desvio-padrão anualizado. "
        "Beta = Cov(Ra, Rm)/Var(Rm). Retorno CAPM = Rf + β·(E(Rm) − Rf).", corpo))
    H_.append(Spacer(1, 4))
    tab_ = tab.copy(); tab_.index.name = "Ativo"
    H_.append(tabela(tab_, {"Retorno a.a.": pct, "Risco a.a.": pct, "Retorno CAPM": pct,
                            "Beta": lambda x: f"{x:.2f}", "Sharpe": lambda x: f"{x:.2f}"}))
    H_ += [Spacer(1, 6), img("precos"), img("riscoret", 10), img("corr", 8.5)]

    H_.append(PageBreak())
    H_.append(Paragraph("2. Previsão do risco com Machine Learning", h2))
    H_.append(Paragraph(
        "Pergunta: qual será a volatilidade de cada ação nos próximos 21 dias? Os modelos foram "
        "treinados só com o passado e avaliados no período de teste (mais recente), com uma "
        "folga de 21 dias entre treino e teste para não haver vazamento de informação do futuro. "
        "O modelo <i>Ingênuo</i> (risco futuro = risco do último mês) serve de referência: "
        "um modelo complexo só vale a pena se vencer a referência.", corpo))
    H_.append(Spacer(1, 4))
    m_ = metricas.copy(); m_.index.name = "Modelo"
    H_.append(tabela(m_, {"RMSE": lambda x: f"{x:.4f}", "MAE": lambda x: f"{x:.4f}",
                          "R²": lambda x: f"{x:.3f}"}))
    H_ += [Spacer(1, 6), img("rmse", 11), img("imp", 11), img("prev")]

    H_.append(PageBreak())
    H_.append(Paragraph("3. Usando a previsão para montar carteiras", h2))
    H_.append(Paragraph(
        "A cada 21 dias a carteira é rebalanceada. <i>Pesos iguais</i> dá 25% a cada ação. "
        "As carteiras de <i>volatilidade inversa</i> dão mais peso às ações de menor risco "
        "(previsto pelo Random Forest ou medido no último mês).", corpo))
    H_.append(Spacer(1, 4))
    c_ = cart_met.copy(); c_.index.name = "Carteira"
    H_.append(tabela(c_, {"Retorno a.a.": pct, "Risco a.a.": pct, "Queda máx.": pct,
                          "Sharpe": lambda x: f"{x:.2f}"}))
    H_ += [Spacer(1, 6), img("cart")]

    H_.append(Paragraph("4. Conclusões", h2))
    for c in conclusoes:
        H_.append(Paragraph("• " + c, corpo)); H_.append(Spacer(1, 3))
    H_.append(Spacer(1, 6))
    H_.append(Paragraph(
        "Limitações: desempenho passado não garante resultado futuro; o estudo ignora custos de "
        "transação e impostos; poucos ativos; o resultado depende do período de teste escolhido.", peq))
    doc.build(H_)
    return arquivo


def escrever_conclusoes(metricas, cart_met, tab):
    """Texto automático: as conclusões mudam conforme os resultados."""
    c = []
    melhor = metricas["RMSE"].idxmin()
    ing = metricas.loc["Ingênuo (vol 21d)", "RMSE"]
    ganho = (1 - metricas.loc[melhor, "RMSE"] / ing) * 100
    if melhor == "Ingênuo (vol 21d)":
        c.append("Nenhum modelo de ML superou a referência ingênua: a volatilidade passada recente já "
                 "é um ótimo previsor do risco futuro (volatilidade é persistente).")
    else:
        c.append(f"O melhor previsor de risco foi <b>{melhor}</b>, com erro (RMSE) {ganho:.1f}% menor "
                 f"que a referência ingênua. Risco é previsível porque a volatilidade se agrupa "
                 f"no tempo: períodos agitados tendem a ser seguidos por períodos agitados.")
    c.append("Já os <b>retornos</b> são muito mais difíceis de prever do que o risco — por isso o "
             "trabalho prevê volatilidade (risco), e não preço. Isso é coerente com a hipótese de "
             "mercados eficientes.")
    ml, ig, ing_ = (cart_met.loc["Inv. vol. prevista (ML)"], cart_met.loc["Pesos iguais"],
                    cart_met.loc["Inv. vol. passada (ingênua)"])
    c.append(f"Carteira guiada pelo ML: risco {pct(ml['Risco a.a.'])} e Sharpe {ml['Sharpe']:.2f}, contra "
             f"risco {pct(ig['Risco a.a.'])} e Sharpe {ig['Sharpe']:.2f} da carteira de pesos iguais "
             f"({'menor' if ml['Risco a.a.'] < ig['Risco a.a.'] else 'maior'} risco). "
             f"Queda máxima: {pct(ml['Queda máx.'])} (ML) x {pct(ig['Queda máx.'])} (pesos iguais).")
    c.append(f"Comparada à versão ingênua (Sharpe {ing_['Sharpe']:.2f}), a previsão por ML "
             f"{'melhorou' if ml['Sharpe'] > ing_['Sharpe'] else 'não melhorou'} o desempenho ajustado ao risco.")
    c.append("Relação com a teoria: diversificar reduz o risco da carteira abaixo do risco médio dos "
             "ativos, e saber estimar o risco de cada ativo permite alocar capital de forma mais "
             "eficiente — a base do modelo de Markowitz e do CAPM.")
    return c


# ---------------------------------------------------------------------
# 8. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="usa opções padrão (sem perguntas)")
    ap.add_argument("--demo", action="store_true", help="dados simulados (teste)")
    ap.add_argument("--tickers", nargs="+", help="ex.: --tickers PETR4 VALE3 ITUB4 WEGE3")
    args = ap.parse_args()

    # --- Entrada do usuário (interatividade) ---
    tickers = args.tickers or TICKERS_PADRAO
    if not args.yes and not args.tickers and not args.demo and sys.stdin.isatty():
        txt = input(f"Ações (sem .SA, separadas por espaço) [Enter = {' '.join(TICKERS_PADRAO)}]: ").strip()
        if txt:
            tickers = txt.upper().split()
    tickers = [t.upper().replace(".SA", "") for t in tickers]
    if len(tickers) < 2:
        sys.exit("Use pelo menos 2 ações.")

    # --- Captura de dados ---
    simulado = args.demo
    if not simulado:
        precos, falhou = baixar_precos(tickers, DATA_INICIO)
        if falhou:
            print("[aviso] Sem internet e sem cache: usando DADOS SIMULADOS.")
            simulado = True
    if simulado:
        precos = gerar_dados_simulados(tickers, DATA_INICIO)
        rf, fonte = SELIC_PADRAO / 100, "valor padrão (modo simulado)"
    else:
        rf, fonte = baixar_selic()

    ativos = [c for c in precos.columns if c != "IBOV"]
    ret_tab, ret = tabela_risco_retorno(precos, rf)
    print("\n=== RISCO E RETORNO ===\n", ret_tab.round(3))

    # --- Machine learning ---
    base = montar_base(ret, ativos)
    previsoes, metricas, imp, corte, _ = avaliar_modelos(base, ret, ativos)
    print("\n=== ERRO DOS MODELOS (menor = melhor) ===\n", metricas.round(4))

    # --- Carteiras ---
    cart, cart_met, _ = simular_carteiras(precos, previsoes, ativos, corte, rf)
    print("\n=== CARTEIRAS ===\n", cart_met.round(3))

    # --- Relatório ---
    graf = gerar_graficos(precos, ret, ret_tab, previsoes, metricas, imp, cart, ativos)
    concl = escrever_conclusoes(metricas, cart_met, ret_tab)
    info = {"simulado": simulado, "selic": rf, "fonte_selic": fonte,
            "ini": f"{precos.index[0]:%d/%m/%Y}", "fim": f"{precos.index[-1]:%d/%m/%Y}",
            "corte": f"{corte:%d/%m/%Y}"}
    pdf = gerar_pdf(info, ret_tab, metricas, cart_met, graf, ativos, concl)
    # Tabelas também em CSV (úteis para anexar ao trabalho)
    ret_tab.to_csv(os.path.join(PASTA_SAIDA, "tabela_risco_retorno.csv"))
    metricas.to_csv(os.path.join(PASTA_SAIDA, "metricas_modelos.csv"))
    cart_met.to_csv(os.path.join(PASTA_SAIDA, "metricas_carteiras.csv"))
    print(f"\nRelatório gerado: {pdf}")


if __name__ == "__main__":
    main()
