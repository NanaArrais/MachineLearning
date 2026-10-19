from flask import (
    Flask,
    render_template,
    request,
    Response,
    jsonify,
    send_file,
    redirect,
    url_for
)

import pandas as pd
import numpy as np

from io import BytesIO

import os
import shutil
import tempfile
import zipfile
import base64
from pathlib import Path
from PIL import Image

import json
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# APLICAÇÃO
# ============================================================

app = Flask(__name__)


# ============================================================
# ESTADO DA APLICAÇÃO
# ============================================================

estado = {

    # BASE ORIGINAL
    "df_original": None,
    "df_limpo": None,
    "df_normalizado": None,
    "nome_arquivo": None,

    # NORMALIZAÇÃO
    "parametros_normalizacao": {},

    # MODELO
    "entradas_selecionadas": [],
    "saida_selecionada": None,
    "schema_modelo": None,
    "mapa_saida": None,

    # TREINAMENTO
    "pesos": None,
    "bias": None,
    "dados_treino": None,
    "dados_teste": None,
    "historico_treinamento": [],

    # STATUS
    "variaveis_confirmadas": False,
    "modelo_treinado": False,
    "avaliacao_concluida": False,
    "predicao_concluida": False,

    # PREDIÇÃO
    "df_predicao_original": None,
    "df_predicao_normalizado": None,
    "resultado_predicao": None,
    "base_predicao_validada": False,
    "nome_arquivo_predicao": None,
}

# ==========================================================
# ESTADO - APRENDIZADO SUPERVISIONADO / CLASSIFICAÇÃO
# ==========================================================

estado_classificacao = {
    "df_original": None,
    "df_preparado": None,
    "nome_arquivo": None,

    "entradas": [],
    "saida": None,
    "mapa_saida": None,

    "X_treino": None,
    "X_teste": None,
    "y_treino": None,
    "y_teste": None,
    "scaler": None,
    "colunas_transformadas": [],
    "percentual_treino": 80,

    "algoritmo": None,
    "modelo": None,

    "avaliacao": None,
    "resultado_predicao": None,
    "resultado_predicao_df": None,
    "comparacao": None,

    "preparacao_concluida": False,
    "variaveis_confirmadas": False,
    "divisao_concluida": False,
    "algoritmo_selecionado": False,
    "modelo_treinado": False,
    "avaliacao_concluida": False,
    "predicao_concluida": False,
    "comparacao_concluida": False,

    "etapa_atual": 1,
}

# ============================================================
# AUXILIAR
# ============================================================

def converter_valor_python(valor):

    if isinstance(valor, np.generic):
        return valor.item()

    return valor


# ============================================================
# ANALISAR VARIÁVEIS
# ============================================================

def analisar_variaveis(df):

    variaveis = []
    entradas_sugeridas = []
    saidas_sugeridas = []

    for coluna in df.columns:

        serie = df[coluna]

        unicos = int(
            serie.nunique(
                dropna=True
            )
        )

        nome = (
            str(coluna)
            .lower()
            .strip()
        )

        eh_identificador = (
            nome == "id"
            or nome.startswith("id_")
            or nome.endswith("_id")
            or "codigo" in nome
            or "código" in nome
        )

        if eh_identificador:

            tipo = "Identificador"
            sugestao = "Ignorar"

        elif not pd.api.types.is_numeric_dtype(
            serie
        ):

            tipo = "Texto"
            sugestao = "Ignorar"

        elif unicos == 2:

            tipo = "Binária"
            sugestao = "Boa saída"

            saidas_sugeridas.append(
                coluna
            )

        else:

            tipo = "Numérica"
            sugestao = "Boa entrada"

            entradas_sugeridas.append(
                coluna
            )

        variaveis.append({
            "coluna": coluna,
            "tipo": tipo,
            "unicos": unicos,
            "sugestao": sugestao,
            "identificador": eh_identificador,
        })

    return (
        variaveis,
        entradas_sugeridas,
        saidas_sugeridas
    )


# ============================================================
# NORMALIZAR BASE
# ============================================================

def normalizar_base(df):

    df_normalizado = df.copy()

    parametros = {}

    for coluna in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[coluna]
        ):

            minimo = float(
                df[coluna].min()
            )

            maximo = float(
                df[coluna].max()
            )

            parametros[coluna] = {
                "minimo": minimo,
                "maximo": maximo,
            }

            if maximo != minimo:

                df_normalizado[coluna] = (
                    (
                        df[coluna]
                        -
                        minimo
                    )
                    /
                    (
                        maximo
                        -
                        minimo
                    )
                )

            else:

                df_normalizado[coluna] = 0.0

    return (
        df_normalizado,
        parametros
    )


# ============================================================
# RESETAR PREDIÇÃO
# ============================================================

def resetar_predicao():

    estado["df_predicao_original"] = None
    estado["df_predicao_normalizado"] = None
    estado["resultado_predicao"] = None
    estado["base_predicao_validada"] = False
    estado["nome_arquivo_predicao"] = None
    estado["predicao_concluida"] = False


# ============================================================
# RESETAR MODELO
# ============================================================

def resetar_modelo():

    estado["entradas_selecionadas"] = []
    estado["saida_selecionada"] = None
    estado["schema_modelo"] = None
    estado["mapa_saida"] = None

    estado["pesos"] = None
    estado["bias"] = None

    estado["dados_treino"] = None
    estado["dados_teste"] = None

    estado["historico_treinamento"] = []

    estado["variaveis_confirmadas"] = False
    estado["modelo_treinado"] = False
    estado["avaliacao_concluida"] = False
    estado["predicao_concluida"] = False

    resetar_predicao()


# ============================================================
# LER PLANILHA
# ============================================================

def ler_planilha(arquivo):

    nome = (
        arquivo.filename
        .lower()
    )

    if nome.endswith(".csv"):

        return pd.read_csv(
            arquivo
        )

    if nome.endswith(".xlsx"):

        return pd.read_excel(
            arquivo
        )

    raise ValueError(
        "Formato não suportado. Utilize CSV ou XLSX."
    )


# ============================================================
# CONTEXTO DA TELA
# ============================================================

def montar_contexto():

    contexto = {

        "dados": None,
        "colunas": None,

        "nome_arquivo":
            estado.get(
                "nome_arquivo"
            ),

        "erro": None,

        "total_registros": 0,
        "total_colunas": 0,
        "valores_ausentes": 0,
        "duplicados": 0,
        "registros_apos_limpeza": 0,

        "analise_colunas": [],

        "normalizacao": [],
        "dados_normalizados": None,

        "variaveis_disponiveis": [],
        "entradas_sugeridas": [],
        "saidas_sugeridas": [],

        "entradas_selecionadas":
            estado.get(
                "entradas_selecionadas",
                []
            ),

        "saida_selecionada":
            estado.get(
                "saida_selecionada"
            ),

        "variaveis_confirmadas":
            estado.get(
                "variaveis_confirmadas",
                False
            ),

        "modelo_treinado":
            estado.get(
                "modelo_treinado",
                False
            ),

        "avaliacao_concluida":
            estado.get(
                "avaliacao_concluida",
                False
            ),

        "predicao_concluida":
            estado.get(
                "predicao_concluida",
                False
            ),

        "mensagem_variaveis": None,
    }

    df = estado.get(
        "df_original"
    )

    df_limpo = estado.get(
        "df_limpo"
    )

    df_normalizado = estado.get(
        "df_normalizado"
    )

    if df is None:

        return contexto

    contexto["colunas"] = (
        df.columns.tolist()
    )

    contexto["total_registros"] = (
        len(df)
    )

    contexto["total_colunas"] = (
        len(df.columns)
    )

    contexto["valores_ausentes"] = int(
        df
        .isnull()
        .sum()
        .sum()
    )

    contexto["duplicados"] = int(
        df
        .duplicated()
        .sum()
    )

    # ========================================================
    # ANÁLISE DAS COLUNAS
    # ========================================================

    for coluna in df.columns:

        serie = df[coluna]

        if pd.api.types.is_numeric_dtype(
            serie
        ):

            tipo = "Numérica"

        else:

            tipo = "Texto"

        contexto[
            "analise_colunas"
        ].append({

            "coluna":
                coluna,

            "tipo":
                tipo,

            "tipo_python":
                str(
                    serie.dtype
                ),

            "nulos":
                int(
                    serie
                    .isnull()
                    .sum()
                ),

            "unicos":
                int(
                    serie
                    .nunique(
                        dropna=True
                    )
                ),
        })

    # ========================================================
    # BASE LIMPA
    # ========================================================

    if df_limpo is not None:

        contexto[
            "registros_apos_limpeza"
        ] = len(
            df_limpo
        )

        contexto[
            "dados"
        ] = (
            df_limpo
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        (
            variaveis,
            entradas_sugeridas,
            saidas_sugeridas

        ) = analisar_variaveis(
            df_limpo
        )

        contexto[
            "variaveis_disponiveis"
        ] = variaveis

        contexto[
            "entradas_sugeridas"
        ] = entradas_sugeridas

        contexto[
            "saidas_sugeridas"
        ] = saidas_sugeridas

    # ========================================================
    # NORMALIZAÇÃO
    # ========================================================

    if (
        df_limpo is not None
        and
        df_normalizado is not None
        and
        len(df_limpo) > 0
    ):

        parametros = estado.get(
            "parametros_normalizacao",
            {}
        )

        for coluna in df_limpo.columns:

            if coluna in parametros:

                contexto[
                    "normalizacao"
                ].append({

                    "coluna":
                        coluna,

                    "minimo":
                        round(
                            parametros[
                                coluna
                            ][
                                "minimo"
                            ],
                            4
                        ),

                    "maximo":
                        round(
                            parametros[
                                coluna
                            ][
                                "maximo"
                            ],
                            4
                        ),

                    "original":
                        round(
                            float(
                                df_limpo[
                                    coluna
                                ].iloc[0]
                            ),
                            4
                        ),

                    "normalizado":
                        round(
                            float(
                                df_normalizado[
                                    coluna
                                ].iloc[0]
                            ),
                            4
                        ),
                })

        contexto[
            "dados_normalizados"
        ] = (
            df_normalizado
            .head(10)
            .round(4)
            .to_dict(
                orient="records"
            )
        )

    return contexto


# ============================================================
# SSE
# ============================================================

def enviar_evento(
    tipo,
    mensagem
):

    return (
        "data: "
        +
        json.dumps(
            {
                "tipo": tipo,
                "mensagem": mensagem,
            },
            ensure_ascii=False
        )
        +
        "\n\n"
    )


# ============================================================
# ROTA PRINCIPAL
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    contexto = montar_contexto()

    if request.method == "POST":

        acao = request.form.get(
            "acao"
        )

        # ====================================================
        # IMPORTAR BASE ORIGINAL
        # ====================================================

        if acao == "importar":

            arquivo = request.files.get(
                "arquivo"
            )

            if (
                not arquivo
                or
                not arquivo.filename
            ):

                contexto[
                    "erro"
                ] = (
                    "Selecione um arquivo."
                )

                return render_template(
                    "index.html",
                    **contexto
                )

            try:

                df = ler_planilha(
                    arquivo
                )

                if df.empty:

                    raise ValueError(
                        "A planilha está vazia."
                    )

                estado[
                    "df_original"
                ] = df.copy()

                estado[
                    "nome_arquivo"
                ] = arquivo.filename

                # =============================================
                # LIMPEZA
                # =============================================

                df_limpo = (
                    df
                    .drop_duplicates()
                    .dropna()
                    .reset_index(
                        drop=True
                    )
                )

                if df_limpo.empty:

                    raise ValueError(
                        (
                            "Após a limpeza, "
                            "não restaram registros válidos."
                        )
                    )

                estado[
                    "df_limpo"
                ] = df_limpo.copy()

                # =============================================
                # NORMALIZAÇÃO
                # =============================================

                (
                    df_normalizado,
                    parametros

                ) = normalizar_base(
                    df_limpo
                )

                estado[
                    "df_normalizado"
                ] = df_normalizado

                estado[
                    "parametros_normalizacao"
                ] = parametros

                resetar_modelo()

                contexto = (
                    montar_contexto()
                )

            except Exception as erro:

                contexto = (
                    montar_contexto()
                )

                contexto[
                    "erro"
                ] = (
                    "Erro ao processar o arquivo: "
                    +
                    str(
                        erro
                    )
                )

        # ====================================================
        # CONFIRMAR VARIÁVEIS
        # ====================================================

        elif acao == "confirmar_variaveis":

            df_limpo = estado.get(
                "df_limpo"
            )

            df_normalizado = estado.get(
                "df_normalizado"
            )

            if (
                df_limpo is None
                or
                df_normalizado is None
            ):

                contexto[
                    "erro"
                ] = (
                    (
                        "A base não está disponível. "
                        "Importe novamente."
                    )
                )

            else:

                entradas = (
                    request.form.getlist(
                        "entradas"
                    )
                )

                saida = (
                    request.form.get(
                        "saida"
                    )
                )

                contexto = (
                    montar_contexto()
                )

                contexto[
                    "entradas_selecionadas"
                ] = entradas

                contexto[
                    "saida_selecionada"
                ] = saida

                if not entradas:

                    contexto[
                        "mensagem_variaveis"
                    ] = (
                        (
                            "Selecione pelo menos "
                            "uma variável de entrada."
                        )
                    )

                elif not saida:

                    contexto[
                        "mensagem_variaveis"
                    ] = (
                        "Selecione uma variável de saída."
                    )

                elif saida in entradas:

                    contexto[
                        "mensagem_variaveis"
                    ] = (
                        (
                            "A variável de saída "
                            "não pode ser usada como entrada."
                        )
                    )

                elif saida not in df_limpo.columns:

                    contexto[
                        "mensagem_variaveis"
                    ] = (
                        "A variável de saída não existe."
                    )

                elif any(
                    coluna not in df_limpo.columns
                    for coluna in entradas
                ):

                    contexto[
                        "mensagem_variaveis"
                    ] = (
                        (
                            "Uma ou mais entradas "
                            "não existem na base."
                        )
                    )

                elif any(
                    not pd.api.types.is_numeric_dtype(
                        df_limpo[
                            coluna
                        ]
                    )
                    for coluna
                    in entradas
                ):

                    contexto[
                        "mensagem_variaveis"
                    ] = (
                        (
                            "As variáveis de entrada "
                            "precisam ser numéricas."
                        )
                    )

                else:

                    valores_saida = (
                        df_limpo[
                            saida
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    if len(
                        valores_saida
                    ) != 2:

                        contexto[
                            "mensagem_variaveis"
                        ] = (
                            (
                                "A saída precisa possuir "
                                "exatamente duas classes."
                            )
                        )

                    else:

                        try:

                            valores_ordenados = sorted(
                                valores_saida
                            )

                        except TypeError:

                            valores_ordenados = (
                                valores_saida
                            )

                        mapa_saida = {
                            valores_ordenados[0]:
                                0,

                            valores_ordenados[1]:
                                1,
                        }

                        df_modelo = (
                            df_normalizado.copy()
                        )

                        # A saída é convertida para 0/1
                        # e não deve ser usada normalizada.

                        df_modelo[
                            saida
                        ] = (
                            df_limpo[
                                saida
                            ]
                            .map(
                                mapa_saida
                            )
                            .astype(
                                int
                            )
                        )

                        estado[
                            "df_normalizado"
                        ] = df_modelo

                        estado[
                            "entradas_selecionadas"
                        ] = list(
                            entradas
                        )

                        estado[
                            "saida_selecionada"
                        ] = saida

                        estado[
                            "mapa_saida"
                        ] = mapa_saida

                        estado[
                            "schema_modelo"
                        ] = {
                            "entradas":
                                list(
                                    entradas
                                ),

                            "saida":
                                saida,
                        }

                        estado[
                            "variaveis_confirmadas"
                        ] = True

                        # Reseta treinamento anterior

                        estado[
                            "pesos"
                        ] = None

                        estado[
                            "bias"
                        ] = None

                        estado[
                            "dados_treino"
                        ] = None

                        estado[
                            "dados_teste"
                        ] = None

                        estado[
                            "historico_treinamento"
                        ] = []

                        estado[
                            "modelo_treinado"
                        ] = False

                        estado[
                            "avaliacao_concluida"
                        ] = False

                        estado[
                            "predicao_concluida"
                        ] = False

                        resetar_predicao()

                        contexto = (
                            montar_contexto()
                        )

    return render_template(
        "index.html",
        **contexto
    )


# ============================================================
# PASSO 6 - TREINAMENTO
# ============================================================

@app.route(
    "/treinar-stream"
)
def treinar_stream():

    try:

        taxa = float(
            request.args.get(
                "taxa",
                0.1
            )
        )

        max_epocas = int(
            request.args.get(
                "epocas",
                100
            )
        )

        percentual_treino = float(
            request.args.get(
                "percentual",
                80
            )
        )

    except (
        TypeError,
        ValueError
    ):

        taxa = 0.1
        max_epocas = 100
        percentual_treino = 80

    df = estado.get(
        "df_normalizado"
    )

    entradas = estado.get(
        "entradas_selecionadas",
        []
    )

    saida = estado.get(
        "saida_selecionada"
    )

    def gerar():

        if df is None:

            yield enviar_evento(
                "erro",
                "Base normalizada não encontrada."
            )

            return

        if not entradas or not saida:

            yield enviar_evento(
                "erro",
                "Confirme as variáveis antes do treinamento."
            )

            return

        if taxa <= 0:

            yield enviar_evento(
                "erro",
                "A taxa de aprendizado deve ser maior que zero."
            )

            return

        if max_epocas < 1:

            yield enviar_evento(
                "erro",
                "Informe pelo menos uma época."
            )

            return

        if (
            percentual_treino <= 0
            or
            percentual_treino >= 100
        ):

            yield enviar_evento(
                "erro",
                (
                    "O percentual de treinamento "
                    "deve estar entre 1 e 99."
                )
            )

            return

        colunas_modelo = (
            entradas
            +
            [saida]
        )

        dados = (
            df[
                colunas_modelo
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(
            dados
        ) < 2:

            yield enviar_evento(
                "erro",
                (
                    "São necessários pelo menos "
                    "dois registros."
                )
            )

            return

        # ====================================================
        # EMBARALHAR
        # ====================================================

        dados = (
            dados
            .sample(
                frac=1,
                random_state=42
            )
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # TREINO / TESTE
        # ====================================================

        total_registros = len(
            dados
        )

        quantidade_teste = int(
            round(
                total_registros
                *
                (
                    1
                    -
                    percentual_treino / 100
                )
            )
        )

        # Para fins didáticos,
        # tenta manter pelo menos 2 registros no teste.

        if total_registros >= 4:

            quantidade_teste = max(
                2,
                quantidade_teste
            )

        else:

            quantidade_teste = max(
                1,
                quantidade_teste
            )

        quantidade_teste = min(
            quantidade_teste,
            total_registros - 1
        )

        quantidade_treino = (
            total_registros
            -
            quantidade_teste
        )

        df_treino = (
            dados
            .iloc[
                :quantidade_treino
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        df_teste = (
            dados
            .iloc[
                quantidade_treino:
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        estado[
            "dados_treino"
        ] = df_treino.copy()

        estado[
            "dados_teste"
        ] = df_teste.copy()

        # ====================================================
        # PERCEPTRON
        # ====================================================

        pesos = np.zeros(
            len(
                entradas
            ),
            dtype=float
        )

        bias = 0.0

        historico = []

        # ====================================================
        # LOG INICIAL
        # ====================================================

        yield enviar_evento(
            "inicio",
            "Iniciando treinamento..."
        )

        yield enviar_evento(
            "info",
            f"Registros totais: {len(dados)}"
        )

        yield enviar_evento(
            "info",
            (
                "Registros de treinamento: "
                f"{len(df_treino)}"
            )
        )

        yield enviar_evento(
            "info",
            (
                "Registros reservados para avaliação: "
                f"{len(df_teste)}"
            )
        )

        yield enviar_evento(
            "info",
            (
                "Entradas: "
                +
                ", ".join(
                    entradas
                )
            )
        )

        yield enviar_evento(
            "info",
            f"Saída: {saida}"
        )

        yield enviar_evento(
            "info",
            (
                "Taxa de aprendizado: "
                f"{taxa}"
            )
        )

        yield enviar_evento(
            "info",
            (
                "Máximo de épocas: "
                f"{max_epocas}"
            )
        )

        yield enviar_evento(
            "separador",
            "========================================"
        )

        # ====================================================
        # ÉPOCAS
        # ====================================================

        for epoca in range(
            1,
            max_epocas + 1
        ):

            erros_epoca = 0

            yield enviar_evento(
                "epoca",
                f"ÉPOCA {epoca}"
            )

            yield enviar_evento(
                "separador",
                "----------------------------------------"
            )

            for indice, linha in (
                df_treino
                .iterrows()
            ):

                x = (
                    linha[
                        entradas
                    ]
                    .astype(
                        float
                    )
                    .to_numpy()
                )

                esperado = int(
                    linha[
                        saida
                    ]
                )

                soma = float(
                    np.dot(
                        x,
                        pesos
                    )
                    +
                    bias
                )

                previsto = (
                    1
                    if soma >= 0
                    else 0
                )

                erro = (
                    esperado
                    -
                    previsto
                )

                yield enviar_evento(
                    "registro",
                    (
                        "Registro "
                        f"{indice + 1}"
                    )
                )

                entradas_texto = (
                    " | ".join(
                        f"{nome}={x[i]:.4f}"
                        for i, nome
                        in enumerate(
                            entradas
                        )
                    )
                )

                yield enviar_evento(
                    "calculo",
                    (
                        "Entradas: "
                        +
                        entradas_texto
                    )
                )

                parcelas = (
                    " + ".join(
                        (
                            f"({x[i]:.4f}"
                            f" × "
                            f"{pesos[i]:.4f})"
                        )
                        for i
                        in range(
                            len(
                                entradas
                            )
                        )
                    )
                )

                yield enviar_evento(
                    "calculo",
                    (
                        f"Soma = {parcelas} "
                        f"+ Bias({bias:.4f})"
                    )
                )

                yield enviar_evento(
                    "calculo",
                    f"Soma = {soma:.4f}"
                )

                yield enviar_evento(
                    "resultado",
                    (
                        "Saída prevista = "
                        f"{previsto}"
                    )
                )

                yield enviar_evento(
                    "resultado",
                    (
                        "Saída esperada = "
                        f"{esperado}"
                    )
                )

                yield enviar_evento(
                    "resultado",
                    (
                        "Erro = "
                        f"{esperado}"
                        " - "
                        f"{previsto}"
                        " = "
                        f"{erro}"
                    )
                )

                if erro != 0:

                    erros_epoca += 1

                    pesos_anteriores = (
                        pesos.copy()
                    )

                    bias_anterior = (
                        bias
                    )

                    pesos = (
                        pesos
                        +
                        taxa
                        *
                        erro
                        *
                        x
                    )

                    bias = (
                        bias
                        +
                        taxa
                        *
                        erro
                    )

                    yield enviar_evento(
                        "ajuste",
                        (
                            "A rede errou. "
                            "Ajustando pesos e bias..."
                        )
                    )

                    for i, nome in enumerate(
                        entradas
                    ):

                        yield enviar_evento(
                            "ajuste",
                            (
                                f"W{i + 1} ({nome}) = "
                                f"{pesos_anteriores[i]:.4f}"
                                " + "
                                f"({taxa} × "
                                f"{erro} × "
                                f"{x[i]:.4f})"
                                " = "
                                f"{pesos[i]:.4f}"
                            )
                        )

                    yield enviar_evento(
                        "ajuste",
                        (
                            "Bias = "
                            f"{bias_anterior:.4f}"
                            " + "
                            f"({taxa} × "
                            f"{erro})"
                            " = "
                            f"{bias:.4f}"
                        )
                    )

                else:

                    yield enviar_evento(
                        "acerto",
                        (
                            "Classificação correta. "
                            "Pesos e bias não foram alterados."
                        )
                    )

                yield enviar_evento(
                    "separador",
                    "----------------------------------------"
                )

                time.sleep(
                    0.03
                )

            # =================================================
            # FIM DA ÉPOCA
            # =================================================

            historico.append({

                "epoca":
                    epoca,

                "erros":
                    erros_epoca,

                "pesos":
                    [
                        float(p)
                        for p
                        in pesos
                    ],

                "bias":
                    float(
                        bias
                    ),
            })

            yield enviar_evento(
                "resumo",
                (
                    f"Resumo da época {epoca}: "
                    f"{erros_epoca} erro(s)"
                )
            )

            yield enviar_evento(
                "resumo",
                (
                    "Pesos atuais: "
                    +
                    str(
                        [
                            round(
                                float(p),
                                4
                            )
                            for p
                            in pesos
                        ]
                    )
                )
            )

            yield enviar_evento(
                "resumo",
                (
                    "Bias atual: "
                    f"{bias:.4f}"
                )
            )

            yield enviar_evento(
                "separador",
                "========================================"
            )

            if erros_epoca == 0:

                yield enviar_evento(
                    "sucesso",
                    (
                        "Nenhum erro encontrado. "
                        "Treinamento encerrado antecipadamente."
                    )
                )

                break

        # ====================================================
        # SALVAR MODELO
        # ====================================================

        estado[
            "pesos"
        ] = [
            float(p)
            for p
            in pesos
        ]

        estado[
            "bias"
        ] = float(
            bias
        )

        estado[
            "historico_treinamento"
        ] = historico

        estado[
            "modelo_treinado"
        ] = True

        estado[
            "avaliacao_concluida"
        ] = False

        estado[
            "predicao_concluida"
        ] = False

        resetar_predicao()

        resultado = {

            "epocas_executadas":
                len(
                    historico
                ),

            "erros_finais":
                historico[-1][
                    "erros"
                ],

            "pesos":
                [
                    round(
                        float(p),
                        4
                    )
                    for p
                    in pesos
                ],

            "bias":
                round(
                    float(
                        bias
                    ),
                    4
                ),

            "entradas":
                entradas,
        }

        yield enviar_evento(
            "fim",
            resultado
        )

    return Response(

        gerar(),

        mimetype=
            "text/event-stream",

        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",
        }
    )


# ============================================================
# PASSO 7 - AVALIAÇÃO
# ============================================================

@app.route(
    "/avaliar-modelo"
)
def avaliar_modelo():

    pesos = estado.get(
        "pesos"
    )

    bias = estado.get(
        "bias"
    )

    dados_teste = estado.get(
        "dados_teste"
    )

    entradas = estado.get(
        "entradas_selecionadas",
        []
    )

    saida = estado.get(
        "saida_selecionada"
    )

    if (
        pesos is None
        or
        bias is None
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                "O modelo ainda não foi treinado."
        })

    if (
        dados_teste is None
        or
        len(
            dados_teste
        ) == 0
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                "Não há dados reservados para avaliação."
        })

    pesos = np.array(
        pesos,
        dtype=float
    )

    vp = 0
    vn = 0
    fp = 0
    fn = 0

    resultados = []

    for numero, (_, linha) in enumerate(
        dados_teste.iterrows(),
        start=1
    ):

        x = (
            linha[
                entradas
            ]
            .astype(
                float
            )
            .to_numpy()
        )

        esperado = int(
            linha[
                saida
            ]
        )

        soma = float(
            np.dot(
                x,
                pesos
            )
            +
            bias
        )

        previsto = (
            1
            if soma >= 0
            else 0
        )

        if (
            esperado == 1
            and
            previsto == 1
        ):

            vp += 1

        elif (
            esperado == 0
            and
            previsto == 0
        ):

            vn += 1

        elif (
            esperado == 0
            and
            previsto == 1
        ):

            fp += 1

        else:

            fn += 1

        resultados.append({

            "registro":
                numero,

            "soma":
                round(
                    soma,
                    4
                ),

            "esperado":
                esperado,

            "previsto":
                previsto,

            "resultado":
                (
                    "Acerto"
                    if esperado == previsto
                    else "Erro"
                ),
        })

    total = len(
        resultados
    )

    acertos = (
        vp
        +
        vn
    )

    erros = (
        fp
        +
        fn
    )

    acuracia = (
        acertos
        /
        total
        if total > 0
        else 0
    )

    precisao = (
        vp
        /
        (
            vp
            +
            fp
        )
        if (
            vp
            +
            fp
        ) > 0
        else 0
    )

    recall = (
        vp
        /
        (
            vp
            +
            fn
        )
        if (
            vp
            +
            fn
        ) > 0
        else 0
    )

    f1 = (
        2
        *
        precisao
        *
        recall
        /
        (
            precisao
            +
            recall
        )
        if (
            precisao
            +
            recall
        ) > 0
        else 0
    )

    estado[
        "avaliacao_concluida"
    ] = True

    return jsonify({

        "sucesso":
            True,

        "total":
            total,

        "acertos":
            acertos,

        "erros":
            erros,

        "acuracia":
            round(
                acuracia
                *
                100,
                2
            ),

        "precisao":
            round(
                precisao
                *
                100,
                2
            ),

        "recall":
            round(
                recall
                *
                100,
                2
            ),

        "f1":
            round(
                f1
                *
                100,
                2
            ),

        "matriz": {
            "vp": vp,
            "vn": vn,
            "fp": fp,
            "fn": fn,
        },

        "resultados":
            resultados,
    })


# ============================================================
# PASSO 8 - VALIDAR BASE
# ============================================================

@app.route(
    "/validar-base-predicao",
    methods=[
        "POST"
    ]
)
def validar_base_predicao():

    if not estado.get(
        "modelo_treinado"
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                (
                    "O modelo precisa ser treinado "
                    "antes da predição."
                )
        })

    schema = estado.get(
        "schema_modelo"
    )

    if not schema:

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                "O esquema do modelo não foi encontrado."
        })

    arquivo = request.files.get(
        "arquivo_predicao"
    )

    if (
        not arquivo
        or
        not arquivo.filename
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                "Selecione uma planilha."
        })

    try:

        df = ler_planilha(
            arquivo
        )

    except Exception as erro:

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                str(
                    erro
                )
        })

    if df.empty:

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                "A planilha está vazia."
        })

    entradas = schema[
        "entradas"
    ]

    saida = schema[
        "saida"
    ]

    colunas_recebidas = (
        df.columns.tolist()
    )

    # ========================================================
    # COLUNAS AUSENTES
    # ========================================================

    colunas_ausentes = [
        coluna
        for coluna
        in entradas
        if coluna
        not in colunas_recebidas
    ]

    if colunas_ausentes:

        resetar_predicao()

        return jsonify({

            "sucesso":
                False,

            "mensagem":
                (
                    "A base é incompatível com o modelo. "
                    "Existem colunas obrigatórias ausentes."
                ),

            "colunas_ausentes":
                colunas_ausentes,

            "colunas_esperadas":
                entradas,

            "colunas_encontradas":
                colunas_recebidas,
        })

    # ========================================================
    # AUSENTES
    # ========================================================

    ausentes = {}

    for coluna in entradas:

        quantidade = int(
            df[
                coluna
            ]
            .isnull()
            .sum()
        )

        if quantidade > 0:

            ausentes[
                coluna
            ] = quantidade

    if ausentes:

        resetar_predicao()

        return jsonify({

            "sucesso":
                False,

            "mensagem":
                (
                    "A base possui valores ausentes "
                    "nas variáveis utilizadas pelo modelo."
                ),

            "valores_ausentes":
                ausentes,
        })

    # ========================================================
    # TIPOS NUMÉRICOS
    # ========================================================

    erros_tipo = {}

    df_convertido = (
        df.copy()
    )

    for coluna in entradas:

        serie_original = (
            df[
                coluna
            ]
        )

        serie_numerica = pd.to_numeric(
            serie_original,
            errors="coerce"
        )

        invalidos = (
            serie_numerica.isna()
            &
            serie_original.notna()
        )

        if invalidos.any():

            linhas = (
                np.where(
                    invalidos
                )[0]
                +
                2
            ).tolist()

            erros_tipo[
                coluna
            ] = linhas

        else:

            df_convertido[
                coluna
            ] = serie_numerica

    if erros_tipo:

        resetar_predicao()

        return jsonify({

            "sucesso":
                False,

            "mensagem":
                (
                    "Foram encontrados valores não numéricos "
                    "em variáveis utilizadas pelo modelo."
                ),

            "erros_tipo":
                erros_tipo,
        })

    # ========================================================
    # COLUNAS EXTRAS
    # ========================================================

    colunas_extras = [
        coluna
        for coluna
        in colunas_recebidas
        if coluna
        not in entradas
        and
        coluna != saida
    ]

    saida_presente = (
        saida
        in colunas_recebidas
    )

    # ========================================================
    # NORMALIZAÇÃO DA NOVA BASE
    # ========================================================

    parametros = estado.get(
        "parametros_normalizacao",
        {}
    )

    df_normalizado = (
        df_convertido.copy()
    )

    variaveis_sem_parametros = []

    valores_fora_intervalo = {}

    for coluna in entradas:

        if coluna not in parametros:

            variaveis_sem_parametros.append(
                coluna
            )

            continue

        minimo = parametros[
            coluna
        ][
            "minimo"
        ]

        maximo = parametros[
            coluna
        ][
            "maximo"
        ]

        fora = (
            (
                df_convertido[
                    coluna
                ]
                <
                minimo
            )
            |
            (
                df_convertido[
                    coluna
                ]
                >
                maximo
            )
        )

        quantidade_fora = int(
            fora.sum()
        )

        if quantidade_fora > 0:

            valores_fora_intervalo[
                coluna
            ] = quantidade_fora

        if maximo != minimo:

            df_normalizado[
                coluna
            ] = (
                (
                    df_convertido[
                        coluna
                    ]
                    -
                    minimo
                )
                /
                (
                    maximo
                    -
                    minimo
                )
            )

        else:

            df_normalizado[
                coluna
            ] = 0.0

    if variaveis_sem_parametros:

        resetar_predicao()

        return jsonify({

            "sucesso":
                False,

            "mensagem":
                (
                    "Não foi possível recuperar "
                    "os parâmetros de normalização do modelo."
                ),

            "variaveis":
                variaveis_sem_parametros,
        })

    # ========================================================
    # SALVA BASE VALIDADA
    # ========================================================

    estado[
        "df_predicao_original"
    ] = df_convertido.copy()

    estado[
        "df_predicao_normalizado"
    ] = df_normalizado.copy()

    estado[
        "base_predicao_validada"
    ] = True

    estado[
        "nome_arquivo_predicao"
    ] = arquivo.filename

    estado[
        "resultado_predicao"
    ] = None

    estado[
        "predicao_concluida"
    ] = False

    previa = (
        df_convertido
        .head(10)
        .where(
            pd.notnull(
                df_convertido
                .head(10)
            ),
            None
        )
        .to_dict(
            orient="records"
        )
    )

    return jsonify({

        "sucesso":
            True,

        "mensagem":
            "Base compatível com o modelo.",

        "arquivo":
            arquivo.filename,

        "total_registros":
            len(
                df_convertido
            ),

        "colunas_esperadas":
            entradas,

        "colunas_extras":
            colunas_extras,

        "saida_presente":
            saida_presente,

        "nome_saida":
            saida,

        "valores_fora_intervalo":
            valores_fora_intervalo,

        "colunas":
            colunas_recebidas,

        "previa":
            previa,
    })


# ============================================================
# PASSO 8 - EXECUTAR PREDIÇÃO
# ============================================================

@app.route(
    "/executar-predicao",
    methods=[
        "POST"
    ]
)
def executar_predicao():

    if not estado.get(
        "modelo_treinado"
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                "O modelo ainda não foi treinado."
        })

    if not estado.get(
        "base_predicao_validada"
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                (
                    "A nova base precisa ser validada "
                    "antes da predição."
                )
        })

    df_original = estado.get(
        "df_predicao_original"
    )

    df_normalizado = estado.get(
        "df_predicao_normalizado"
    )

    pesos = estado.get(
        "pesos"
    )

    bias = estado.get(
        "bias"
    )

    entradas = estado.get(
        "entradas_selecionadas",
        []
    )

    saida = estado.get(
        "saida_selecionada"
    )

    mapa_saida = estado.get(
        "mapa_saida",
        {}
    )

    if (
        df_original is None
        or
        df_normalizado is None
        or
        pesos is None
        or
        bias is None
    ):

        return jsonify({
            "sucesso":
                False,

            "mensagem":
                (
                    "Não foi possível recuperar "
                    "os dados necessários para a predição."
                )
        })

    pesos = np.array(
        pesos,
        dtype=float
    )

    mapa_inverso = {
        valor_binario:
            converter_valor_python(
                classe_original
            )

        for classe_original,
        valor_binario
        in mapa_saida.items()
    }

    previsoes = []
    somas = []
    classificacoes = []

    for _, linha in (
        df_normalizado
        .iterrows()
    ):

        x = (
            linha[
                entradas
            ]
            .astype(
                float
            )
            .to_numpy()
        )

        soma = float(
            np.dot(
                x,
                pesos
            )
            +
            bias
        )

        previsto = (
            1
            if soma >= 0
            else 0
        )

        classificacao = (
            mapa_inverso.get(
                previsto,
                previsto
            )
        )

        somas.append(
            round(
                soma,
                6
            )
        )

        previsoes.append(
            previsto
        )

        classificacoes.append(
            classificacao
        )

    resultado = (
        df_original.copy()
    )

    resultado[
        "Soma_Rede"
    ] = somas

    resultado[
        "Previsao"
    ] = previsoes

    resultado[
        "Classificacao_Prevista"
    ] = classificacoes

    estado[
        "resultado_predicao"
    ] = resultado.copy()

    estado[
        "predicao_concluida"
    ] = True

    quantidade_zero = int(
        (
            resultado[
                "Previsao"
            ]
            ==
            0
        ).sum()
    )

    quantidade_um = int(
        (
            resultado[
                "Previsao"
            ]
            ==
            1
        ).sum()
    )

    previa_resultado = (
        resultado
        .head(20)
        .where(
            pd.notnull(
                resultado
                .head(20)
            ),
            None
        )
        .to_dict(
            orient="records"
        )
    )

    return jsonify({

        "sucesso":
            True,

        "mensagem":
            "Predição realizada com sucesso.",

        "total":
            len(
                resultado
            ),

        "quantidade_zero":
            quantidade_zero,

        "quantidade_um":
            quantidade_um,

        "saida_prevista":
            saida,

        "colunas":
            resultado.columns.tolist(),

        "resultados":
            previa_resultado,
    })


# ============================================================
# DOWNLOAD DAS PREDIÇÕES
# ============================================================

@app.route(
    "/baixar-predicoes"
)
def baixar_predicoes():

    resultado = estado.get(
        "resultado_predicao"
    )

    if resultado is None:

        return (
            "Nenhuma predição disponível.",
            400
        )

    arquivo_excel = (
        BytesIO()
    )

    with pd.ExcelWriter(
        arquivo_excel,
        engine="openpyxl"
    ) as writer:

        resultado.to_excel(
            writer,
            index=False,
            sheet_name="Predicoes"
        )

    arquivo_excel.seek(
        0
    )

    return send_file(

        arquivo_excel,

        as_attachment=True,

        download_name=
            "resultado_predicoes.xlsx",

        mimetype=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

# ============================================================
# HOME
# ============================================================

@app.route("/home")
def home():

    return render_template(
        "home.html"
    )






# ==========================================================
# AUXILIARES - CLASSIFICAÇÃO SUPERVISIONADA
# ==========================================================

def coluna_identificadora(nome):
    nome = str(nome).strip().casefold()
    return (nome == "id" or nome.startswith("id_") or nome.endswith("_id")
            or nome.startswith("codigo") or nome.startswith("código"))


def listar_variaveis_classificacao(dados):
    """Permite alvos categóricos (incluindo texto) e entradas numéricas/textuais."""
    opcoes = []
    for coluna in dados.columns:
        serie = dados[coluna]
        unicos = int(serie.nunique(dropna=True))
        identificador = coluna_identificadora(coluna)
        numerica = pd.api.types.is_numeric_dtype(serie)
        if identificador:
            tipo, sugestao = "Identificador", "Não utilizar (identificador)."
        elif numerica:
            tipo, sugestao = "Numérica", "Pode ser entrada (X)."
        else:
            tipo, sugestao = "Categórica", "Pode ser entrada (X); conversão automática."
        # Não oferecer variáveis contínuas de alta cardinalidade como alvo.
        # 2 a 30 classes são suficientes para este laboratório.
        saida_valida = not identificador and 2 <= unicos <= 30
        entrada_valida = not identificador and unicos >= 1 and (numerica or unicos <= 50)
        if saida_valida:
            if 'prioridade' in str(coluna).casefold() or 'classe' in str(coluna).casefold():
                sugestao = "Saída recomendada (Y): classificação por categorias."
            elif unicos <= 5:
                sugestao = "Pode ser entrada (X) ou saída (Y)."
        if not entrada_valida and not saida_valida and not identificador:
            sugestao = "Muitas categorias; não recomendado como entrada ou saída."
        opcoes.append({
            "coluna": coluna, "tipo": tipo, "unicos": unicos,
            "sugestao": sugestao, "pode_entrada": entrada_valida,
            "pode_saida": saida_valida,
        })
    return opcoes


def criar_preprocessador_classificacao(dados, entradas):
    numericas = [c for c in entradas if pd.api.types.is_numeric_dtype(dados[c])]
    categoricas = [c for c in entradas if c not in numericas]
    passos = []
    if numericas:
        passos.append(("num", Pipeline([
            ("ausentes", SimpleImputer(strategy="median")),
            ("escala", MinMaxScaler()),
        ]), numericas))
    if categoricas:
        # Compatibilidade com scikit-learn anterior a 1.2.
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        passos.append(("cat", Pipeline([
            ("ausentes", SimpleImputer(strategy="most_frequent")),
            ("categorias", encoder),
        ]), categoricas))
    return ColumnTransformer(passos, sparse_threshold=0)


def calcular_avaliacao_classificacao(modelo, X_teste, y_teste, mapa_saida):
    previstos = modelo.predict(X_teste)
    codigos = list(range(len(mapa_saida)))
    nomes = [str(next(chave for chave, codigo in mapa_saida.items() if codigo == i))
             for i in codigos]
    multiclasse = len(codigos) > 2
    op = {"average": "macro" if multiclasse else "binary", "zero_division": 0}
    matriz = confusion_matrix(y_teste, previstos, labels=codigos)
    dados_matriz = matriz.tolist()
    resultados = [
        {"registro": i + 1, "esperado": nomes[int(esperado)],
         "previsto": nomes[int(previsto)],
         "resultado": "Acerto" if int(esperado) == int(previsto) else "Erro"}
        for i, (esperado, previsto) in enumerate(zip(y_teste, previstos))
    ]
    supports = [int(x) for x in matriz.sum(axis=1)]
    avaliacao = {
        "acuracia": round(float(accuracy_score(y_teste, previstos)) * 100, 2),
        "precisao": round(float(precision_score(y_teste, previstos, **op)) * 100, 2),
        "recall": round(float(recall_score(y_teste, previstos, **op)) * 100, 2),
        "f1": round(float(f1_score(y_teste, previstos, **op)) * 100, 2),
        "tipo_media": "macro" if multiclasse else "binária (classe 1)",
        "classes": nomes, "codigos": codigos, "multiclasse": multiclasse,
        "matriz_completa": dados_matriz,
        "support": [{"classe": nome, "quantidade": valor}
                    for nome, valor in zip(nomes, supports)],
        "total": len(resultados), "acertos": sum(i["resultado"] == "Acerto" for i in resultados),
        "resultados": resultados,
    }
    if not multiclasse:
        vn, fp, fn, vp = matriz.ravel()
        avaliacao.update(vn=int(vn), fp=int(fp), fn=int(fn), vp=int(vp))
    return avaliacao


def resetar_classificacao_apos_importacao():
    estado_classificacao["df_preparado"] = None
    estado_classificacao["entradas"] = []
    estado_classificacao["saida"] = None
    estado_classificacao["mapa_saida"] = None
    estado_classificacao["X_treino"] = None
    estado_classificacao["X_teste"] = None
    estado_classificacao["y_treino"] = None
    estado_classificacao["y_teste"] = None
    estado_classificacao["scaler"] = None
    estado_classificacao["algoritmo"] = None
    estado_classificacao["modelo"] = None
    estado_classificacao["avaliacao"] = None
    estado_classificacao["resultado_predicao"] = None
    estado_classificacao["resultado_predicao_df"] = None
    estado_classificacao["comparacao"] = None

    estado_classificacao["preparacao_concluida"] = False
    estado_classificacao["variaveis_confirmadas"] = False
    estado_classificacao["divisao_concluida"] = False
    estado_classificacao["algoritmo_selecionado"] = False
    estado_classificacao["modelo_treinado"] = False
    estado_classificacao["avaliacao_concluida"] = False
    estado_classificacao["predicao_concluida"] = False
    estado_classificacao["comparacao_concluida"] = False
    estado_classificacao["etapa_atual"] = 2


def resetar_classificacao_apos_preparacao():
    estado_classificacao["entradas"] = []
    estado_classificacao["saida"] = None
    estado_classificacao["mapa_saida"] = None
    estado_classificacao["X_treino"] = None
    estado_classificacao["X_teste"] = None
    estado_classificacao["y_treino"] = None
    estado_classificacao["y_teste"] = None
    estado_classificacao["scaler"] = None
    estado_classificacao["algoritmo"] = None
    estado_classificacao["modelo"] = None
    estado_classificacao["avaliacao"] = None
    estado_classificacao["resultado_predicao"] = None
    estado_classificacao["resultado_predicao_df"] = None
    estado_classificacao["comparacao"] = None

    estado_classificacao["variaveis_confirmadas"] = False
    estado_classificacao["divisao_concluida"] = False
    estado_classificacao["algoritmo_selecionado"] = False
    estado_classificacao["modelo_treinado"] = False
    estado_classificacao["avaliacao_concluida"] = False
    estado_classificacao["predicao_concluida"] = False
    estado_classificacao["comparacao_concluida"] = False


def criar_modelo_classificacao(nome_algoritmo, quantidade_treino):
    if nome_algoritmo == "logistica":
        return LogisticRegression(max_iter=1000, random_state=42)

    if nome_algoritmo == "knn":
        k = min(5, max(1, quantidade_treino))
        return KNeighborsClassifier(n_neighbors=k)

    if nome_algoritmo == "arvore":
        return DecisionTreeClassifier(random_state=42, max_depth=5)

    if nome_algoritmo == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    raise ValueError("Algoritmo de classificação inválido.")


def nome_amigavel_algoritmo(nome_algoritmo):
    nomes = {
        "logistica": "Regressão Logística",
        "knn": "K-Nearest Neighbors (KNN)",
        "arvore": "Árvore de Decisão",
        "random_forest": "Random Forest",
    }
    return nomes.get(nome_algoritmo, nome_algoritmo)


def montar_contexto_classificacao(mensagem=None, erro=None):
    df_original = estado_classificacao.get("df_original")
    df_preparado = estado_classificacao.get("df_preparado")

    contexto = {
        "mensagem": mensagem,
        "erro": erro,

        "nome_arquivo": estado_classificacao.get("nome_arquivo"),
        "etapa_atual": estado_classificacao.get("etapa_atual", 1),

        "quantidade_registros": 0,
        "quantidade_colunas": 0,
        "colunas": [],
        "previa": [],
        "analise_colunas": [],
        "estatisticas": [],
        "total_ausentes": 0,
        "total_duplicados": 0,

        "preparacao_concluida": estado_classificacao.get("preparacao_concluida", False),
        "registros_preparados": 0,
        "ausentes_preparados": 0,
        "duplicados_preparados": 0,
        "previa_preparada": [],

        "variaveis_confirmadas": estado_classificacao.get("variaveis_confirmadas", False),
        "entradas": estado_classificacao.get("entradas", []),
        "saida": estado_classificacao.get("saida"),
        "colunas_numericas": [],
        "variaveis_disponiveis": [],

        "divisao_concluida": estado_classificacao.get("divisao_concluida", False),
        "percentual_treino": estado_classificacao.get("percentual_treino", 80),
        "qtd_treino": 0,
        "qtd_teste": 0,
        "previa_treino": [],
        "previa_teste": [],

        "algoritmo_selecionado": estado_classificacao.get("algoritmo_selecionado", False),
        "algoritmo": estado_classificacao.get("algoritmo"),
        "algoritmo_nome": nome_amigavel_algoritmo(
            estado_classificacao.get("algoritmo")
        ) if estado_classificacao.get("algoritmo") else None,

        "modelo_treinado": estado_classificacao.get("modelo_treinado", False),

        "avaliacao_concluida": estado_classificacao.get("avaliacao_concluida", False),
        "avaliacao": estado_classificacao.get("avaliacao"),

        "predicao_concluida": estado_classificacao.get("predicao_concluida", False),
        "resultado_predicao": estado_classificacao.get("resultado_predicao"),

        "comparacao_concluida": estado_classificacao.get("comparacao_concluida", False),
        "comparacao": estado_classificacao.get("comparacao"),
    }

    if df_original is not None:
        contexto["quantidade_registros"] = len(df_original)
        contexto["quantidade_colunas"] = len(df_original.columns)
        contexto["colunas"] = df_original.columns.tolist()

        contexto["previa"] = (
            df_original
            .head(10)
            .where(pd.notnull(df_original.head(10)), None)
            .to_dict(orient="records")
        )

        contexto["total_ausentes"] = int(
            df_original.isnull().sum().sum()
        )

        contexto["total_duplicados"] = int(
            df_original.duplicated().sum()
        )

        for coluna in df_original.columns:
            serie = df_original[coluna]
            unicos = int(serie.nunique(dropna=True))

            if pd.api.types.is_numeric_dtype(serie):
                tipo = "Binária" if unicos == 2 else "Numérica"
            else:
                tipo = "Texto"

            contexto["analise_colunas"].append({
                "coluna": coluna,
                "tipo": tipo,
                "tipo_python": str(serie.dtype),
                "ausentes": int(serie.isnull().sum()),
                "unicos": unicos,
            })

            if pd.api.types.is_numeric_dtype(serie):
                serie_valida = serie.dropna()

                if len(serie_valida) > 0:
                    desvio = serie_valida.std()
                    if pd.isna(desvio):
                        desvio = 0

                    contexto["estatisticas"].append({
                        "coluna": coluna,
                        "minimo": round(float(serie_valida.min()), 2),
                        "maximo": round(float(serie_valida.max()), 2),
                        "media": round(float(serie_valida.mean()), 2),
                        "mediana": round(float(serie_valida.median()), 2),
                        "desvio_padrao": round(float(desvio), 2),
                    })

    if df_preparado is not None:
        contexto["registros_preparados"] = len(df_preparado)
        contexto["ausentes_preparados"] = int(
            df_preparado.isnull().sum().sum()
        )
        contexto["duplicados_preparados"] = int(
            df_preparado.duplicated().sum()
        )

        contexto["previa_preparada"] = (
            df_preparado
            .head(10)
            .where(pd.notnull(df_preparado.head(10)), None)
            .to_dict(orient="records")
        )

        contexto["colunas_numericas"] = [
            coluna
            for coluna in df_preparado.columns
            if pd.api.types.is_numeric_dtype(df_preparado[coluna])
        ]
        contexto["variaveis_disponiveis"] = listar_variaveis_classificacao(df_preparado)

    X_treino = estado_classificacao.get("X_treino")
    X_teste = estado_classificacao.get("X_teste")
    y_treino = estado_classificacao.get("y_treino")
    y_teste = estado_classificacao.get("y_teste")

    if X_treino is not None and y_treino is not None:
        contexto["qtd_treino"] = len(X_treino)
        treino_preview = X_treino.copy()
        treino_preview["Saida"] = y_treino.reset_index(drop=True)
        contexto["previa_treino"] = (
            treino_preview.head(8).round(4).to_dict(orient="records")
        )

    if X_teste is not None and y_teste is not None:
        contexto["qtd_teste"] = len(X_teste)
        teste_preview = X_teste.copy()
        teste_preview["Saida"] = y_teste.reset_index(drop=True)
        contexto["previa_teste"] = (
            teste_preview.head(8).round(4).to_dict(orient="records")
        )

    return contexto


# ==========================================================
# APRENDIZADO SUPERVISIONADO - CLASSIFICAÇÃO
# ==========================================================

@app.route(
    "/supervisionado/classificacao",
    methods=["GET", "POST"]
)
def classificacao():

    mensagem = None
    erro = None

    if request.method == "POST":
        acao = request.form.get("acao")

        try:

            # ==================================================
            # PASSO 1 - IMPORTAÇÃO
            # ==================================================
            if acao == "importar":
                arquivo = request.files.get("arquivo")

                if not arquivo or not arquivo.filename:
                    raise ValueError(
                        "Selecione um arquivo para continuar."
                    )

                dados = ler_planilha(arquivo)

                if dados.empty:
                    raise ValueError(
                        "A base de dados está vazia."
                    )

                estado_classificacao["df_original"] = dados.copy()
                estado_classificacao["nome_arquivo"] = arquivo.filename

                resetar_classificacao_apos_importacao()

                mensagem = "Base de dados importada com sucesso."


            # ==================================================
            # PASSO 3 - PREPARAÇÃO
            # ==================================================
            elif acao == "preparar":
                dados = estado_classificacao.get("df_original")

                if dados is None:
                    raise ValueError(
                        "Importe uma base antes da preparação."
                    )

                preparado = (
                    dados
                    .drop_duplicates()
                    .copy()
                )

                # Valores ausentes de X serão tratados por imputadores
                # ajustados apenas aos dados de treino, sem vazamento.
                # Ausentes de Y não são inventados: serão excluídos
                # da modelagem ao dividir a base.

                preparado = preparado.reset_index(drop=True)

                estado_classificacao["df_preparado"] = preparado
                estado_classificacao["preparacao_concluida"] = True
                estado_classificacao["etapa_atual"] = 4

                resetar_classificacao_apos_preparacao()

                estado_classificacao["preparacao_concluida"] = True
                estado_classificacao["etapa_atual"] = 4

                mensagem = (
                    "Preparação concluída: duplicados removidos. "
                    "Os valores ausentes de entrada serão tratados somente após a divisão."
                )


            # ==================================================
            # PASSO 4 - VARIÁVEIS
            # ==================================================
            elif acao == "selecionar_variaveis":
                dados = estado_classificacao.get("df_preparado")

                if dados is None:
                    raise ValueError(
                        "Prepare a base antes de selecionar as variáveis."
                    )

                entradas = request.form.getlist("entradas")
                saida = request.form.get("saida")

                if not entradas:
                    raise ValueError(
                        "Selecione pelo menos uma variável de entrada."
                    )

                if not saida:
                    raise ValueError(
                        "Selecione uma variável de saída."
                    )

                if saida in entradas:
                    raise ValueError(
                        "A variável de saída não pode ser usada como entrada."
                    )

                if saida not in dados.columns:
                    raise ValueError("A variável de saída não existe na base.")
                opcoes = {x["coluna"]: x for x in listar_variaveis_classificacao(dados)}
                if len(set(entradas)) != len(entradas):
                    raise ValueError("Há variáveis de entrada repetidas.")
                for coluna in entradas:
                    if coluna not in opcoes or not opcoes[coluna]["pode_entrada"]:
                        raise ValueError(f"A coluna {coluna} não pode ser usada como entrada.")
                if not opcoes[saida]["pode_saida"]:
                    raise ValueError("A saída precisa conter de 2 a 30 classes identificáveis.")
                valores_saida = dados[saida].dropna().unique().tolist()
                valores_ordenados = sorted(valores_saida, key=lambda v: str(v).casefold())
                mapa_saida = {valor: indice for indice, valor in enumerate(valores_ordenados)}

                estado_classificacao["entradas"] = entradas
                estado_classificacao["saida"] = saida
                estado_classificacao["mapa_saida"] = mapa_saida
                estado_classificacao["variaveis_confirmadas"] = True

                estado_classificacao["X_treino"] = None
                estado_classificacao["X_teste"] = None
                estado_classificacao["y_treino"] = None
                estado_classificacao["y_teste"] = None
                estado_classificacao["scaler"] = None
                estado_classificacao["algoritmo"] = None
                estado_classificacao["modelo"] = None
                estado_classificacao["avaliacao"] = None
                estado_classificacao["resultado_predicao"] = None
                estado_classificacao["resultado_predicao_df"] = None
                estado_classificacao["comparacao"] = None
                estado_classificacao["divisao_concluida"] = False
                estado_classificacao["algoritmo_selecionado"] = False
                estado_classificacao["modelo_treinado"] = False
                estado_classificacao["avaliacao_concluida"] = False
                estado_classificacao["predicao_concluida"] = False
                estado_classificacao["comparacao_concluida"] = False
                estado_classificacao["etapa_atual"] = 5

                mensagem = "Variáveis confirmadas com sucesso."


            # ==================================================
            # PASSO 5 - TREINO / TESTE
            # ==================================================
            elif acao == "dividir":
                dados = estado_classificacao.get("df_preparado")
                entradas = estado_classificacao.get("entradas", [])
                saida = estado_classificacao.get("saida")
                mapa_saida = estado_classificacao.get("mapa_saida")

                if (
                    dados is None
                    or not entradas
                    or not saida
                    or not mapa_saida
                ):
                    raise ValueError(
                        "Confirme as variáveis antes da divisão."
                    )

                percentual = int(
                    request.form.get("percentual_treino", 80)
                )

                if percentual < 50 or percentual > 90:
                    raise ValueError(
                        "Use um percentual de treino entre 50% e 90%."
                    )

                X = dados[entradas].copy()
                y = dados[saida].map(mapa_saida)
                validos = y.notna()
                if validos.sum() < 4:
                    raise ValueError("A base precisa ter pelo menos 4 exemplos classificados.")
                X, y = X.loc[validos], y.loc[validos].astype(int)
                contagem = y.value_counts()
                if (contagem < 2).any():
                    raise ValueError("Cada classe precisa ter ao menos 2 registros para dividir treino e teste.")
                n_classes = len(contagem)
                n_testes = int(np.ceil(len(y) * (100 - percentual) / 100))
                n_treino = len(y) - n_testes
                if min(n_treino, n_testes) < n_classes:
                    raise ValueError("Não há registros suficientes para manter todas as classes nos conjuntos de treino e teste. Aumente a base ou ajuste a divisão.")
                X_treino_bruto, X_teste_bruto, y_treino, y_teste = train_test_split(
                    X, y, test_size=(100 - percentual) / 100,
                    random_state=42, stratify=y
                )
                preprocessador = criar_preprocessador_classificacao(dados, entradas)
                treino_transformado = preprocessador.fit_transform(X_treino_bruto)
                teste_transformado = preprocessador.transform(X_teste_bruto)
                nomes_transformados = list(preprocessador.get_feature_names_out())
                X_treino = pd.DataFrame(treino_transformado, columns=nomes_transformados)
                X_teste = pd.DataFrame(teste_transformado, columns=nomes_transformados)
                scaler = preprocessador
                estado_classificacao["colunas_transformadas"] = nomes_transformados

                estado_classificacao["X_treino"] = X_treino
                estado_classificacao["X_teste"] = X_teste
                estado_classificacao["y_treino"] = y_treino.reset_index(drop=True)
                estado_classificacao["y_teste"] = y_teste.reset_index(drop=True)
                estado_classificacao["scaler"] = scaler
                estado_classificacao["percentual_treino"] = percentual
                estado_classificacao["divisao_concluida"] = True

                estado_classificacao["algoritmo"] = None
                estado_classificacao["modelo"] = None
                estado_classificacao["avaliacao"] = None
                estado_classificacao["resultado_predicao"] = None
                estado_classificacao["resultado_predicao_df"] = None
                estado_classificacao["comparacao"] = None
                estado_classificacao["algoritmo_selecionado"] = False
                estado_classificacao["modelo_treinado"] = False
                estado_classificacao["avaliacao_concluida"] = False
                estado_classificacao["predicao_concluida"] = False
                estado_classificacao["comparacao_concluida"] = False
                estado_classificacao["etapa_atual"] = 6

                mensagem = (
                    f"Base dividida em {len(X_treino)} registros de treino "
                    f"e {len(X_teste)} registros de teste."
                )


            # ==================================================
            # PASSO 6 - ESCOLHA DO ALGORITMO
            # ==================================================
            elif acao == "selecionar_algoritmo":
                if not estado_classificacao.get(
                    "divisao_concluida"
                ):
                    raise ValueError(
                        "Divida a base antes de escolher o algoritmo."
                    )

                algoritmo = request.form.get("algoritmo")

                if algoritmo not in {
                    "logistica",
                    "knn",
                    "arvore",
                    "random_forest"
                }:
                    raise ValueError(
                        "Selecione um algoritmo válido."
                    )

                estado_classificacao["algoritmo"] = algoritmo
                estado_classificacao["algoritmo_selecionado"] = True
                estado_classificacao["modelo"] = None
                estado_classificacao["avaliacao"] = None
                estado_classificacao["resultado_predicao"] = None
                estado_classificacao["resultado_predicao_df"] = None
                estado_classificacao["comparacao"] = None
                estado_classificacao["modelo_treinado"] = False
                estado_classificacao["avaliacao_concluida"] = False
                estado_classificacao["predicao_concluida"] = False
                estado_classificacao["comparacao_concluida"] = False
                estado_classificacao["etapa_atual"] = 7

                mensagem = (
                    "Algoritmo selecionado: "
                    + nome_amigavel_algoritmo(algoritmo)
                )


            # ==================================================
            # PASSO 7 - TREINAMENTO
            # ==================================================
            elif acao == "treinar":
                algoritmo = estado_classificacao.get("algoritmo")
                X_treino = estado_classificacao.get("X_treino")
                y_treino = estado_classificacao.get("y_treino")

                if (
                    not algoritmo
                    or X_treino is None
                    or y_treino is None
                ):
                    raise ValueError(
                        "Escolha o algoritmo antes do treinamento."
                    )

                modelo = criar_modelo_classificacao(
                    algoritmo,
                    len(X_treino)
                )

                modelo.fit(
                    X_treino,
                    y_treino
                )

                estado_classificacao["modelo"] = modelo
                estado_classificacao["modelo_treinado"] = True
                estado_classificacao["avaliacao"] = None
                estado_classificacao["resultado_predicao"] = None
                estado_classificacao["resultado_predicao_df"] = None
                estado_classificacao["comparacao"] = None
                estado_classificacao["avaliacao_concluida"] = False
                estado_classificacao["predicao_concluida"] = False
                estado_classificacao["comparacao_concluida"] = False
                estado_classificacao["etapa_atual"] = 8

                mensagem = (
                    "Modelo treinado com sucesso utilizando "
                    + nome_amigavel_algoritmo(algoritmo)
                    + "."
                )


            # ==================================================
            # PASSO 8 - AVALIAÇÃO
            # ==================================================
            elif acao == "avaliar":
                modelo = estado_classificacao.get("modelo")
                X_teste = estado_classificacao.get("X_teste")
                y_teste = estado_classificacao.get("y_teste")

                if (
                    modelo is None
                    or X_teste is None
                    or y_teste is None
                ):
                    raise ValueError(
                        "Treine o modelo antes da avaliação."
                    )

                estado_classificacao["avaliacao"] = calcular_avaliacao_classificacao(
                    modelo, X_teste, y_teste, estado_classificacao["mapa_saida"]
                )

                estado_classificacao["avaliacao_concluida"] = True
                estado_classificacao["etapa_atual"] = 9

                mensagem = "Avaliação concluída com sucesso."


            # ==================================================
            # PASSO 9 - PREDIÇÃO
            # ==================================================
            elif acao == "predizer":
                modelo = estado_classificacao.get("modelo")
                scaler = estado_classificacao.get("scaler")
                entradas = estado_classificacao.get("entradas", [])
                mapa_saida = estado_classificacao.get("mapa_saida", {})

                if (
                    modelo is None
                    or scaler is None
                    or not entradas
                ):
                    raise ValueError(
                        "Avalie o modelo antes de realizar novas predições."
                    )

                arquivo = request.files.get("arquivo_predicao")

                if not arquivo or not arquivo.filename:
                    raise ValueError(
                        "Selecione uma base para predição."
                    )

                novos_dados = ler_planilha(arquivo)

                if novos_dados.empty:
                    raise ValueError(
                        "A base de predição está vazia."
                    )

                ausentes = [
                    coluna
                    for coluna in entradas
                    if coluna not in novos_dados.columns
                ]

                if ausentes:
                    raise ValueError(
                        "A base não contém as variáveis obrigatórias: "
                        + ", ".join(ausentes)
                    )

                dados_modelo = novos_dados[entradas].copy()
                # O mesmo preprocessador do treino lida com textos,
                # categorias não vistas, ausentes e dados numéricos.
                try:
                    dados_normalizados = scaler.transform(dados_modelo)
                except Exception as exc:
                    raise ValueError(f"Não foi possível preparar a base de predição: {exc}") from exc

                dados_tratados = pd.DataFrame(
                    dados_normalizados,
                    columns=estado_classificacao["colunas_transformadas"]
                )
                previstos = modelo.predict(dados_tratados)

                mapa_inverso = {
                    valor: chave
                    for chave, valor
                    in mapa_saida.items()
                }

                resultado = novos_dados.copy()
                resultado["Previsao"] = [
                    int(valor)
                    for valor in previstos
                ]
                resultado["Classe_Prevista"] = [
                    mapa_inverso.get(
                        int(valor),
                        int(valor)
                    )
                    for valor in previstos
                ]

                estado_classificacao["resultado_predicao_df"] = resultado.copy()

                estado_classificacao["resultado_predicao"] = (
                    resultado
                    .head(30)
                    .where(pd.notnull(resultado.head(30)), None)
                    .to_dict(orient="records")
                )

                estado_classificacao["predicao_concluida"] = True
                estado_classificacao["etapa_atual"] = 10

                mensagem = (
                    f"Predição concluída para {len(resultado)} registro(s)."
                )


            # ==================================================
            # PASSO 10 - COMPARAÇÃO
            # ==================================================
            elif acao == "comparar":
                X_treino = estado_classificacao.get("X_treino")
                X_teste = estado_classificacao.get("X_teste")
                y_treino = estado_classificacao.get("y_treino")
                y_teste = estado_classificacao.get("y_teste")

                if (
                    X_treino is None
                    or X_teste is None
                    or y_treino is None
                    or y_teste is None
                ):
                    raise ValueError(
                        "Divida a base antes de comparar os algoritmos."
                    )

                algoritmos = [
                    "logistica",
                    "knn",
                    "arvore",
                    "random_forest"
                ]

                comparacao = []

                for algoritmo in algoritmos:
                    modelo = criar_modelo_classificacao(
                        algoritmo,
                        len(X_treino)
                    )

                    modelo.fit(
                        X_treino,
                        y_treino
                    )

                    previstos = modelo.predict(
                        X_teste
                    )

                    comparacao.append({
                        "algoritmo": nome_amigavel_algoritmo(
                            algoritmo
                        ),
                        "acuracia": round(
                            accuracy_score(
                                y_teste,
                                previstos
                            ) * 100,
                            2
                        ),
                        "precisao": round(
                            precision_score(
                                y_teste,
                                previstos,
                                average=("macro" if len(estado_classificacao["mapa_saida"]) > 2 else "binary"),
                                zero_division=0
                            ) * 100,
                            2
                        ),
                        "recall": round(
                            recall_score(
                                y_teste,
                                previstos,
                                average=("macro" if len(estado_classificacao["mapa_saida"]) > 2 else "binary"),
                                zero_division=0
                            ) * 100,
                            2
                        ),
                        "f1": round(
                            f1_score(
                                y_teste,
                                previstos,
                                average=("macro" if len(estado_classificacao["mapa_saida"]) > 2 else "binary"),
                                zero_division=0
                            ) * 100,
                            2
                        ),
                    })

                estado_classificacao["comparacao"] = comparacao
                estado_classificacao["comparacao_concluida"] = True
                estado_classificacao["etapa_atual"] = 10

                mensagem = (
                    "Comparação dos algoritmos concluída."
                )

        except Exception as exc:
            erro = str(exc)

    contexto = montar_contexto_classificacao(
        mensagem=mensagem,
        erro=erro
    )

    return render_template(
        "classificacao.html",
        **contexto
    )




# ==========================================================
# CLASSIFICAÇÃO - TREINAMENTO COM TERMINAL EM TEMPO REAL
# ==========================================================

@app.route(
    "/supervisionado/classificacao/treinar-stream"
)
def treinar_stream_classificacao():

    def gerar_eventos():

        algoritmo = estado_classificacao.get(
            "algoritmo"
        )

        X_treino = estado_classificacao.get(
            "X_treino"
        )

        X_teste = estado_classificacao.get(
            "X_teste"
        )

        y_treino = estado_classificacao.get(
            "y_treino"
        )

        entradas = estado_classificacao.get(
            "entradas",
            []
        )

        saida = estado_classificacao.get(
            "saida"
        )


        def enviar(
            mensagem,
            tipo="info",
            **extras
        ):

            dados = {

                "mensagem":
                    mensagem,

                "tipo":
                    tipo

            }

            dados.update(
                extras
            )

            return (
                "data: "
                +
                json.dumps(
                    dados,
                    ensure_ascii=False
                )
                +
                "\n\n"
            )


        if (
            not algoritmo
            or X_treino is None
            or y_treino is None
        ):

            yield enviar(
                "Não foi possível iniciar o treinamento. "
                "Verifique a seleção do algoritmo e a divisão da base.",
                "erro"
            )

            return


        try:

            nome_algoritmo = (
                nome_amigavel_algoritmo(
                    algoritmo
                )
            )


            yield enviar(
                "==============================================",
                "separador"
            )

            yield enviar(
                "INICIANDO TREINAMENTO DO MODELO",
                "inicio"
            )

            yield enviar(
                "==============================================",
                "separador"
            )

            time.sleep(0.25)


            yield enviar(
                f"Algoritmo selecionado: {nome_algoritmo}",
                "etapa"
            )

            yield enviar(
                f"Registros de treinamento: {len(X_treino)}",
                "info"
            )

            yield enviar(
                f"Registros de teste: "
                f"{len(X_teste) if X_teste is not None else 0}",
                "info"
            )

            yield enviar(
                f"Variável de saída: {saida}",
                "info"
            )

            time.sleep(0.20)


            yield enviar(
                "",
                "separador"
            )

            yield enviar(
                "Variáveis de entrada:",
                "etapa"
            )


            for indice, entrada in enumerate(
                entradas,
                start=1
            ):

                yield enviar(
                    f"  X{indice} = {entrada}",
                    "calculo"
                )

                time.sleep(0.08)


            yield enviar(
                "",
                "separador"
            )

            yield enviar(
                "Pré-processamento:",
                "etapa"
            )

            yield enviar(
                "  ✓ Base já dividida em treino e teste (estratificada)",
                "sucesso"
            )

            yield enviar(
                "  ✓ Codificação de categorias e normalização numérica "
                "ajustadas somente com os dados de treino",
                "sucesso"
            )

            yield enviar(
                "  ✓ Mesmo pré-processamento aplicado aos dados de teste",
                "sucesso"
            )

            time.sleep(0.25)


            yield enviar(
                "",
                "separador"
            )

            yield enviar(
                f"Criando modelo: {nome_algoritmo}",
                "etapa"
            )


            modelo = criar_modelo_classificacao(
                algoritmo,
                len(X_treino)
            )


            if algoritmo == "logistica":

                yield enviar(
                    "A Regressão Logística buscará uma combinação "
                    "das entradas capaz de estimar a probabilidade "
                    "das diferentes classes de saída.",
                    "info"
                )


            elif algoritmo == "knn":

                quantidade_vizinhos = (
                    modelo.n_neighbors
                )

                yield enviar(
                    f"KNN configurado com K = "
                    f"{quantidade_vizinhos} vizinhos.",
                    "info"
                )

                yield enviar(
                    "O algoritmo armazenará os registros de treino "
                    "para comparar a proximidade de novos casos.",
                    "info"
                )


            elif algoritmo == "arvore":

                yield enviar(
                    "A Árvore de Decisão aprenderá regras "
                    "de separação entre as classes.",
                    "info"
                )


            elif algoritmo == "random_forest":

                yield enviar(
                    f"Random Forest configurado com "
                    f"{modelo.n_estimators} árvores.",
                    "info"
                )

                yield enviar(
                    "As árvores serão combinadas para formar "
                    "a classificação final.",
                    "info"
                )


            time.sleep(0.30)


            yield enviar(
                "",
                "separador"
            )

            yield enviar(
                "Executando modelo.fit(X_treino, y_treino)...",
                "destaque"
            )


            inicio_treinamento = time.perf_counter()


            modelo.fit(
                X_treino,
                y_treino
            )


            tempo_treinamento = (
                time.perf_counter()
                -
                inicio_treinamento
            )


            yield enviar(
                "✓ Ajuste do modelo concluído.",
                "sucesso"
            )


            # Informações reais do modelo treinado
            if algoritmo == "logistica":

                coeficientes = modelo.coef_[0]
                intercepto = float(modelo.intercept_[0])
                yield enviar(
                    "Regressão logística: exibindo coeficientes da primeira "
                    "equação; em multiclasse existem equações por classe.",
                    "info"
                )


                yield enviar(
                    "",
                    "separador"
                )

                yield enviar(
                    "Parâmetros aprendidos:",
                    "etapa"
                )


                for indice, coeficiente in enumerate(
                    coeficientes
                ):

                    nome_entrada = (
                        estado_classificacao.get("colunas_transformadas", [])[indice]
                        if indice < len(estado_classificacao.get("colunas_transformadas", []))
                        else f"X{indice + 1}"
                    )

                    yield enviar(
                        f"  {nome_entrada}: "
                        f"coeficiente = {float(coeficiente):.6f}",
                        "calculo"
                    )


                yield enviar(
                    f"  Intercepto = {intercepto:.6f}",
                    "calculo"
                )


            elif algoritmo == "arvore":

                yield enviar(
                    "",
                    "separador"
                )

                yield enviar(
                    "Estrutura aprendida:",
                    "etapa"
                )

                yield enviar(
                    f"  Profundidade da árvore: "
                    f"{modelo.get_depth()}",
                    "calculo"
                )

                yield enviar(
                    f"  Quantidade de folhas: "
                    f"{modelo.get_n_leaves()}",
                    "calculo"
                )


            elif algoritmo == "random_forest":

                profundidades = [
                    arvore.get_depth()
                    for arvore
                    in modelo.estimators_
                ]

                profundidade_media = (
                    sum(profundidades)
                    /
                    len(profundidades)
                )


                yield enviar(
                    "",
                    "separador"
                )

                yield enviar(
                    "Estrutura aprendida:",
                    "etapa"
                )

                yield enviar(
                    f"  Árvores treinadas: "
                    f"{len(modelo.estimators_)}",
                    "calculo"
                )

                yield enviar(
                    f"  Profundidade média: "
                    f"{profundidade_media:.2f}",
                    "calculo"
                )


            elif algoritmo == "knn":

                yield enviar(
                    "",
                    "separador"
                )

                yield enviar(
                    "Estrutura do KNN:",
                    "etapa"
                )

                yield enviar(
                    f"  Registros armazenados: "
                    f"{len(X_treino)}",
                    "calculo"
                )

                yield enviar(
                    f"  Número de vizinhos (K): "
                    f"{modelo.n_neighbors}",
                    "calculo"
                )


            estado_classificacao[
                "modelo"
            ] = modelo


            estado_classificacao[
                "modelo_treinado"
            ] = True


            estado_classificacao[
                "avaliacao"
            ] = None


            estado_classificacao[
                "resultado_predicao"
            ] = None
            estado_classificacao["resultado_predicao_df"] = None


            estado_classificacao[
                "comparacao"
            ] = None


            estado_classificacao[
                "avaliacao_concluida"
            ] = False


            estado_classificacao[
                "predicao_concluida"
            ] = False


            estado_classificacao[
                "comparacao_concluida"
            ] = False


            estado_classificacao[
                "etapa_atual"
            ] = 7


            yield enviar(
                "",
                "separador"
            )

            yield enviar(
                f"Tempo de treinamento: "
                f"{tempo_treinamento:.4f} s",
                "info"
            )

            yield enviar(
                "Modelo salvo e pronto para avaliação.",
                "sucesso"
            )

            yield enviar(
                "==============================================",
                "separador"
            )


            yield enviar(
                "TREINAMENTO CONCLUÍDO COM SUCESSO.",
                "fim",
                algoritmo=nome_algoritmo,
                registros_treino=len(X_treino),
                quantidade_entradas=len(entradas)
            )


        except Exception as erro:

            yield enviar(
                "Erro durante o treinamento: "
                + str(erro),
                "erro"
            )


    return Response(
        gerar_eventos(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no"
        }
    )


# ==========================================================
# CLASSIFICAÇÃO - AVALIAÇÃO ASSÍNCRONA
# MESMO PADRÃO DA TELA DE REDE NEURAL
# ==========================================================

@app.route("/supervisionado/classificacao/avaliar-modelo")
def avaliar_modelo_classificacao():
    modelo = estado_classificacao.get("modelo")
    X_teste = estado_classificacao.get("X_teste")
    y_teste = estado_classificacao.get("y_teste")
    if modelo is None:
        return jsonify({"sucesso": False, "mensagem": "Treine o modelo antes de avaliá-lo."})
    if X_teste is None or y_teste is None or len(X_teste) == 0:
        return jsonify({"sucesso": False, "mensagem": "Não há dados reservados para avaliação."})
    try:
        avaliacao = calcular_avaliacao_classificacao(
            modelo, X_teste, y_teste, estado_classificacao["mapa_saida"]
        )
        estado_classificacao["avaliacao"] = avaliacao
        estado_classificacao["avaliacao_concluida"] = True
        estado_classificacao["etapa_atual"] = 8
        resposta = dict(avaliacao, sucesso=True, erros=avaliacao["total"] - avaliacao["acertos"])
        # Preserva a API anterior para classificações binárias.
        resposta["matriz"] = {
            chave: avaliacao[chave] for chave in ("vp", "vn", "fp", "fn")
        } if not avaliacao["multiclasse"] else None
        return jsonify(resposta)
    except Exception as exc:
        return jsonify({"sucesso": False, "mensagem": str(exc)})


# ==========================================================
# DOWNLOAD - PREDIÇÕES DA CLASSIFICAÇÃO
# ==========================================================

@app.route(
    "/supervisionado/classificacao/baixar-predicoes"
)
def baixar_predicoes_classificacao():

    resultado = estado_classificacao.get(
        "resultado_predicao_df"
    )

    if resultado is None:

        return (
            "Nenhuma predição de classificação disponível.",
            400
        )

    arquivo_excel = BytesIO()

    with pd.ExcelWriter(
        arquivo_excel,
        engine="openpyxl"
    ) as writer:

        resultado.to_excel(
            writer,
            index=False,
            sheet_name="Predicoes"
        )

    arquivo_excel.seek(0)

    return send_file(
        arquivo_excel,
        as_attachment=True,
        download_name=
            "resultado_classificacao.xlsx",
        mimetype=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )



# ==========================================================
# APRENDIZADO SUPERVISIONADO - CLASSIFICAÇÃO DE IMAGENS
# ==========================================================

estado_imagens = {
    "diretorio": None,
    "nome_arquivo": None,
    "classes": [],
    "arquivos": [],
    "amostras": [],
    "largura": 96,
    "altura": 96,
    "percentual_treino": 80,
    "modelo_tipo": "cnn",
    "modelo": None,
    "historico": None,
    "X_treino": None,
    "X_teste": None,
    "y_treino": None,
    "y_teste": None,
    "nomes_teste": [],
    "avaliacao": None,
    "predicao": None,
    "comparacao": None,
    "preparacao_concluida": False,
    "classes_confirmadas": False,
    "divisao_concluida": False,
    "modelo_selecionado": False,
    "modelo_treinado": False,
    "avaliacao_concluida": False,
    "predicao_concluida": False,
    "comparacao_concluida": False,
    "etapa_atual": 1,
    "etapa_maxima": 1,
}

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ValueError(
            "TensorFlow não está instalado. Adicione tensorflow ao requirements.txt "
            "para treinar a classificação de imagens."
        )

def _limpar_diretorio_imagens():
    diretorio = estado_imagens.get("diretorio")
    if diretorio and os.path.isdir(diretorio):
        shutil.rmtree(diretorio, ignore_errors=True)

def _reset_imagens_total():
    _limpar_diretorio_imagens()
    estado_imagens.update({
        "diretorio": None, "nome_arquivo": None, "classes": [], "arquivos": [],
        "amostras": [], "modelo": None, "historico": None,
        "X_treino": None, "X_teste": None, "y_treino": None, "y_teste": None,
        "nomes_teste": [], "avaliacao": None, "predicao": None, "comparacao": None,
        "preparacao_concluida": False, "classes_confirmadas": False,
        "divisao_concluida": False, "modelo_selecionado": False,
        "modelo_treinado": False, "avaliacao_concluida": False,
        "predicao_concluida": False, "comparacao_concluida": False,
        "etapa_atual": 1, "etapa_maxima": 1
    })

def _imagem_data_uri(caminho, tamanho=(180, 180)):
    try:
        with Image.open(caminho) as im:
            im = im.convert("RGB")
            im.thumbnail(tamanho)
            buff = BytesIO()
            im.save(buff, format="JPEG", quality=82)
            return "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode("ascii")
    except Exception:
        return None

def _extrair_zip_imagens(arquivo):
    if not arquivo.filename.lower().endswith(".zip"):
        raise ValueError("Envie um arquivo ZIP com uma pasta para cada classe.")

    tmp = tempfile.mkdtemp(prefix="lab_ml_imagens_")
    zip_path = os.path.join(tmp, "base.zip")
    arquivo.save(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            p = Path(info.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("O ZIP contém um caminho inválido.")
        z.extractall(tmp)

    os.remove(zip_path)

    arquivos = []
    for raiz, _, nomes in os.walk(tmp):
        for nome in nomes:
            caminho = os.path.join(raiz, nome)
            if Path(nome).suffix.lower() not in EXTENSOES_IMAGEM:
                continue
            rel = Path(os.path.relpath(caminho, tmp))
            partes = [p for p in rel.parts if not p.startswith(".") and p != "__MACOSX"]
            if len(partes) < 2:
                continue
            classe = partes[-2]
            try:
                with Image.open(caminho) as im:
                    largura, altura = im.size
                    modo = im.mode
                    im.verify()
                arquivos.append({
                    "caminho": caminho, "nome": nome, "classe": classe,
                    "largura": largura, "altura": altura, "modo": modo
                })
            except Exception:
                continue

    classes = sorted({x["classe"] for x in arquivos})
    if len(classes) < 2:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(
            "A base precisa possuir pelo menos duas classes. "
            "Organize o ZIP como classe_1/imagem.jpg, classe_2/imagem.jpg..."
        )
    if len(arquivos) < 4:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("A base possui poucas imagens válidas.")

    return tmp, classes, arquivos

def _carregar_arrays_imagens():
    classes = estado_imagens["classes"]
    indice = {c: i for i, c in enumerate(classes)}
    largura, altura = estado_imagens["largura"], estado_imagens["altura"]
    X, y, nomes = [], [], []
    for item in estado_imagens["arquivos"]:
        try:
            with Image.open(item["caminho"]) as im:
                im = im.convert("RGB").resize((largura, altura))
                arr = np.asarray(im, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(indice[item["classe"]])
            nomes.append(item["nome"])
        except Exception:
            continue
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), nomes

def _criar_modelo_imagens(tipo, n_classes):
    tf = _tf()
    if tipo == "densa":
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(estado_imagens["altura"], estado_imagens["largura"], 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(n_classes, activation="softmax")
        ])
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(estado_imagens["altura"], estado_imagens["largura"], 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

def _contexto_imagens(mensagem=None, erro=None):
    contagem = {c: 0 for c in estado_imagens["classes"]}
    for item in estado_imagens["arquivos"]:
        contagem[item["classe"]] = contagem.get(item["classe"], 0) + 1
    return {
        **estado_imagens,
        "mensagem": mensagem, "erro": erro,
        "total_imagens": len(estado_imagens["arquivos"]),
        "contagem_classes": contagem,
        "qtd_treino": 0 if estado_imagens["X_treino"] is None else len(estado_imagens["X_treino"]),
        "qtd_teste": 0 if estado_imagens["X_teste"] is None else len(estado_imagens["X_teste"]),
    }

@app.route("/supervisionado/classificacao-imagens/treinar-stream", methods=["POST"])
def treinar_stream_imagens():
    """Treina a rede por épocas e transmite o progresso para o terminal da Etapa 7."""
    # IMPORTANTE: tudo que depende de request precisa ser lido enquanto
    # o contexto HTTP ainda está ativo. O generator abaixo continua sendo
    # executado depois que a função da rota retorna.
    try:
        epocas_solicitadas = int(request.form.get("epocas", 8))
    except (TypeError, ValueError):
        epocas_solicitadas = 8
    epocas_solicitadas = min(max(epocas_solicitadas, 1), 30)

    def evento(tipo, mensagem, **dados):
        payload = {"tipo": tipo, "mensagem": mensagem, **dados}
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    def gerar():
        try:
            if not estado_imagens["modelo_selecionado"]:
                yield evento("erro", "Selecione o modelo antes do treinamento.")
                return
            if estado_imagens["X_treino"] is None or estado_imagens["X_teste"] is None:
                yield evento("erro", "Divida a base em treino e teste antes do treinamento.")
                return

            epocas = epocas_solicitadas

            tf = _tf()
            np.random.seed(42)
            tf.random.set_seed(42)
            modelo = _criar_modelo_imagens(
                estado_imagens["modelo_tipo"], len(estado_imagens["classes"])
            )
            modelo.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            nome_modelo = "CNN" if estado_imagens["modelo_tipo"] == "cnn" else "Rede Neural Densa"
            yield evento("linha", "$ Iniciando treinamento...")
            yield evento("linha", f"$ Modelo: {nome_modelo}")
            yield evento("linha", f"$ Imagens de treinamento: {len(estado_imagens['X_treino'])}")
            yield evento("linha", f"$ Imagens de teste: {len(estado_imagens['X_teste'])}")
            yield evento("linha", f"$ Dimensão das imagens: {estado_imagens['largura']}×{estado_imagens['altura']} RGB")
            yield evento("linha", "$ Pixels normalizados: 0–1")
            yield evento("linha", f"$ Total de épocas: {epocas}\n")

            historico_total = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            for epoca in range(1, epocas + 1):
                h = modelo.fit(
                    estado_imagens["X_treino"], estado_imagens["y_treino"],
                    validation_data=(estado_imagens["X_teste"], estado_imagens["y_teste"]),
                    epochs=1, batch_size=16, verbose=0
                ).history

                valores = {}
                for chave in historico_total:
                    valor = float(h.get(chave, [0.0])[0])
                    historico_total[chave].append(round(valor, 5))
                    valores[chave] = valor

                yield evento(
                    "epoca",
                    (
                        f"$ Época {epoca}/{epocas}\n"
                        f"  loss: {valores['loss']:.4f}\n"
                        f"  accuracy: {valores['accuracy'] * 100:.2f}%\n"
                        f"  val_loss: {valores['val_loss']:.4f}\n"
                        f"  val_accuracy: {valores['val_accuracy'] * 100:.2f}%\n"
                    ),
                    epoca=epoca,
                    total=epocas,
                    progresso=round(epoca * 100 / epocas)
                )

            estado_imagens.update({
                "modelo": modelo,
                "historico": historico_total,
                "modelo_treinado": True,
                "avaliacao": None,
                "avaliacao_concluida": False,
                "etapa_atual": 7,
                "etapa_maxima": max(estado_imagens["etapa_maxima"], 8)
            })
            yield evento("concluido", "✓ Treinamento concluído.\n✓ Modelo pronto para avaliação.", progresso=100)

        except Exception as exc:
            yield evento("erro", f"Erro durante o treinamento: {exc}")

    return Response(
        gerar(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/supervisionado/classificacao-imagens", methods=["GET", "POST"])
def classificacao_imagens():
    mensagem = erro = None

    if request.method == "POST":
        acao = request.form.get("acao")
        try:
            if acao == "resetar":
                _reset_imagens_total()
                mensagem = "Laboratório reiniciado."

            elif acao == "navegar_etapa":
                destino = int(request.form.get("etapa_destino", 1))
                maxima = int(estado_imagens.get("etapa_maxima", 1))
                if destino < 1 or destino > 10:
                    raise ValueError("Etapa inválida.")
                if destino > maxima:
                    raise ValueError("Conclua as etapas anteriores antes de avançar.")
                estado_imagens["etapa_atual"] = destino

            elif acao == "importar":
                arquivo = request.files.get("arquivo")
                if not arquivo or not arquivo.filename:
                    raise ValueError("Selecione uma base de imagens em formato ZIP.")
                _limpar_diretorio_imagens()
                diretorio, classes, arquivos = _extrair_zip_imagens(arquivo)
                amostras = []
                vistos = set()
                for item in arquivos:
                    if item["classe"] not in vistos:
                        uri = _imagem_data_uri(item["caminho"])
                        if uri:
                            amostras.append({**item, "uri": uri})
                            vistos.add(item["classe"])
                    if len(vistos) == len(classes):
                        break
                estado_imagens.update({
                    "diretorio": diretorio, "nome_arquivo": arquivo.filename,
                    "classes": classes, "arquivos": arquivos, "amostras": amostras,
                    "preparacao_concluida": False, "classes_confirmadas": False,
                    "divisao_concluida": False, "modelo_selecionado": False,
                    "modelo_treinado": False, "avaliacao_concluida": False,
                    "predicao_concluida": False, "comparacao_concluida": False,
                    "modelo": None, "avaliacao": None, "predicao": None, "comparacao": None,
                    "etapa_atual": 2, "etapa_maxima": 2
                })
                mensagem = f"{len(arquivos)} imagens válidas encontradas em {len(classes)} classes."

            elif acao == "preparar":
                if not estado_imagens["arquivos"]:
                    raise ValueError("Importe as imagens primeiro.")
                largura = int(request.form.get("largura", 96))
                altura = int(request.form.get("altura", 96))
                if largura < 32 or altura < 32 or largura > 256 or altura > 256:
                    raise ValueError("Use dimensões entre 32 e 256 pixels.")
                estado_imagens.update({
                    "largura": largura, "altura": altura,
                    "preparacao_concluida": True, "etapa_atual": 4,
                    "etapa_maxima": max(estado_imagens["etapa_maxima"], 4)
                })
                mensagem = (
                    f"Preparação configurada: RGB, {largura}×{altura} pixels e "
                    "normalização dos pixels de 0–255 para 0–1."
                )

            elif acao == "confirmar_classes":
                if not estado_imagens["preparacao_concluida"]:
                    raise ValueError("Conclua a preparação primeiro.")
                estado_imagens.update({
                    "classes_confirmadas": True, "etapa_atual": 5,
                    "etapa_maxima": max(estado_imagens["etapa_maxima"], 5)
                })
                mensagem = "Classes confirmadas: " + ", ".join(estado_imagens["classes"])

            elif acao == "dividir":
                if not estado_imagens["classes_confirmadas"]:
                    raise ValueError("Confirme as classes primeiro.")
                percentual = int(request.form.get("percentual_treino", 80))
                if percentual < 50 or percentual > 90:
                    raise ValueError("Use um percentual de treino entre 50% e 90%.")
                X, y, nomes = _carregar_arrays_imagens()
                if len(X) < 4:
                    raise ValueError("Não há imagens válidas suficientes.")
                _, counts = np.unique(y, return_counts=True)
                stratify = y if counts.min() >= 2 else None
                indices = np.arange(len(X))
                it, iv = train_test_split(
                    indices, test_size=(100-percentual)/100,
                    random_state=42, stratify=stratify
                )
                estado_imagens.update({
                    "X_treino": X[it], "X_teste": X[iv],
                    "y_treino": y[it], "y_teste": y[iv],
                    "nomes_teste": [nomes[i] for i in iv],
                    "percentual_treino": percentual, "divisao_concluida": True,
                    "etapa_atual": 6, "etapa_maxima": max(estado_imagens["etapa_maxima"], 6)
                })
                mensagem = f"Base dividida em {len(it)} imagens de treino e {len(iv)} de teste."

            elif acao == "selecionar_modelo":
                if not estado_imagens["divisao_concluida"]:
                    raise ValueError("Divida a base antes de selecionar o modelo.")
                tipo = request.form.get("modelo_tipo", "cnn")
                if tipo not in {"cnn", "densa"}:
                    raise ValueError("Modelo inválido.")
                estado_imagens.update({
                    "modelo_tipo": tipo, "modelo_selecionado": True,
                    "modelo": None, "modelo_treinado": False,
                    "avaliacao": None, "avaliacao_concluida": False,
                    "etapa_atual": 7, "etapa_maxima": max(estado_imagens["etapa_maxima"], 7)
                })
                mensagem = "Modelo selecionado: " + ("CNN" if tipo == "cnn" else "Rede Neural Densa")

            elif acao == "treinar":
                if not estado_imagens["modelo_selecionado"]:
                    raise ValueError("Selecione o modelo antes do treinamento.")
                tf = _tf()
                np.random.seed(42)
                tf.random.set_seed(42)
                modelo = _criar_modelo_imagens(estado_imagens["modelo_tipo"], len(estado_imagens["classes"]))
                modelo.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )
                epocas = int(request.form.get("epocas", 8))
                epocas = min(max(epocas, 1), 30)
                historico = modelo.fit(
                    estado_imagens["X_treino"], estado_imagens["y_treino"],
                    validation_data=(estado_imagens["X_teste"], estado_imagens["y_teste"]),
                    epochs=epocas, batch_size=16, verbose=0
                )
                hist = {k: [round(float(v), 5) for v in vals] for k, vals in historico.history.items()}
                estado_imagens.update({
                    "modelo": modelo, "historico": hist, "modelo_treinado": True,
                    "avaliacao": None, "avaliacao_concluida": False,
                    "etapa_atual": 8, "etapa_maxima": max(estado_imagens["etapa_maxima"], 8)
                })
                mensagem = f"Treinamento concluído em {len(hist.get('loss', []))} época(s)."

            elif acao == "avaliar":
                modelo = estado_imagens["modelo"]
                if modelo is None:
                    raise ValueError("Treine o modelo antes da avaliação.")
                probabilidades = modelo.predict(estado_imagens["X_teste"], verbose=0)
                previstos = np.argmax(probabilidades, axis=1)
                y = estado_imagens["y_teste"]
                classes = estado_imagens["classes"]
                labels = list(range(len(classes)))
                ac = accuracy_score(y, previstos)
                pr = precision_score(y, previstos, average="weighted", zero_division=0)
                rc = recall_score(y, previstos, average="weighted", zero_division=0)
                f1v = f1_score(y, previstos, average="weighted", zero_division=0)
                matriz = confusion_matrix(y, previstos, labels=labels).tolist()
                resultados = []
                for i in range(len(y)):
                    resultados.append({
                        "registro": i+1,
                        "arquivo": estado_imagens["nomes_teste"][i],
                        "real": classes[int(y[i])],
                        "previsto": classes[int(previstos[i])],
                        "confianca": round(float(np.max(probabilidades[i]))*100, 2),
                        "resultado": "Acerto" if int(y[i]) == int(previstos[i]) else "Erro",
                        "imagem": "data:image/jpeg;base64," + base64.b64encode(
                            (lambda b: (Image.fromarray((estado_imagens["X_teste"][i]*255).astype(np.uint8)).save(b, format="JPEG"), b.getvalue())[1])(BytesIO())
                        ).decode("ascii")
                    })
                estado_imagens["avaliacao"] = {
                    "acuracia": round(ac*100,2), "precisao": round(pr*100,2),
                    "recall": round(rc*100,2), "f1": round(f1v*100,2),
                    "support": len(y), "matriz": matriz, "classes": classes,
                    "resultados": resultados
                }
                estado_imagens.update({
                    "avaliacao_concluida": True, "etapa_atual": 9,
                    "etapa_maxima": max(estado_imagens["etapa_maxima"], 9)
                })
                mensagem = "Avaliação concluída com sucesso."

            elif acao == "predizer":
                modelo = estado_imagens["modelo"]
                if modelo is None:
                    raise ValueError("Treine o modelo antes da predição.")
                arquivo = request.files.get("imagem_predicao")
                if not arquivo or not arquivo.filename:
                    raise ValueError("Selecione uma imagem.")
                with Image.open(arquivo.stream) as im:
                    im = im.convert("RGB")
                    original = im.copy()
                    im = im.resize((estado_imagens["largura"], estado_imagens["altura"]))
                    arr = np.asarray(im, dtype=np.float32)/255.0
                probs = modelo.predict(np.expand_dims(arr, 0), verbose=0)[0]
                idx = int(np.argmax(probs))
                buff = BytesIO()
                original.thumbnail((420,420))
                original.save(buff, format="JPEG", quality=88)
                estado_imagens["predicao"] = {
                    "classe": estado_imagens["classes"][idx],
                    "confianca": round(float(probs[idx])*100,2),
                    "imagem": "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode("ascii"),
                    "probabilidades": [
                        {"classe": c, "valor": round(float(probs[i])*100,2)}
                        for i,c in enumerate(estado_imagens["classes"])
                    ]
                }
                estado_imagens.update({
                    "predicao_concluida": True, "etapa_atual": 10,
                    "etapa_maxima": 10
                })
                mensagem = "Imagem classificada com sucesso."

            elif acao == "comparar":
                if estado_imagens["X_treino"] is None:
                    raise ValueError("Divida a base antes da comparação.")
                tf = _tf()
                comparacao = []
                for tipo in ("densa", "cnn"):
                    tf.keras.backend.clear_session()
                    modelo = _criar_modelo_imagens(tipo, len(estado_imagens["classes"]))
                    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                    modelo.fit(
                        estado_imagens["X_treino"], estado_imagens["y_treino"],
                        epochs=5, batch_size=16, verbose=0
                    )
                    probs = modelo.predict(estado_imagens["X_teste"], verbose=0)
                    pred = np.argmax(probs, axis=1)
                    comparacao.append({
                        "modelo": "CNN" if tipo == "cnn" else "Rede Neural Densa",
                        "acuracia": round(accuracy_score(estado_imagens["y_teste"], pred)*100,2),
                        "precisao": round(precision_score(estado_imagens["y_teste"], pred, average="weighted", zero_division=0)*100,2),
                        "recall": round(recall_score(estado_imagens["y_teste"], pred, average="weighted", zero_division=0)*100,2),
                        "f1": round(f1_score(estado_imagens["y_teste"], pred, average="weighted", zero_division=0)*100,2)
                    })
                estado_imagens["comparacao"] = comparacao
                estado_imagens["comparacao_concluida"] = True
                estado_imagens["etapa_atual"] = 10
                estado_imagens["etapa_maxima"] = 10
                mensagem = "Comparação concluída."

        except Exception as exc:
            erro = str(exc)

    return render_template("classificacao_imagens.html", **_contexto_imagens(mensagem, erro))


# ==========================================================

estado_regressao = {
    "df_original": None, "df_preparado": None, "nome_arquivo": None,
    "normalizacao_parametros": {}, "normalizacao_resumo": [],
    "entradas": [], "saida": None,
    "X_treino": None, "X_teste": None, "y_treino": None, "y_teste": None,
    "percentual_treino": 80, "algoritmo": None, "modelo": None,
    "avaliacao": None, "resultado_predicao": None, "resultado_predicao_df": None,
    "comparacao": None, "preparacao_concluida": False,
    "variaveis_confirmadas": False, "divisao_concluida": False,
    "algoritmo_selecionado": False, "modelo_treinado": False,
    "avaliacao_concluida": False, "predicao_concluida": False,
    "comparacao_concluida": False, "etapa_atual": 1, "etapa_maxima": 1,
}

def resetar_regressao_completa():
    estado_regressao.update({
        "df_original": None, "df_preparado": None, "nome_arquivo": None,
        "normalizacao_parametros": {}, "normalizacao_resumo": [], "entradas": [], "saida": None,
        "X_treino": None, "X_teste": None, "y_treino": None, "y_teste": None,
        "percentual_treino": 80, "algoritmo": None, "modelo": None,
        "avaliacao": None, "resultado_predicao": None, "resultado_predicao_df": None,
        "comparacao": None, "preparacao_concluida": False, "variaveis_confirmadas": False,
        "divisao_concluida": False, "algoritmo_selecionado": False, "modelo_treinado": False,
        "avaliacao_concluida": False, "predicao_concluida": False, "comparacao_concluida": False,
        "etapa_atual": 1,
    })

def limpar_regressao_depois(nivel):
    # nível: 2 importou, 4 preparou, 5 variáveis, 6 divisão, 7 algoritmo
    if nivel <= 2:
        estado_regressao.update({"df_preparado":None,"normalizacao_parametros":{},"normalizacao_resumo":[],"preparacao_concluida":False})
    if nivel <= 4:
        estado_regressao.update({"entradas":[],"saida":None,"variaveis_confirmadas":False})
    if nivel <= 5:
        estado_regressao.update({"X_treino":None,"X_teste":None,"y_treino":None,"y_teste":None,"divisao_concluida":False})
    if nivel <= 6:
        estado_regressao.update({"algoritmo":None,"algoritmo_selecionado":False})
    if nivel <= 7:
        estado_regressao.update({"modelo":None,"modelo_treinado":False})
    estado_regressao.update({"avaliacao":None,"avaliacao_concluida":False,"resultado_predicao":None,"resultado_predicao_df":None,"predicao_concluida":False,"comparacao":None,"comparacao_concluida":False})

def nome_algoritmo_regressao(nome):
    return {"linear":"Regressão Linear","ridge":"Regressão Ridge","knn":"KNN Regressor","arvore":"Árvore de Decisão","random_forest":"Random Forest"}.get(nome,nome)

def criar_modelo_regressao(nome, n):
    if nome == "linear": return LinearRegression()
    if nome == "ridge": return Ridge(alpha=1.0)
    if nome == "knn": return KNeighborsRegressor(n_neighbors=min(5,max(1,n)))
    if nome == "arvore": return DecisionTreeRegressor(max_depth=5, random_state=42)
    if nome == "random_forest": return RandomForestRegressor(n_estimators=100, random_state=42)
    raise ValueError("Algoritmo de regressão inválido.")

def contexto_regressao(mensagem=None, erro=None):
    o=estado_regressao.get("df_original"); p=estado_regressao.get("df_preparado")
    c={"mensagem":mensagem,"erro":erro, **{k:estado_regressao.get(k) for k in ["nome_arquivo","etapa_atual","etapa_maxima","preparacao_concluida","variaveis_confirmadas","entradas","saida","divisao_concluida","percentual_treino","algoritmo","modelo_treinado","avaliacao_concluida","avaliacao","predicao_concluida","resultado_predicao","comparacao_concluida","comparacao"]},
       "algoritmo_nome":nome_algoritmo_regressao(estado_regressao.get("algoritmo")) if estado_regressao.get("algoritmo") else None,
       "quantidade_registros":0,"quantidade_colunas":0,"colunas":[],"previa":[],"analise_colunas":[],"total_ausentes":0,"total_duplicados":0,"previa_preparada":[],"normalizacao_resumo":estado_regressao.get("normalizacao_resumo",[]),"variaveis_disponiveis":[],"qtd_treino":0,"qtd_teste":0}
    if o is not None:
        c.update(quantidade_registros=len(o),quantidade_colunas=len(o.columns),colunas=o.columns.tolist(),previa=o.head(10).where(pd.notnull(o.head(10)),None).to_dict('records'),total_ausentes=int(o.isnull().sum().sum()),total_duplicados=int(o.duplicated().sum()))
        for col in o.columns:
            s=o[col]; c["analise_colunas"].append({"coluna":col,"tipo":"Numérica" if pd.api.types.is_numeric_dtype(s) else "Texto","ausentes":int(s.isnull().sum()),"unicos":int(s.nunique(dropna=True))})
    if p is not None:
        c["previa_preparada"]=p.head(10).round(4).where(pd.notnull(p.head(10)),None).to_dict('records')
        for col in p.columns:
            s=p[col]; num=pd.api.types.is_numeric_dtype(s)
            c["variaveis_disponiveis"].append({"coluna":col,"tipo":"Numérica" if num else "Texto","unicos":int(s.nunique(dropna=True)),"pode_entrada":num,"pode_saida":num,"sugestao":"Pode ser X ou Y." if num else "Não disponível nesta versão."})
    if estado_regressao.get("X_treino") is not None: c["qtd_treino"]=len(estado_regressao["X_treino"])
    if estado_regressao.get("X_teste") is not None: c["qtd_teste"]=len(estado_regressao["X_teste"])
    return c

@app.route('/supervisionado/regressao/resetar', methods=['POST'])
def resetar_regressao_rota():
    resetar_regressao_completa(); return redirect(url_for('regressao'))

@app.route('/supervisionado/regressao', methods=['GET','POST'])
def regressao():
    mensagem=erro=None
    if request.method=='POST':
        acao=request.form.get('acao')
        try:
            if acao=='navegar_etapa':
                destino=int(request.form.get('etapa_destino', estado_regressao.get('etapa_atual',1)))
                maxima=int(estado_regressao.get('etapa_maxima', estado_regressao.get('etapa_atual',1)))
                if destino < 1 or destino > 10: raise ValueError('Etapa inválida.')
                if destino > maxima: raise ValueError('Conclua esta etapa antes de avançar.')
                estado_regressao['etapa_atual']=destino
                mensagem='Etapa aberta para revisão.'
            elif acao=='importar':
                arq=request.files.get('arquivo')
                if not arq or not arq.filename: raise ValueError('Selecione um arquivo para continuar.')
                d=ler_planilha(arq)
                if d.empty: raise ValueError('A base de dados está vazia.')
                estado_regressao['df_original']=d.copy(); estado_regressao['nome_arquivo']=arq.filename
                limpar_regressao_depois(2); estado_regressao['etapa_atual']=2; mensagem='Base importada com sucesso.'
            elif acao=='ir_preparacao':
                if estado_regressao.get('df_original') is None: raise ValueError('Importe uma base antes da preparação.')
                estado_regressao['etapa_atual']=3; mensagem='Revise a preparação e execute a limpeza e normalização.'
            elif acao=='preparar':
                d=estado_regressao.get('df_original')
                if d is None: raise ValueError('Importe uma base antes da preparação.')
                p=d.drop_duplicates().copy()
                for col in p.columns:
                    if p[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(p[col]): p[col]=p[col].fillna(p[col].median() if not pd.isna(p[col].median()) else 0)
                        else:
                            m=p[col].mode(dropna=True); p[col]=p[col].fillna(m.iloc[0] if len(m) else 'Não informado')
                params={}; resumo=[]
                for col in p.columns:
                    if pd.api.types.is_numeric_dtype(p[col]):
                        mn=float(p[col].min()); mx=float(p[col].max()); params[col]={"minimo":mn,"maximo":mx}; resumo.append({"coluna":col,"minimo":round(mn,4),"maximo":round(mx,4)})
                        p[col]=0.0 if mx==mn else (p[col].astype(float)-mn)/(mx-mn)
                estado_regressao.update(df_preparado=p.reset_index(drop=True),normalizacao_parametros=params,normalizacao_resumo=resumo,preparacao_concluida=True,etapa_atual=4)
                limpar_regressao_depois(4); estado_regressao.update(preparacao_concluida=True,etapa_atual=4); mensagem='Limpeza e normalização Min-Max concluídas.'
            elif acao=='selecionar_variaveis':
                d=estado_regressao.get('df_preparado'); xs=request.form.getlist('entradas'); y=request.form.get('saida')
                if d is None: raise ValueError('Prepare a base antes de selecionar as variáveis.')
                if not xs: raise ValueError('Selecione pelo menos uma variável de entrada.')
                if not y: raise ValueError('Selecione a variável de saída.')
                if y in xs: raise ValueError('A saída não pode ser usada também como entrada.')
                if not pd.api.types.is_numeric_dtype(d[y]): raise ValueError('Em regressão, a variável de saída deve ser numérica.')
                for col in xs:
                    if not pd.api.types.is_numeric_dtype(d[col]): raise ValueError('As entradas precisam ser numéricas.')
                estado_regressao.update(entradas=xs,saida=y,variaveis_confirmadas=True,etapa_atual=5); limpar_regressao_depois(5); estado_regressao.update(entradas=xs,saida=y,variaveis_confirmadas=True,etapa_atual=5); mensagem='Variáveis confirmadas.'
            elif acao=='dividir':
                d=estado_regressao.get('df_preparado'); xs=estado_regressao.get('entradas'); y=estado_regressao.get('saida'); pct=int(request.form.get('percentual_treino',80))
                if d is None or not xs or not y: raise ValueError('Confirme as variáveis antes da divisão.')
                if pct<50 or pct>90: raise ValueError('Use um percentual entre 50% e 90%.')
                X=d[xs].astype(float); Y=d[y].astype(float)
                Xt,Xv,yt,yv=train_test_split(X,Y,test_size=(100-pct)/100,random_state=42)
                estado_regressao.update(X_treino=Xt.reset_index(drop=True),X_teste=Xv.reset_index(drop=True),y_treino=yt.reset_index(drop=True),y_teste=yv.reset_index(drop=True),percentual_treino=pct,divisao_concluida=True,etapa_atual=6); limpar_regressao_depois(6); estado_regressao.update(X_treino=Xt.reset_index(drop=True),X_teste=Xv.reset_index(drop=True),y_treino=yt.reset_index(drop=True),y_teste=yv.reset_index(drop=True),percentual_treino=pct,divisao_concluida=True,etapa_atual=6); mensagem=f'Base dividida: {len(Xt)} treino e {len(Xv)} teste.'
            elif acao=='selecionar_algoritmo':
                a=request.form.get('algoritmo')
                if not estado_regressao.get('divisao_concluida'): raise ValueError('Divida a base antes de escolher o algoritmo.')
                if a not in {'linear','ridge','knn','arvore','random_forest'}: raise ValueError('Selecione um algoritmo válido.')
                estado_regressao.update(algoritmo=a,algoritmo_selecionado=True,etapa_atual=7); limpar_regressao_depois(7); estado_regressao.update(algoritmo=a,algoritmo_selecionado=True,etapa_atual=7); mensagem='Algoritmo selecionado: '+nome_algoritmo_regressao(a)
            elif acao=='treinar':
                a=estado_regressao.get('algoritmo'); X=estado_regressao.get('X_treino'); y=estado_regressao.get('y_treino')
                if not a or X is None: raise ValueError('Escolha o algoritmo antes do treinamento.')
                m=criar_modelo_regressao(a,len(X)); m.fit(X,y); estado_regressao.update(modelo=m,modelo_treinado=True,etapa_atual=8); mensagem='Modelo treinado com sucesso.'
            elif acao=='avaliar':
                m=estado_regressao.get('modelo'); X=estado_regressao.get('X_teste'); y=estado_regressao.get('y_teste')
                if m is None or X is None: raise ValueError('Treine o modelo antes da avaliação.')
                pred_norm=np.asarray(m.predict(X), dtype=float)
                y_norm=np.asarray(y, dtype=float)
                # O modelo trabalha com Y normalizado, mas a avaliação é apresentada
                # na unidade original do alvo para que MAE/RMSE e Real × Previsto
                # tenham significado prático para o aluno.
                sy=estado_regressao.get('saida'); py=estado_regressao.get('normalizacao_parametros',{}).get(sy)
                if py:
                    mn=float(py['minimo']); mx=float(py['maximo']); escala=mx-mn
                    if escala == 0:
                        y_real=np.full_like(y_norm, mn, dtype=float); pred_real=np.full_like(pred_norm, mn, dtype=float)
                    else:
                        y_real=y_norm*escala+mn; pred_real=pred_norm*escala+mn
                else:
                    y_real=y_norm; pred_real=pred_norm
                mae=mean_absolute_error(y_real,pred_real); mse=mean_squared_error(y_real,pred_real); rmse=float(np.sqrt(mse)); r2=r2_score(y_real,pred_real)
                res=[{"registro":i+1,"real":round(float(y_real[i]),2),"previsto":round(float(pred_real[i]),2),"erro":round(float(y_real[i]-pred_real[i]),2),"erro_abs":round(abs(float(y_real[i]-pred_real[i])),2)} for i in range(len(y_real))]
                estado_regressao.update(avaliacao={"mae":round(float(mae),2),"mse":round(float(mse),2),"rmse":round(rmse,2),"r2":round(float(r2),4),"total":len(res),"resultados":res,"escala_original":True,"variavel_saida":sy},avaliacao_concluida=True,etapa_atual=8); mensagem='Avaliação concluída na escala original da variável de saída.'
            elif acao=='ir_predicao':
                if not estado_regressao.get('avaliacao_concluida'): raise ValueError('Avalie o modelo antes da predição.')
                estado_regressao['etapa_atual']=9; mensagem='Envie uma nova base para realizar predições.'
            elif acao=='predizer':
                m=estado_regressao.get('modelo'); xs=estado_regressao.get('entradas',[]); params=estado_regressao.get('normalizacao_parametros',{}); arq=request.files.get('arquivo_predicao')
                if m is None: raise ValueError('Treine e avalie o modelo antes da predição.')
                if not arq or not arq.filename: raise ValueError('Selecione uma base para predição.')
                nd=ler_planilha(arq); falt=[x for x in xs if x not in nd.columns]
                if falt: raise ValueError('Faltam as variáveis: '+', '.join(falt))
                dm=nd[xs].apply(pd.to_numeric,errors='coerce')
                if dm.isnull().any().any(): raise ValueError('Há valores ausentes ou não numéricos nas entradas.')
                for col in xs:
                    mn=params[col]['minimo']; mx=params[col]['maximo']; dm[col]=0.0 if mx==mn else (dm[col].astype(float)-mn)/(mx-mn)
                pred=m.predict(dm); out=nd.copy()
                # Reverte a normalização do alvo para facilitar interpretação, quando possível.
                sy=estado_regressao.get('saida'); py=params.get(sy)
                if py: out['Valor_Previsto']=np.round(pred*(py['maximo']-py['minimo'])+py['minimo'], 2)
                else: out['Valor_Previsto']=np.round(pred, 2)
                estado_regressao.update(resultado_predicao_df=out,resultado_predicao=out.head(100).where(pd.notnull(out.head(100)),None).to_dict('records'),predicao_concluida=True,etapa_atual=10); mensagem=f'Predição concluída para {len(out)} registro(s).'
            elif acao=='comparar':
                X=estado_regressao.get('X_treino'); Xv=estado_regressao.get('X_teste'); y=estado_regressao.get('y_treino'); yv=estado_regressao.get('y_teste')
                if X is None: raise ValueError('Divida a base antes da comparação.')
                sy=estado_regressao.get('saida'); py=estado_regressao.get('normalizacao_parametros',{}).get(sy)
                yv_norm=np.asarray(yv,dtype=float)
                if py:
                    mn=float(py['minimo']); mx=float(py['maximo']); escala=mx-mn
                    yv_real=np.full_like(yv_norm,mn,dtype=float) if escala==0 else yv_norm*escala+mn
                else:
                    mn=0.0; escala=1.0; yv_real=yv_norm
                comp=[]
                for a in ['linear','ridge','knn','arvore','random_forest']:
                    m=criar_modelo_regressao(a,len(X)); m.fit(X,y); p_norm=np.asarray(m.predict(Xv),dtype=float)
                    p_real=np.full_like(p_norm,mn,dtype=float) if py and escala==0 else (p_norm*escala+mn if py else p_norm)
                    mse=mean_squared_error(yv_real,p_real)
                    comp.append({"algoritmo":nome_algoritmo_regressao(a),"mae":round(float(mean_absolute_error(yv_real,p_real)),2),"rmse":round(float(np.sqrt(mse)),2),"r2":round(float(r2_score(yv_real,p_real)),4)})
                estado_regressao.update(comparacao=comp,comparacao_concluida=True,etapa_atual=10); mensagem='Comparação concluída na escala original da variável de saída.'

            # Guarda a maior etapa já alcançada. Assim o wizard pode navegar
            # para trás e depois voltar para frente sem refazer etapas já concluídas.
            if acao != 'navegar_etapa':
                atual = int(estado_regressao.get('etapa_atual', 1))
                maxima = int(estado_regressao.get('etapa_maxima', 1))
                estado_regressao['etapa_maxima'] = max(maxima, atual)
        except Exception as exc: erro=str(exc)
    return render_template('regressao.html', **contexto_regressao(mensagem,erro))

@app.route('/supervisionado/regressao/avaliar-modelo')
def avaliar_modelo_regressao():
    a=estado_regressao.get('avaliacao')
    if not a: return jsonify({"sucesso":False,"mensagem":"O modelo ainda não foi avaliado."}),400
    return jsonify({"sucesso":True,**a})

@app.route('/supervisionado/regressao/baixar-predicoes')
def baixar_predicoes_regressao():
    df=estado_regressao.get('resultado_predicao_df')
    if df is None: return jsonify({"erro":"Nenhuma predição disponível."}),400
    b=BytesIO()
    with pd.ExcelWriter(b,engine='openpyxl') as w: df.to_excel(w,index=False,sheet_name='Predicoes')
    b.seek(0); return send_file(b,as_attachment=True,download_name='resultado_regressao.xlsx',mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )