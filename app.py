from flask import (
    Flask,
    render_template,
    request,
    Response,
    jsonify,
    send_file
)

import pandas as pd
import numpy as np

from io import BytesIO

import json
import time


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

# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )