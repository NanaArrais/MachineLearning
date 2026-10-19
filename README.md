# Sistema ML — Classificação binária e multiclasse

Versão revisada do projeto Flask para aulas de Machine Learning. A classificação agora aceita alvos com **2 a 30 classes**, incluindo rótulos em texto (por exemplo, **Alta, Média e Baixa**), e entradas numéricas ou categóricas.

## Executar no Mac (VS Code)

1. Extraia o ZIP e abra a pasta `Sistema_ML_Corrigido` no VS Code.
2. No terminal, estando dentro dessa pasta, execute:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m flask --app app run --port 5001
```

3. Abra **http://127.0.0.1:5001/home**.

Se você já tem um `.venv` configurado com as dependências, basta ativá-lo e executar o último comando. Use `Ctrl+C` para parar o servidor.

> O pacote não inclui `.venv`. O treinamento de imagens depende de TensorFlow e pode exigir uma instalação compatível com a versão do Python e o sistema operacional.

## Testar a triagem hospitalar

Os arquivos de exemplo estão em `exemplos/`.

1. Acesse **Aprendizado Supervisionado → Classificação**.
2. Importe `Hospital_Treinamento_3_Niveis.xlsx`.
3. Execute a preparação. A remoção de duplicados acontece nessa etapa; o tratamento de entradas ausentes, a codificação de categorias e a normalização serão ajustados **somente com o conjunto de treino** na divisão.
4. Em **Seleção das variáveis**, escolha como **X**: `Idade`, `Temperatura_C`, `Frequencia_Cardiaca_BPM`, `Pressao_Sistolica_mmHg`, `Pressao_Diastolica_mmHg`, `Nivel_Dor_0_10`, `Falta_de_Ar`, `Dor_no_Peito`, `Sangramento`, `Historico_Relevante`. Marque `Nivel_Prioridade` como **Y**. Não selecione `ID_Atendimento`.
5. Divida em **80% treino / 20% teste**, escolha um algoritmo (por exemplo, **Random Forest**) e execute o treinamento.
6. Na avaliação, veja **acurácia**, **precisão**, **recall**, **F1-Score**, **support** por classe e **matriz de confusão 3×3**. Em problemas multiclasse, precisão, recall e F1 representam a **média macro** entre as classes; em problemas binários, referem-se à classe codificada como 1.
7. Em predição, importe `Hospital_Pacientes_Novos_Predicao.xlsx` e exporte o resultado, que contém `Previsao` (código interno) e `Classe_Prevista` (nome da classe).

### Outras opções

- `Hospital_Treinamento_Binario.xlsx`: classificação com duas classes, variável Y = `Prioridade_Alta` (0 ou 1). Mantém a matriz binária e suas métricas usuais.
- Módulos de regressão, rede neural e classificação de imagens foram preservados conforme os arquivos da versão enviada. Esta revisão foi direcionada ao módulo de classificação tabular.

## Mudanças principais

- `Nivel_Prioridade` passa a ser elegível como variável de saída.
- Entradas categóricas (`Falta_de_Ar`, `Dor_no_Peito`, `Sangramento`, `Historico_Relevante`) são codificadas com **OneHotEncoder**.
- Divisão treino/teste estratificada; transformações aprendidas somente com os registros de treino para evitar vazamento de dados.
- A avaliação agora exibe a matriz de confusão com linhas **reais** e colunas **previstas**, independentemente do número de classes.
- Precisão, recall e F1 macro em classificações multiclasse; indicadores binários preservados.
- Predição reaproveita o mesmo pré-processador e aceita categorias novas, que são ignoradas pelo OneHotEncoder quando desconhecidas.
- Corrigido o contexto do formulário que não recebia `variaveis_disponiveis`.

## Validação

Foram executados testes funcionais do backend com a base hospitalar de 450 registros: seleção X/Y, separação 80/20, treino (Logística, KNN, Árvore, Random Forest), avaliação 3×3, comparação, predição, categoria não vista e compatibilidade com o fluxo binário. O template Jinja e o JavaScript foram validados sintaticamente. **O servidor Flask real não pôde ser inicializado neste ambiente de teste, pois Flask não está instalado; valide a execução via VS Code no seu Mac.**

**Uso exclusivo educacional:** os dados são simulados, não são regras médicas validadas e o sistema não deve ser usado para decisões clínicas reais.
