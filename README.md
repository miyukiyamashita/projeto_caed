# Classificação de Complexidade Textual e Mutação de Textos

## 1. Descrição

Este projeto foi desenvolvido como parte de um desafio técnico para a área de Pesquisa de Avaliação, com o objetivo de implementar um sistema de classificação de complexidade textual e geração automática de variações linguísticas.

O sistema classifica textos em quatro níveis de complexidade:

- Ensino Fundamental I
- Ensino Fundamental II
- Ensino Médio
- Ensino Superior

Além da classificação, foram implementados três operadores de mutação textual: troca de palavras, troca de gênero e paráfrase. O projeto também permite avaliar como essas transformações influenciam as previsões do classificador.

A implementação utiliza Python, técnicas de Processamento de Linguagem Natural (PLN) e algoritmos de Aprendizado de Máquina.

## 2. Tecnologias utilizadas

- **Python:** linguagem principal do projeto.
- **scikit-learn:** treinamento, comparação e avaliação dos classificadores.
- **spaCy:** análise morfológica e sintática de textos em português.
- **NLTK e WordNet:** recursos lexicais utilizados na substituição de palavras.
- **TF-IDF:** representação numérica dos textos.
- **Streamlit:** interface para interação com o sistema.
- **pytest:** testes automatizados dos operadores de mutação.

## 3. Estrutura do projeto

```text
projeto_caed/
│
├── dataset/
│   └── dataset_caed/           # Textos organizados por nível
│
├── models/                    # Modelos treinados
│
├── relatorio/                 # Resultados dos experimentos
│   ├── resultados_t4_b10.json
│   ├── resultados_t6.json
│   ├── comparacao_mutacoes_t4_b10.json
│   ├── mutacoes_troca_t4_b10.json
│   └── exemplos_*.json
│
├── src/
│   ├── pre_processamento.py   # Preparação dos dados
│   ├── canonicidade.py        # Análise da estrutura silábica
│   ├── features.py            # Características linguísticas
│   ├── modelo.py              # Treinamento e classificação
│   ├── mutacoes.py            # Troca de palavras
│   ├── troca_genero.py        # Troca de gênero
│   ├── parafrase.py           # Paráfrases
│   ├── avaliar_mutacoes.py    # Avaliação das mutações
│   └── diagnostico.py         # Diagnósticos
│
├── tests/
│   ├── test_mutacoes.py
│   ├── test_parafrase.py
│   └── test_troca_genero.py
│
├── app.py                     # Interface Streamlit
├── requirements.txt           # Dependências
└── README.md                  # Documentação
```

## 4. Instalação

### Requisitos

- Python 3.11 ou superior, preferencialmente 3.12.
- pip.
- Acesso à internet para instalação inicial das dependências e dos modelos linguísticos.

### Criar ambiente virtual

No terminal, dentro da pasta do projeto:

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Instalar o modelo de português do spaCy

```bash
python -m spacy download pt_core_news_sm
```

### Instalar recursos do NLTK

```bash
python -m nltk.downloader wordnet omw-1.4
```

Os recursos lexicais são utilizados pelo operador de substituição de palavras.

## 5. Execução

Todos os comandos devem ser executados a partir da raiz do projeto, com o ambiente virtual ativado.

### Treinamento e avaliação

Para treinar e avaliar os classificadores:

```bash
python -m src.modelo
```

Caso seja necessário indicar explicitamente o diretório dos dados:

```bash
python -m src.modelo --dados dataset/dataset_caed
```

O treinamento compara diferentes modelos e registra métricas de desempenho, incluindo F1-score macro, acurácia, erro ordinal e matriz de confusão.

### Executar a interface

Para iniciar a aplicação Streamlit:

```bash
streamlit run app.py
```

A interface permite utilizar as funcionalidades de classificação e mutação textual disponibilizadas pela aplicação.

### Executar troca de palavras

```bash
python -m src.mutacoes --arquivo caminho/para/texto.txt --prob 0.3 --semente 42
```

### Executar troca de gênero

```bash
python -m src.troca_genero --arquivo caminho/para/texto.txt --prob 1.0 --semente 42
```

Os parâmetros de execução dos demais módulos podem ser consultados nos respectivos arquivos Python.

## 6. Limitações conhecidas

- A classificação pode ser influenciada pelo vocabulário e pelo conteúdo temático dos materiais didáticos.
- A análise de canonicidade depende de regras computacionais de segmentação silábica.
- A substituição lexical possui cobertura limitada de sinônimos em português.
- A troca de gênero não contempla todas as irregularidades morfológicas.
- A paráfrase depende de regras linguísticas e não cobre todas as estruturas possíveis.
- A transformação geral de voz passiva para ativa não foi concluída.
- As mutações podem alterar o significado ou a naturalidade dos textos.
- A comparação das previsões não substitui uma avaliação humana da qualidade textual.

## 7. Considerações finais

O projeto apresenta um protótipo de classificação de complexidade textual e geração de variações linguísticas.

A comparação experimental demonstrou a eficácia do modelo TF-IDF com Regressão Logística na configuração avaliada. Os operadores de mutação oferecem diferentes estratégias de transformação textual, com limitações documentadas e possibilidades de aperfeiçoamento.

Os detalhes metodológicos, os resultados completos e as atividades não concluídas estão descritos no relatório técnico que acompanha o projeto.