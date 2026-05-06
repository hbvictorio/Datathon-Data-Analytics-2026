Datathon Passos Mágicos - FIAP PosTech 2024

Este repositório contém a solução completa para o desafio Datathon da Associação Passos Mágicos, desenvolvido como parte da pós-graduação FIAP PosTech.
🌟 Acesse a Aplicação (Sistema de Alerta Precoce)
A aplicação interativa desenvolvida para a equipe pedagógica da Passos Mágicos já está no ar! Você pode testar o modelo preditivo e explorar os dashboards através do link abaixo:
👉 Acessar o Streamlit Cloud - Datathon Passos Mágicos
---
🎯 Objetivo do Projeto
O objetivo deste projeto é analisar a base de dados do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) referente aos anos de 2020 a 2024, extrair insights acionáveis e desenvolver um modelo preditivo para identificar alunos em risco de defasagem educacional.
📁 Estrutura do Repositório (Bloco 1)
```text
datathon_passos/
├── data/                   # Datasets originais e processados
│   └── dataset_completo_2020_2024.csv  # Dados longitudinais limpos
├── notebooks/              # Scripts de análise e modelagem
│   ├── 01_eda_limpeza.py       # Limpeza e Análise Exploratória (EDA)
│   └── 02_modelo_preditivo.py  # Feature Engineering e Modelagem (Forecasting)
├── models/                 # Modelos treinados e artefatos
│   ├── modelo_risco.pkl        # Modelo Random Forest treinado
│   ├── scaler.pkl              # Scaler para normalização
│   ├── features.txt            # Lista de features utilizadas
│   ├── threshold.txt           # Threshold otimizado
│   └── metadata.json           # Metadados e métricas do modelo
├── app/                    # Aplicação Streamlit
│   └── app.py                  # Código fonte da aplicação interativa
├── figures/                # Gráficos gerados na EDA e Modelagem
│   ├── p1_ian_defasagem.png    # Gráficos das 10 perguntas obrigatórias
│   ├── ...
│   └── modelo_comparacao.png   # Gráficos de avaliação do modelo
├── docs/                   # Documentação adicional
│   ├── eda_findings.txt        # Resumo dos achados da EDA
│   ├── slide_content.md        # Conteúdo da apresentação gerencial
│   └── roteiro_video.md        # Roteiro para o pitch de 5 minutos
├── presentation/           # Apresentação Gerencial em HTML
│   ├── capa.html               # Slides da apresentação
│   └── ...
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```
📊 Análise Exploratória (Bloco 2)
A análise exploratória respondeu a 10 perguntas cruciais sobre o desenvolvimento dos alunos:
Defasagem (IAN): Queda de 13.6% no IAN médio entre 2020 e 2024.
Desempenho (IDA): Queda acentuada em 2021 (pandemia), com recuperação parcial em 2022.
Engajamento (IEG): Forte correlação com desempenho e ponto de virada. É o motor do programa.
Autoavaliação (IAA): Viés positivo consistente (+2.7 pontos). Alunos superestimam seu desempenho.
Psicossocial (IPS): Isoladamente, não prediz quedas de desempenho.
Psicopedagógico (IPP): Mede dimensões distintas da adequação ao nível (IAN).
Ponto de Virada (IPV): Preditores mudaram de IPP (2020) para IEG e IDA (2021-2022).
Pilares do INDE: IDA (r=0.81) e IEG (r=0.80) são os maiores preditores do índice global.
Alunos em Risco: Perfil claro de baixo IDA (3.4) e baixo IEG (4.3).
Impacto das Fases: Retenção é o padrão dominante (50.5% mantiveram a mesma pedra).
🤖 Modelagem Preditiva: Forecasting T→T+1 (Bloco 3)
Desenvolvemos um modelo de Forecasting Verdadeiro (Random Forest) para prever o risco de um aluno cair para a classificação Quartzo (INDE < 5.506) no ano seguinte, com base nos indicadores do ano atual.
Abordagem: Treino com dados de 2020 a 2023, Teste com dados futuros não vistos de 2023→2024. Isso elimina o data leakage e cria um Sistema de Alerta Precoce real.
Features: 18 variáveis (indicadores base + feature engineering avançado).
Otimizações: SMOTE para balanceamento de classes e threshold otimizado para maximizar o Recall.
Métricas (Teste): AUC-ROC: 63.5% | Recall: 58.3% | Accuracy: 69.1%
Impacto: Permite identificação precoce e intervenção proativa antes da queda no desempenho ocorrer.
💻 Aplicação Streamlit (Bloco 4)
A aplicação foi desenvolvida com foco na usabilidade pela equipe da Passos Mágicos:
Interface Acolhedora: Design limpo, cores institucionais e mensagens claras.
Predição Individual: Formulário interativo para avaliar o risco de um aluno no próximo ano letivo.
Análise Exploratória: Dashboards interativos com a evolução dos indicadores.
Deploy: Configurada para deploy fácil no Streamlit Community Cloud (`requirements.txt` incluso).
📈 Apresentação e Pitch (Bloco 5)
Apresentação Gerencial: 12 slides focados em storytelling de dados, destacando o problema (defasagem), a solução (modelo de forecasting) e recomendações estratégicas.
Roteiro de Vídeo: Script detalhado para um pitch de 5 minutos, estruturado para engajar a diretoria e a banca avaliadora.
🚀 Como Executar o Projeto Localmente
Pré-requisitos
```bash
pip install -r requirements.txt
```
Executar a Aplicação Streamlit
```bash
streamlit run app/app.py
```
Executar os Scripts de Análise
```bash
python notebooks/01_eda_limpeza.py
python notebooks/02_modelo_preditivo.py
```
---
Desenvolvido com dedicação para transformar dados em oportunidades.
