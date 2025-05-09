# SVforensics

Um kit de ferramentas para fonética forense com foco na verificação de locutor, processamento e análise de embeddings de gravações de voz.

O ambiente recomendado para execução do SVforensics é o Google Colab, que oferece uma interface interativa e estruturada para análise forense:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sepai-dev/SVforensics/blob/main/notebooks/svforensics.ipynb)

[![Metodologia e Fundamentação](https://img.shields.io/badge/Metodologia%20e%20Fundamenta%C3%A7%C3%A3o-Google%20Docs-blue?logo=google-docs&logoColor=white)](https://docs.google.com/document/d/1PCvHK_CqQVjBnwv5hcJmPwflBAQdmsR_G-3RdNJufdo/edit?usp=sharing)

O notebook do Colab fornece um ambiente completo e integrado que gerencia a instalação do pacote `svforensics`, download dos recursos necessários, processamento dos áudios do caso e geração dos gráficos de análise.

Para instruções sobre instalação local e uso da interface de linha de comando (CLI), consulte nossa [Documentação de Uso da CLI](docs/cli_usage.md) (em inglês).

## Estrutura do Projeto

```
SVforensics/
├── data/                  # Diretório de dados padrão
│   ├── raw/               # Dados brutos baixados
│   └── processed/         # Saídas de dados processados
├── svforensics/           # Código fonte principal do pacote
│   ├── __init__.py        # Inicializador do pacote, expõe módulos primários
│   ├── __main__.py        # Ponto de entrada principal da CLI (comando `svf`)
│   ├── audioprep.py       # Pré-processamento de áudio (VAD, segmentação)
│   ├── case_embeddings.py # Extração de embeddings para áudio de caso customizado
│   ├── config.py          # Carregamento e gerenciamento de configuração
│   ├── download.py        # Funcionalidade de download de dados
│   ├── embeddings.py      # Carregamento e extração de modelo de embedding principal
│   ├── metadata_embedding_merge.py # Mesclagem de embeddings com CSV de metadados
│   ├── similarity.py      # Funções de cálculo de similaridade/distância
│   ├── testlists.py       # Geração de listas de teste (pares) para verificação
│   ├── verification.py    # Lógica de pontuação e análise de verificação de locutor
│   └── __pycache__/       # Arquivos de cache do Python (geralmente ignorados)
├── tests/                 # Testes unitários e de integração
├── pyproject.toml         # Configuração do pacote
└── README.md              # Este arquivo
```

## Como Funciona

O projeto segue um design limpo onde cada módulo principal serve a dois propósitos:
1.  **Funcionalidade de biblioteca** - Funções que podem ser importadas e usadas programaticamente
2.  **Interface de linha de comando** - Cada módulo também fornece um ponto de entrada CLI

Esta abordagem unificada torna o código mais fácil de manter, ao mesmo tempo que oferece padrões de uso tanto de API quanto de linha de comando.

## Licença

Este projeto foi desenvolvido por Rafaello Virgilli ([@rvirgilli](https://github.com/rvirgilli)) e Lucas Alcântara Souza ([@lucasalcs](https://github.com/lucasalcs)) como parte de suas atribuições oficiais na [Polícia Científica de Goiás](https://www.policiacientifica.go.gov.br/).

Este software está licenciado sob a Licença Pública de Software Brasileira (LPS Brasil) Versão 3.0.

Esta licença garante que o software:
- Pode ser livremente usado, modificado e distribuído
- Deve manter a atribuição aos autores originais e à Polícia Científica de Goiás
- Deve disponibilizar o código fonte a todos os usuários
- Não pode ser incorporado em software proprietário
- Deve preservar as mesmas liberdades em todos os trabalhos derivados

O texto oficial da licença está disponível em Português (LICENSE) com uma tradução para o Inglês fornecida para referência (LICENSE.en).

© 2025 Polícia Científica de Goiás. Todos os direitos reservados. 