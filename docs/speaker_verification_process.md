# Documentação Técnica: Processo de Comparação de Falantes no SVforensics

**1. Visão Geral**

O toolkit `SVforensics` fornece um fluxo de trabalho para realizar a comparação forense de falantes (Speaker Verification). O objetivo principal é **fornecer uma medida da força da evidência em apoio à hipótese de que as amostras de áudio (uma 'questionada' ou 'probe' e uma 'de referência' ou 'reference') vieram do mesmo locutor, em comparação com a hipótese de que vieram de locutores diferentes.** Inicialmente, o foco está na análise de áudios provenientes do aplicativo WhatsApp.

Este processo utiliza modelos de deep learning para extrair representações vetoriais (embeddings) das características da voz e, em seguida, compara esses embeddings usando uma métrica de similaridade (cosseno). Para contextualizar a pontuação obtida no caso específico, o sistema calcula e visualiza distribuições de scores de uma população de referência maior, permitindo uma avaliação mais informada da força da evidência.

O toolkit é licenciado sob a Licença Pública de Software Brasileiro (LPS Brasil) v3.0 (ver `LICENSE`).

**2. Pré-requisitos e Configuração**

*   **Instalação:** O pacote é instalado via `pip` diretamente do repositório GitHub.
*   **Dependências:** A instalação via `pip` gerencia as dependências Python necessárias, incluindo `torch`, `torchaudio`, `pandas`, `numpy`, `huggingface_hub`, `requests`, `soundfile`, `librosa`, `tqdm`, `matplotlib`, `seaborn`, etc.
*   **Configuração:** O comportamento do toolkit é controlado por arquivos JSON no diretório `config/`:
    *   `svforensics.json`: Define caminhos padrão para dados e resultados, parâmetros de processamento de áudio (taxa de amostragem, duração de chunk), detalhes do modelo de embedding (repositório, nome do arquivo), e configurações de UI. Gerenciado por `config.py`.
    *   `download_info.json`: Lista os arquivos a serem baixados (modelo, dados de referência) com suas URLs e checksums (opcional). Utilizado por `download.py`.
    *   `plot_config.json`: Define parâmetros de visualização para os gráficos de resultados (tamanho da figura, cores, fontes, textos localizados). Utilizado por `verification.py` e `config.py`.
*   **Modelo de Embedding:** Um modelo pré-treinado de extração de embeddings (o modelo padrão é ECAPA2, especificado em `svforensics.json`) é necessário. Ele é baixado do Hugging Face Hub (via `huggingface_hub.hf_hub_download`) para um diretório de cache local (definido em `paths_config["models_cache_dir"]`).

**3. Processamento da População de Referência (Contextualização)**

Para que a pontuação de similaridade de um caso específico tenha significado forense, ela precisa ser comparada com distribuições de scores obtidas de uma população relevante.

*   **3.1. Aquisição dos Dados:**
    *   Dados de uma população de referência, especificamente o subset `test` do dataset VoxCeleb1 ([Nagrani et al., 2017](https://arxiv.org/abs/1706.08612), [Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)), são baixados. Este subset `test` contém 4.874 enunciados (utterances) de 40 locutores distintos, extraídos de 677 vídeos.
    *   **Importante:** Para adequar a análise ao contexto forense comum de áudios de WhatsApp, os arquivos de áudio originais (WAV, mono) desta população de referência foram pré-processados *antes* da extração dos embeddings. Esse pré-processamento simulou o ciclo de compressão e descompressão típico do WhatsApp. Especificamente, cada arquivo WAV foi primeiro convertido para o formato Ogg usando o codec `libopus` através do `ffmpeg`, com as seguintes configurações: taxa de amostragem de 16 kHz, bitrate de 18 kbps, canal mono, nível de compressão 10 e preset de aplicação 'voip'. Em seguida, este arquivo Ogg/Opus foi convertido de volta para o formato WAV (16 kHz, 16-bit, mono) usando `pydub`. Os embeddings foram então extraídos a partir destes arquivos WAV resultantes, que carregam os artefatos da compressão Opus. É por isso que os arquivos de embeddings e metadados pré-calculados referenciam "whatsapp" em seus nomes (ex: `vox1_test_whatsapp_ecapa2.pth`, `vox1_meta.csv`).
    *   O resultado desse pré-processamento e extração de embeddings é um arquivo `.pth` contendo os embeddings adaptados ao domínio do WhatsApp e um arquivo de metadados `.csv` associando esses embeddings aos IDs dos locutores e outras informações (como gênero).
    *   Este download dos arquivos pré-calculados é realizado usando a lógica de `download.py`, baseada nas informações em `download_info.json`.
*   **3.2. Merge de Embeddings e Metadados:**
    *   Os embeddings brutos (`.pth`) e os metadados (`.csv`) da população de referência são carregados e combinados.
    *   O módulo `metadata_embedding_merge.py` realiza essa tarefa, associando cada embedding ao ID do locutor, gênero, etc., e salvando o resultado em um único arquivo `.pth` processado (ex: `files/generated/metadata_embeddings/processed_embeddings.pth`). Este arquivo contém tanto os embeddings quanto os metadados relevantes para facilitar o próximo passo.
*   **3.3. Geração da Lista de Teste:**
    *   A partir dos dados mesclados da população, uma lista de teste é gerada para calcular as distribuições de scores de referência.
    *   O módulo `testlists.py` realiza esta etapa. Ele:
        *   Filtra a população pelo gênero especificado pelo usuário (parâmetro essencial para criar distribuições relevantes).
        *   Divide os arquivos de cada locutor selecionado em conjuntos de "referência" e "probe" (dentro da população).
        *   Gera pares de comparação:
            *   **Positivos (Mesmo Locutor):** Compara arquivos do *mesmo* locutor (opcionalmente, exigindo que sejam de vídeos diferentes - `different_videos=True` na config).
            *   **Negativos (Locutor Diferente):** Compara arquivos de locutores *diferentes*.
        *   Controla o número de pares positivos (`n_pos`) e negativos (`n_neg`) por arquivo de referência.
        *   Salva a lista de pares (label, arquivo1, arquivo2) em um arquivo `.txt` (ex: `files/generated/testlists/test_list_gender_m_diff_videos.txt`).

**4. Processamento do Material do Caso**

Esta etapa lida com os áudios específicos da investigação.

*   **4.1. Entrada do Usuário:**
    *   O usuário fornece os arquivos de áudio questionados (probe) e de referência (reference) do caso.
    *   Os arquivos são salvos em diretórios específicos (ex: `audios_caso_questionados`, `audios_caso_referencia`).
*   **4.2. Preparação do Áudio (VAD e Segmentação):**
    *   Os áudios brutos do caso (`.ogg`, `.wav`, `.mp3`, etc.) são processados para prepará-los para a extração de embeddings.
    *   O módulo `audioprep.py` realiza as seguintes sub-etapas:
        *   **Carregamento e Resampling:** Carrega cada arquivo de áudio, converte para mono (se necessário) e reamostra para a taxa de amostragem definida na configuração (ex: 16000 Hz).
        *   **Detecção de Atividade de Voz (VAD - Implícito):** Embora não explicitamente detalhado na chamada da API usada, um passo de VAD é tipicamente necessário antes da segmentação para focar nos trechos de fala, descartando silêncio ou ruído puro. A qualidade do VAD impacta diretamente os resultados. *(Nota: A implementação atual em `audioprep.py` parece focar mais na segmentação de tamanho fixo, mas um VAD prévio seria ideal)*.
        *   **Segmentação (Chunking):** Divide os trechos de fala (ou o áudio todo) em segmentos (chunks) de duração fixa (ex: 4.0 segundos, definido em `config.audio.chunk_duration`), com possível sobreposição ou fades ( `config.audio.fade_duration`). Chunks muito curtos podem ser descartados (`config.audio.min_chunk_duration`).
        *   **Salvamento:** Salva cada chunk como um arquivo `.wav` individual em diretórios de saída estruturados (ex: `files/generated/case/audio_chunks/probe/` e `files/generated/case/audio_chunks/reference/`).
*   **4.3. Extração de Embeddings do Caso:**
    *   Os chunks `.wav` gerados na etapa anterior são processados pelo modelo de embedding pré-treinado.
    *   O módulo `case_embeddings.py` (ou `embeddings.py`, que contém a classe `EmbeddingExtractor`) realiza esta etapa:
        *   Instancia o `EmbeddingExtractor`, carregando o modelo (ECAPA2) do cache.
        *   Itera sobre cada arquivo `.wav` nos diretórios de chunks `probe` e `reference` do caso.
        *   Para cada chunk, a função `extract_embedding` passa o áudio pelo modelo, que produz um vetor de embedding de dimensão fixa (ex: 192 ou 256 dimensões) que representa as características do locutor naquele segmento.
        *   Os embeddings resultantes são coletados e salvos em arquivos `.pt` separados para probe e reference (ex: `files/generated/case/embeddings/probe_embeddings.pt`, `reference_embeddings.pt`). A estrutura salva geralmente é um dicionário mapeando um ID lógico do locutor (derivado do nome do diretório de entrada original) para outro dicionário que mapeia o caminho do chunk ao seu tensor de embedding.

**5. Comparação, Análise e Visualização**

Esta é a etapa final onde os dados são comparados e os resultados apresentados.

*   **5.1. Cálculo dos Scores:**
    *   O módulo `verification.py` orquestra esta fase.
    *   **Scores da População:** Se a lista de teste da população (`.txt`) e os embeddings processados da população (`.pth`) foram gerados, a função `calculate_test_scores` é chamada. Ela itera sobre os pares na lista de teste, carrega os embeddings correspondentes do arquivo `.pth` da população, e calcula a similaridade do cosseno (`similarity.cosine_similarity` ou `verification.cosine_similarity`) para cada par. Os scores são armazenados junto com seus labels (0 para diferente, 1 para mesmo locutor).
    *   **Score(s) do Caso:** A função `compare_case_embeddings` é chamada. Ela carrega os embeddings dos arquivos `.pt` de probe e reference do caso. Em seguida, calcula a similaridade do cosseno entre *todos* os embeddings questionados e *todos* os embeddings de referência do caso. Isso gera uma distribuição de scores para o caso.
*   **5.2. Análise Estatística:**
    *   **População:** A função `analyze_scores_distribution` calcula estatísticas descritivas (média, desvio padrão, mínimo, máximo, mediana) separadamente para os scores de mesmo locutor e locutor diferente da população de referência.
    *   **Caso:** A função `compare_case_embeddings` calcula as mesmas estatísticas para a distribuição de scores obtida na comparação do caso. A *média* desses scores é frequentemente usada como a "pontuação do caso" representativa.
*   **5.3. Visualização:**
    *   A função `plot_results` gera um gráfico (usando `matplotlib` e `seaborn`):
        *   Plota histogramas (ou KDEs) sobrepostos das distribuições de scores de mesmo locutor (ex: verde) e locutor diferente (ex: vermelho) da população de referência.
        *   Plota uma linha vertical representando a pontuação média do caso (ex: azul).
        *   Utiliza configurações de aparência (cores, tamanho, fontes, etc.) definidas em `plot_config.json`.
        *   Inclui legendas e títulos, potencialmente localizados com base na configuração de idioma.
        *   Salva o gráfico em um arquivo de imagem (ex: `files/plots/case_analysis.png`).
*   **5.4. Interpretação:**
    *   A função `interpret_results` fornece uma interpretação textual básica comparando a pontuação média do caso com as distribuições de referência (média e desvio padrão). Ela indica se a pontuação do caso está mais próxima da distribuição de mesmo locutor ou de locutor diferente, oferecendo uma sugestão sobre a força da evidência. *(Nota: Esta interpretação é simplificada e não substitui uma avaliação forense completa, que pode envolver Likelihood Ratios)*.

**6. Tecnologias Subjacentes**

*   **Modelo de Embedding:** Utiliza um modelo de Deep Learning pré-treinado para extrair características discriminativas da voz relacionadas à identidade do locutor. O modelo padrão configurado e referenciado no código é o **ECAPA2** ([Thienpondt & Demuynck, ASRU 2023](https://arxiv.org/abs/2401.08342)), uma arquitetura híbrida desenvolvida para gerar embeddings robustos. A arquitetura do `SVforensics` permite a substituição por outros modelos compatíveis.
*   **Métrica de Similaridade:** Similaridade do Cosseno é usada para medir a semelhança entre os vetores de embedding no espaço de características. Valores mais próximos de 1 indicam maior similaridade.

**7. Referências Cruzadas**

*   **Código:** `audioprep.py`, `case_embeddings.py`, `config.py`, `download.py`, `embeddings.py`, `metadata_embedding_merge.py`, `similarity.py`, `testlists.py`, `verification.py`, `__main__.py`. Código de teste em `tests/`.
*   **Arquivos:** `config/svforensics.json`, `config/download_info.json`, `config/plot_config.json`, `files/downloads/vox1_...`, `files/generated/...`, `README.md`, `LICENSE`.

**8. Nota**

*   **Nota:** O processo descrito acima é uma visão geral do que o `SVforensics` pode fazer. O processamento atual da população de referência é específico para simular áudios do WhatsApp, conforme descrito na Seção 3.1. Futuras versões poderão incluir populações de referência para outros contextos. Para uma implementação mais detalhada, consulte o código-fonte do toolkit, que inclui várias outras funcionalidades e detalhes de implementação. 

**9. Referências Bibliográficas**

*   Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: a large-scale speaker identification dataset. *INTERSPEECH*. ([arXiv:1706.08612](https://arxiv.org/abs/1706.08612), [Dataset Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html))
    ```bibtex
    @InProceedings{Nagrani17,
        author       = "Nagrani, A. and Chung, J.~S. and Zisserman, A.",
        title        = "VoxCeleb: a large-scale speaker identification dataset",
        booktitle    = "INTERSPEECH",
        year         = "2017",
    }
    ```
*   Thienpondt, J., & Demuynck, K. (2023). ECAPA2: A Hybrid Architecture for Fine-grained Speaker Embedding Extraction. *ASRU 2023 - IEEE Automatic Speech Recognition and Understanding Workshop*. ([arXiv:2401.08342](https://arxiv.org/abs/2401.08342))
    ```bibtex
    @inproceedings{thienpondt2023ecapa2,
        title={ECAPA2: A Hybrid Architecture for Fine-grained Speaker Embedding Extraction},
        author={Thienpondt, Jenthe and Demuynck, Kris},
        booktitle={ASRU 2023 - IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
        pages={836--840},
        year={2023},
        organization={IEEE}
    }
    ``` 