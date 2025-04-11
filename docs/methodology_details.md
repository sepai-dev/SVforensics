**Fundamentação Metodológica: Comparação Forense de Falantes**

**1. Objetivo e Escopo**

O objetivo central da comparação forense de falantes é a avaliação científica da força da evidência de voz, especificamente no que concerne à questão de origem: determinar em que medida as características vocais e de fala observadas em amostras de origem desconhecida (questionadas) são mais prováveis se elas tiverem sido produzidas pelo mesmo indivíduo que produziu as amostras de origem conhecida (padrão), em comparação com a hipótese de terem sido produzidas por um indivíduo diferente pertencente a uma população relevante [1, 2]. O exame não visa uma identificação categórica, mas sim uma avaliação probabilística da evidência no contexto do caso.

**2. Princípios Fundamentais da Variabilidade da Fala**

A produção de fala humana é um processo complexo influenciado por fatores anatômicos, fisiológicos, neurológicos, linguísticos e sociais, resultando em variabilidade intra-falante (dentro do mesmo indivíduo em diferentes ocasiões) e inter-falante (entre indivíduos diferentes) [3, 4]. As características relevantes para a discriminação forense derivam de:

*   **Morfologia do Aparelho Fonador:** As dimensões e a configuração individual da laringe e do trato vocal supralaríngeo (faringe, cavidades oral e nasal) impõem constrições acústicas que moldam as ressonâncias (formantes) e a qualidade da fonte glotal [4, 5].
*   **Controle Neuromotor e Padrões Articulatórios:** Os padrões idiossincráticos de controle motor para a articulação dos sons da fala (fonemas), incluindo coarticulação, tempo e coordenação, podem ser distintivos [3, 6].
*   **Características Suprassegmentais e Prosódicas:** Padrões habituais de frequência fundamental (f0), intensidade, ritmo e velocidade de elocução, que compõem a prosódia, contribuem para a caracterização do falante [1, 7].
*   **Seleção Linguística e Hábitos de Fala:** A escolha de vocabulário, construções sintáticas, marcadores discursivos, e a influência de fatores dialetais e sociolectais constituem o comportamento linguístico do falante [1, 8].

O exame pericial foca na análise da **similaridade** entre as amostras questionada e padrão em múltiplos níveis, e na avaliação da **tipicidade** (ou poder discriminativo) das características observadas, considerando sua frequência de ocorrência na população relevante [1, 2].

**3. Abordagem Metodológica Integrada**

Adota-se uma metodologia combinada que integra diferentes domínios de análise para uma avaliação robusta da evidência [1, 9]:

*   **3.1. Análise Perceptivo-Auditiva Forense:** Conduzida por peritos com treinamento específico em fonética forense, esta análise envolve a escuta crítica e sistemática para identificar e comparar características auditivamente salientes [1, 8]. Os aspectos avaliados incluem:
    *   **Qualidade Vocal:** Avaliação de parâmetros como altura tonal (pitch), loudness, tipo de fonação (e.g., modal, soprosa, rangente, áspera), ressonância (e.g., nasalidade).
    *   **Características Segmentais:** Realização fonética de fonemas específicos, incluindo variantes alofônicas, padrões de assimilação, elisão ou epêntese.
    *   **Características Suprassegmentais:** Padrões de entonação (contornos de f0), ritmo, pausas, velocidade de elocução e acentuação.
    *   **Fluência:** Presença e padrão de disfluências (hesitações, repetições, falsos começos, prolongamentos).
    *   **Características Linguístico-Idiossincráticas:** Uso de léxico particular, expressões idiomáticas, marcadores discursivos, tiques verbais.

*   **3.2. Análise Acústica:** Utilizando software especializado, esta análise extrai parâmetros físicos mensuráveis do sinal de fala para objetivar e quantificar características vocais [4, 7]. Parâmetros comumente analisados incluem:
    *   **Frequência Fundamental (f0):** Média, desvio padrão, mediana, mínimo, máximo e análise de contornos (micro e macroprosódia).
    *   **Formantes (F1, F2, F3, etc.):** Frequências das principais ressonâncias do trato vocal, particularmente relevantes para a identidade das vogais e características do falante. Análise de trajetórias e espaços vocálicos.
    *   **Medidas Espectrais:** Incluindo inclinação espectral (spectral tilt), cepstrum, LTAS (Long-Term Average Spectrum) para caracterizar a qualidade da fonte glotal e a ressonância geral.
    *   **Medidas Temporais:** Duração de segmentos fonéticos, taxa de fala (speech rate), taxa de articulação (articulation rate), proporção de tempo de fala/silêncio.
    A viabilidade e confiabilidade desta análise dependem criticamente da qualidade do sinal (relação sinal-ruído, ausência de distorções, reverberação) e da quantidade de material de fala disponível [1, 7].

*   **3.3. Análise por Sistema Automático de Reconhecimento de Locutor (ASR/ASV):** Implementa-se um sistema computacional baseado em redes neurais profundas (Deep Neural Networks - DNNs) para extrair representações vetoriais (embeddings) das características do locutor e comparar as amostras [10, 11]. O processo envolve:
    *   **Pré-processamento do Áudio:** Inclui resampling para taxa de amostragem padrão (e.g., 16 kHz), normalização de amplitude e, potencialmente, segmentação baseada em detecção de atividade de voz (VAD). Pode incluir processamento para simular condições específicas do canal, como a codificação Opus utilizada pelo WhatsApp, se relevante para o material do caso.
    *   **Extração de Embeddings:** Utiliza-se um modelo DNN pré-treinado em larga escala, como o **ECAPA2** [12], para mapear segmentos de fala de duração variável (chunks) em vetores de características de dimensão fixa (embeddings) em um espaço latente otimizado para discriminação de locutores.
    *   **Comparação de Embeddings:** Calcula-se a similaridade entre os embeddings extraídos das amostras questionada e padrão usando uma métrica apropriada, tipicamente a **Similaridade do Cosseno** [10]. Isso resulta em um score (ou uma distribuição de scores, se múltiplos segmentos forem comparados).
    *   **Contextualização Estatística e Calibração:** Para interpretar o score obtido, ele é comparado com distribuições de scores calculadas a partir de uma população de referência relevante. Utiliza-se um conjunto de dados amplo, como o subset `test` do **VoxCeleb1** [13] (pré-processado conforme necessário), para gerar distribuições de scores de comparações do mesmo locutor (target) e de locutores diferentes (non-target). Esta contextualização, frequentemente visualizada através de histogramas ou estimativas de densidade de kernel (KDE), permite situar o score do caso e pode servir de base para calibração ou cálculo de Razão de Verossimilhança (Likelihood Ratio - LR), embora a saída direta aqui seja a posição relativa do score [2, 11, 14].

**4. Estrutura da Conclusão Pericial e Avaliação da Evidência**

A integração dos resultados das análises perceptivo-auditiva, acústica (quando viável) e computacional automática forma a base para a conclusão pericial. Reconhecendo a natureza probabilística da evidência de voz e as limitações inerentes [1, 2], a conclusão é expressa utilizando uma escala verbal qualitativa padronizada, alinhada com as diretrizes para avaliação e relato de evidências forenses, como as propostas pela ENFSI (European Network of Forensic Science Institutes) [15].

Esta escala reflete o **grau de suporte** que o conjunto das observações periciais confere à hipótese de origem comum das amostras, em relação à hipótese de origem distinta. A análise automática fornece uma medida quantitativa da similaridade das características extraídas pelo modelo, contextualizada estatisticamente, que é ponderada juntamente com os achados qualitativos e quantitativos das demais análises para determinar o nível apropriado na escala verbal de conclusão. A convergência ou divergência entre os resultados das diferentes abordagens é explicitamente considerada na avaliação final da força da evidência.

---

**5. Referências Bibliográficas**

[1] Rose, P. (2002). *Forensic Speaker Identification*. Taylor & Francis.
[2] Morrison, G. S., Enzinger, E., & Zhang, C. (2018). Forensic speech science. In *Oxford Research Encyclopedia of Linguistics*. Oxford University Press.
[3] Stevens, K. N. (1998). *Acoustic Phonetics*. MIT Press.
[4] Titze, I. R. (1994). *Principles of Voice Production*. Prentice Hall. (Reeditado por National Center for Voice and Speech).
[5] Fant, G. (1960). *Acoustic Theory of Speech Production*. Mouton de Gruyter.
[6] Ladefoged, P., & Johnson, K. (2014). *A Course in Phonetics* (7th ed.). Cengage Learning.
[7] Kent, R. D., & Read, C. (2002). *Acoustic Analysis of Speech* (2nd ed.). Singular Publishing Group.
[8] Jessen, M. (2008). Forensic phonetics. *Language and Linguistics Compass*, 2(4), 671-711.
[9] French, J. P., & Harrison, P. (2007). Practical forensic speaker recognition: An overview. In *Proceedings of the International Workshop on Computational Forensics* (IWCF). Springer.
[10] Kinnunen, T., & Li, H. (2010). An overview of text-independent speaker recognition: From features to supervectors. *Speech Communication*, 52(1), 12-40.
[11] Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). X-vectors: Robust DNN embeddings for speaker recognition. In *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing* (ICASSP).
[12] Thienpondt, J., & Demuynck, K. (2023). ECAPA2: A Hybrid Architecture for Fine-grained Speaker Embedding Extraction. In *Proceedings of the IEEE Automatic Speech Recognition and Understanding Workshop* (ASRU). (arXiv:2401.08342)
[13] Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: a large-scale speaker identification dataset. In *Proceedings of INTERSPEECH*.
[14] Morrison, G. S. (2009). Forensic voice comparison and the likelihood ratio. *Language and Linguistics Compass*, 3(1), 170-191.
[15] ENFSI Guideline for Evaluative Reporting in Forensic Science (2015). European Network of Forensic Science Institutes.

--- 