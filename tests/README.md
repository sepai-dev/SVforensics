# Estrutura de Testes do SVforensics

Esta pasta contém os testes automatizados para o projeto SVforensics, garantindo a confiabilidade, robustez e consistência da ferramenta.

## Framework Utilizado

O projeto utiliza o **pytest** como framework de testes. Todos os testes devem seguir as convenções do pytest (arquivos nomeados como `test_*.py` e funções de teste como `test_*`).

## Estrutura de Diretórios

-   `tests/`: Diretório raiz para todos os testes.
-   `tests/files/`: Contém arquivos de dados estáticos utilizados como entrada nos testes (ex: áudios de exemplo, modelos, etc.). Isso garante que os testes sejam executados com dados consistentes e representativos.
-   `tests/test_*.py`: Cada arquivo de teste é dedicado a um módulo específico do pacote `svforensics`. Por exemplo, `test_audioprep.py` contém os testes para o módulo `svforensics/audioprep.py`.

## Executando os Testes

Para executar a suíte de testes completa, utilize o seguinte comando a partir da raiz do projeto:

```bash
pytest
```

## Filosofia de Testes

Os testes são projetados para serem **isolados** e **reprodutíveis**. Para isso, fazemos uso extensivo de `fixtures` do pytest para:
-   **Criar um ambiente temporário:** Usamos `fixtures` como `tmpdir` para criar arquivos e diretórios que são automaticamente limpos após a execução dos testes.
-   **Gerar dados de teste:** Dados sintéticos (ex: ondas senoidais) são gerados para testar algoritmos de forma previsível.
-   **Usar dados reais:** Amostras de dados reais (como o `voice_test.ogg`) são utilizadas para validar o comportamento da ferramenta em cenários mais próximos do uso final.
-   **Isolar a configuração:** Uma configuração de projeto temporária (`test_config.json`) é criada e carregada durante os testes para não interferir na configuração de desenvolvimento.

## Testes a Serem Implementados (TODO)

A suíte de testes atual cobre partes essenciais do projeto, mas ainda há módulos importantes que necessitam de cobertura de testes. A implementação dos seguintes testes é prioritária para garantir a qualidade completa do software:

-   [ ] **`test_download.py`**:
    -   Testar o download de arquivos do Google Drive e de URLs diretas.
    -   Verificar o cálculo e a atualização do checksum.
    -   Testar o tratamento de erros (ex: URL inválida, falha no download).

-   [ ] **`test_processing.py` (para `metadata_embedding_merge.py`)**:
    -   Testar o carregamento de embeddings e metadados.
    -   Verificar a fusão correta dos DataFrames.
    -   Validar a filtragem de dados com base em um arquivo de teste.
    -   Testar a função de salvamento do dado processado.

-   [ ] **`test_embeddings.py` e `test_case_embeddings.py`**:
    -   Testar o carregamento do modelo ECAPA2.
    -   Verificar a extração de embeddings de um único arquivo e de um diretório.
    -   Validar a estrutura de extração por locutor (probe/reference).
    -   Testar o tratamento de erros (ex: arquivo de áudio corrompido).

-   [ ] **`test_verification.py` e `test_similarity.py`**:
    -   Testar o cálculo da similaridade de cosseno.
    -   Validar a análise de distribuição de scores (mesmo locutor vs. locutor diferente).
    -   Testar a comparação de embeddings de um caso (probe vs. reference).
    -   Verificar a geração correta dos gráficos de resultado.

-   [ ] **`test_config.py`**:
    -   Testar o carregamento e salvamento de configurações.
    -   Verificar o funcionamento dos fallbacks e valores padrão.
    -   Testar a sobreposição de configurações por variáveis de ambiente. 