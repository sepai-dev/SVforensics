# Acesso ao Repositório Privado SVforensics

Este documento explica como obter acesso e configurar autenticação para o repositório privado do SVforensics.

## Para Colaboradores Autorizados

### 1. Solicitação de Acesso

Para obter acesso ao repositório privado, entre em contato com a Polícia Científica de Goiás através dos canais oficiais:

- Website: https://www.policiacientifica.go.gov.br/
- Email: [email institucional a ser definido]

Será necessário:
- Assinatura do Termo de Responsabilidade e Acordo de Colaboração
- Aprovação prévia pela Polícia Científica de Goiás
- Comprometimento com as condições da licença

### 2. Configuração de Token de Acesso

Após aprovação, você receberá um token de acesso pessoal (Personal Access Token - PAT) que permite:
- Clonar o repositório privado
- Criar pull requests para propor melhorias
- Acessar os arquivos de configuração

### 3. Usando o Token

#### Para instalação via pip no Google Colab:

```python
# Substitua SEU_TOKEN_AQUI pelo token fornecido
TOKEN = "SEU_TOKEN_AQUI"
!pip install --no-cache-dir git+https://{TOKEN}@github.com/sepai-dev/SVforensics.git -q
```

#### Para clone local:

```bash
# Clone usando o token
git clone https://SEU_TOKEN_AQUI@github.com/sepai-dev/SVforensics.git

# Ou configure o token globalmente
git config --global credential.helper store
git clone https://github.com/sepai-dev/SVforensics.git
# Digite SEU_TOKEN_AQUI quando solicitado a senha
```

#### Para acessar arquivos de configuração:

```python
import requests

TOKEN = "SEU_TOKEN_AQUI"
headers = {"Authorization": f"token {TOKEN}"}

# Exemplo: baixar svforensics.json
url = "https://raw.githubusercontent.com/sepai-dev/SVforensics/main/config/svforensics.json"
response = requests.get(url, headers=headers)
config = response.json()
```

### 4. Segurança do Token

**IMPORTANTE:**
- Nunca compartilhe seu token de acesso
- Não inclua o token em código que seja commitado
- Use variáveis de ambiente sempre que possível
- Se comprometer o token, entre em contato imediatamente para revogação

### 5. Notebook Atualizado para Colaboradores

Uma versão atualizada do notebook do Google Colab está sendo preparada especificamente para colaboradores autorizados, incluindo:
- Campo para inserção segura do token
- Instruções específicas para download de arquivos
- Configuração adequada para ambiente privado

## Para Uso Institucional (Polícia Científica de Goiás)

O acesso interno não requer token, pois os peritos têm acesso direto ao repositório através da conta organizacional.

## Suporte Técnico

Para problemas técnicos com acesso ao repositório, entre em contato com:
- Perito Criminal Rafaello Virgilli (@rvirgilli)
- Perito Criminal Lucas Alcântara Souza (@lucasalcs)

## Responsabilidades

Ao receber acesso ao repositório, o colaborador concorda em:
- Usar o software apenas para fins autorizados conforme a licença
- Não redistribuir o código-fonte ou tokens de acesso
- Submeter melhorias através do processo oficial de pull request
- Reportar qualquer problema de segurança imediatamente 