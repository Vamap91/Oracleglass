# Oráculo - Sistema de Consulta Multi-PDF

![Oráculo](https://img.shields.io/badge/Oráculo-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red)
![Licença](https://img.shields.io/badge/license-MIT-yellow)

Oráculo é um sistema de análise inteligente que permite carregar, processar e consultar múltiplos documentos PDF usando inteligência artificial. Esta versão simplificada do [Projeto Oráculo original](https://github.com/seu-usuario/oraculo-original) mantém as funcionalidades essenciais de extração de texto e consulta por IA, permitindo acesso eficiente às informações contidas em documentos PDF.

## ✨ Recursos

- 📁 **Upload de múltiplos PDFs**: Carregue e processe vários documentos simultaneamente
- 🔍 **Extração de texto inteligente**: Combina extração direta e OCR para textos em imagens
- 🤖 **Consultas em linguagem natural**: Faça perguntas sobre o conteúdo dos documentos
- 💾 **Persistência de dados**: Salve e restaure o estado do sistema para uso posterior
- 📊 **Histórico de consultas**: Mantenha um registro de todas as interações

## 🚀 Início Rápido

### Executando localmente

1. Clone este repositório:
   ```
   git clone https://github.com/seu-usuario/oraculo.git
   cd oraculo
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Instale as dependências do sistema:
   - **Tesseract OCR** (para OCR)
   - **Poppler** (para processamento de PDF)

4. Execute o aplicativo:
   ```
   streamlit run app.py
   ```

### Implantação no Streamlit Cloud

1. Faça fork deste repositório
2. Acesse [streamlit.io/cloud](https://streamlit.io/cloud)
3. Conecte sua conta GitHub e selecione o repositório
4. Configure os seguintes secrets:
   - `OPENAI_API_KEY`: Sua chave da API OpenAI (opcional, também pode ser inserida na interface)

## 🛠️ Requisitos do Sistema

### Python e Bibliotecas

- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

### Dependências do Sistema

#### Windows:
- [Tesseract OCR para Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- [Poppler para Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

#### macOS:
```
brew install tesseract
brew install poppler
```

#### Linux (Ubuntu/Debian):
```
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-por
sudo apt install poppler-utils
```

## 📖 Como Usar

### Upload de Documentos

1. Navegue até a aba "Upload de Documentos"
2. Arraste e solte um ou mais arquivos PDF
3. Clique em "Processar" para cada arquivo

### Consulta

1. Navegue até a aba "Consultar"
2. Digite sua pergunta na caixa de texto
3. Clique em "Consultar" para obter a resposta

### Gerenciamento de Estado

1. Use "Salvar Estado" para baixar um arquivo com os dados processados
2. Use "Carregar Estado" para restaurar dados salvos anteriormente
3. Use "Resetar Sistema" para limpar todos os dados

## 🧩 Arquitetura

O sistema é composto por três módulos principais:

1. **Módulo de Processamento de PDF**: Extrair texto de PDFs usando PyMuPDF e OCR quando necessário
2. **Módulo de Armazenamento**: Gerenciar dados entre sessões usando pickle e base64
3. **Módulo de IA**: Processar consultas usando a API OpenAI

## 💡 Personalização

### Modelos de IA

Para usar um modelo diferente, modifique o parâmetro `model` na função `query_ai()` dentro de `app.py`:

```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Altere para o modelo desejado
    # outros parâmetros...
)
```

### Idiomas Suportados pelo OCR

Para adicionar suporte a outros idiomas, instale os pacotes de idiomas do Tesseract e modifique o parâmetro `lang`:

```python
text = pytesseract.image_to_string(img, lang='eng+por+spa')
```

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [Streamlit](https://streamlit.io/) por tornar a criação de interfaces web com Python simples
- [OpenAI](https://openai.com/) pelos modelos de linguagem avançados
- [PyMuPDF](https://pymupdf.readthedocs.io/) pelo processamento eficiente de PDF
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) pela capacidade de OCR
