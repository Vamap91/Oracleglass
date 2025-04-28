import streamlit as st
import os
import pypdf
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="Oráculo - Sistema de Consulta Inteligente",
    page_icon="🚗",  # Ícone de carro em vez de bola de cristal
    layout="wide"
)

# Variáveis globais
PDF_PATH = "data/Guia Rápido.pdf"
VALIDATION_OK = True
VALIDATION_MESSAGES = {}

# Inicializar variáveis de sessão
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"  # Modelo padrão mais econômico

def validate_environment():
    """Valida todo o ambiente necessário para funcionamento do sistema"""
    global VALIDATION_OK, VALIDATION_MESSAGES
    
    # Resetar estado
    VALIDATION_OK = True
    VALIDATION_MESSAGES = {}
    
    # 1. Validar chave OpenAI - Verificação mais segura
    try:
        import openai
        # Definir API key explicitamente antes de verificar
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        VALIDATION_MESSAGES["openai_key"] = "OK"
    except Exception as e:
        VALIDATION_OK = False
        VALIDATION_MESSAGES["openai_key"] = f"Erro ao verificar API key: {str(e)}"
    
    # 2. Verificar existência do PDF
    if not os.path.exists(PDF_PATH):
        VALIDATION_OK = False
        VALIDATION_MESSAGES["pdf_exists"] = f"Arquivo {PDF_PATH} não encontrado. Verifique o repositório GitHub."
    else:
        VALIDATION_MESSAGES["pdf_exists"] = "OK"
    
    # 3. Validar extração de texto
    if VALIDATION_MESSAGES.get("pdf_exists") == "OK":
        try:
            st.session_state.pdf_text = extract_text_from_pdf(PDF_PATH)
            if not st.session_state.pdf_text or len(st.session_state.pdf_text) < 10:
                VALIDATION_OK = False
                VALIDATION_MESSAGES["text_extraction"] = "Não foi possível extrair texto válido do PDF."
            else:
                VALIDATION_MESSAGES["text_extraction"] = "OK"
        except Exception as e:
            VALIDATION_OK = False
            VALIDATION_MESSAGES["text_extraction"] = f"Erro ao extrair texto: {str(e)}"
    else:
        VALIDATION_OK = False
        VALIDATION_MESSAGES["text_extraction"] = "Não foi possível extrair texto sem o arquivo PDF."
    
    # 4. Verificar dependências
    try:
        import pypdf
        import openai
        VALIDATION_MESSAGES["dependencies"] = "OK"
    except ImportError as e:
        VALIDATION_OK = False
        VALIDATION_MESSAGES["dependencies"] = f"Dependência faltando: {str(e)}"
    
    return VALIDATION_OK

def extract_text_from_pdf(pdf_path):
    """Extrai texto de um arquivo PDF"""
    text_content = ""
    try:
        # Abrir o arquivo PDF
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_content += page_text + "\n"
    except Exception as e:
        st.error(f"Erro ao processar PDF: {str(e)}")
    
    return text_content

def query_ai(query):
    """Processa uma consulta usando a API da OpenAI - Compatível com versão 1.0+"""
    try:
        # Importar OpenAI dentro da função para evitar erros de escopo
        import openai
        
        # Cliente OpenAI para v1.0+
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Limitar o contexto para reduzir custos
        max_length = 3500 if st.session_state.model == "gpt-3.5-turbo" else 7000
        context = st.session_state.pdf_text[:max_length]
        
        # Chamada da API atualizada para v1.0+
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em carros e veículos. Responda perguntas baseado apenas no conteúdo do documento fornecido. Se a informação não estiver no documento, informe isso claramente."},
                {"role": "user", "content": f"Com base no documento a seguir, responda à pergunta: '{query}'\n\nConteúdo do documento:\n{context}"}
            ],
            temperature=0.3,
            max_tokens=800  # Reduzido para economizar tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao processar consulta: {str(e)}")
        return None

# Validar ambiente na inicialização
validate_environment()

# Interface principal
st.title("🚗 Oráculo - Sistema de Consulta Inteligente")

# Barra lateral com validações e configurações
with st.sidebar:
    st.header("⚙️ Validação do Sistema")
    
    # Status de validação da chave OpenAI
    openai_status = "✅" if VALIDATION_MESSAGES.get("openai_key") == "OK" else "❌"
    st.write(f"{openai_status} **Chave OpenAI**: {VALIDATION_MESSAGES.get('openai_key')}")
    
    # Status de validação do PDF
    pdf_status = "✅" if VALIDATION_MESSAGES.get("pdf_exists") == "OK" else "❌"
    st.write(f"{pdf_status} **PDF Carregado**: {VALIDATION_MESSAGES.get('pdf_exists')}")
    
    # Status de extração de texto
    text_status = "✅" if VALIDATION_MESSAGES.get("text_extraction") == "OK" else "❌"
    st.write(f"{text_status} **Texto Extraído**: {VALIDATION_MESSAGES.get('text_extraction')}")
    
    # Status de dependências
    dep_status = "✅" if VALIDATION_MESSAGES.get("dependencies") == "OK" else "❌"
    st.write(f"{dep_status} **Dependências**: {VALIDATION_MESSAGES.get('dependencies')}")
    
    # Informações do PDF
    if VALIDATION_OK:
        st.divider()
        st.subheader("📄 Informações do PDF")
        st.write(f"Arquivo: {os.path.basename(PDF_PATH)}")
        st.write(f"Tamanho do texto: {len(st.session_state.pdf_text)} caracteres")
        
        # Configurações de modelo
        st.divider()
        st.subheader("⚙️ Configurações")
        model = st.selectbox(
            "Modelo OpenAI:",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,  # Default para o modelo mais econômico
            help="Selecione o modelo da OpenAI. GPT-3.5 é mais rápido e econômico, GPT-4 é mais preciso."
        )
        
        # Atualizar o modelo selecionado
        if model != st.session_state.model:
            st.session_state.model = model
            st.info(f"Modelo alterado para {model}")

# Interface principal
if not VALIDATION_OK:
    st.error("⚠️ Há problemas com a configuração do sistema. Verifique os detalhes na barra lateral.")
else:
    st.write("Digite sua pergunta sobre veículos e clique em 'Consultar'.")
    
    # Campo de consulta
    query = st.text_input("❓ Sua pergunta:", key="query_input")
    
    # Botão de consulta
    if st.button("🔍 Consultar", key="query_button") and query:
        with st.spinner("Processando consulta..."):
            answer = query_ai(query)
            
            if answer:
                st.divider()
                st.subheader("📝 Resposta:")
                st.markdown(answer)
                
                # Adicionar ao histórico
                st.session_state.history.append({
                    "query": query,
                    "answer": answer,
                    "model": st.session_state.model
                })
    
    # Histórico de consultas
    if st.session_state.history:
        st.divider()
        st.subheader("📋 Histórico de Consultas")
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Pergunta: {item['query']} ({item['model']})", key=f"hist_{idx}"):
                st.markdown(item["answer"])
