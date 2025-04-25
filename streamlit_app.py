import streamlit as st
import os
import base64
import pickle
import hashlib
import time
import openai
from io import BytesIO
import pypdf

# Configuração da página
st.set_page_config(
    page_title="Oráculo - Sistema de Consulta Multi-PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variáveis de sessão
if "pdf_contents" not in st.session_state:
    st.session_state.pdf_contents = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "combined_text" not in st.session_state:
    st.session_state.combined_text = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Tentar obter a chave da API do OpenAI
try:
    openai_api_key = st.secrets.get("openai", {}).get("api_key", "")
    if openai_api_key:
        openai.api_key = openai_api_key
except Exception as e:
    openai_api_key = ""

# Funções utilitárias
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def extract_text_from_pdf(pdf_bytes):
    text_content = ""
    try:
        pdf_file = BytesIO(pdf_bytes)
        reader = pypdf.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text_content += f"\n--- Página {page_num+1} ---\n{page_text}"
            progress_bar.progress((page_num + 1) / total_pages)
            status_text.text(f"Processando página {page_num+1}/{total_pages}")
        
        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
    
    return text_content

def process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)
    
    if file_hash in st.session_state.processed_files:
        st.info(f"O arquivo '{uploaded_file.name}' já foi processado.")
        return
    
    with st.spinner(f"Processando '{uploaded_file.name}'..."):
        text_content = extract_text_from_pdf(file_bytes)
        
        st.session_state.pdf_contents[uploaded_file.name] = {
            "text": text_content,
            "hash": file_hash
        }
        
        st.session_state.processed_files.append(file_hash)
        update_combined_text()
        save_state_to_file()
        
        st.success(f"Arquivo '{uploaded_file.name}' processado com sucesso!")

def update_combined_text():
    combined = ""
    for filename, content in st.session_state.pdf_contents.items():
        combined += f"\n\n=== DOCUMENTO: {filename} ===\n\n"
        combined += content["text"]
    
    st.session_state.combined_text = combined

def query_ai(query):
    if not openai_api_key:
        st.error("A chave da API OpenAI não está configurada.")
        return None
    
    if not st.session_state.combined_text:
        st.warning("Nenhum documento foi processado ainda.")
        return None
    
    try:
        text_context = st.session_state.combined_text[:7500]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em analisar documentos e responder perguntas com base no conteúdo fornecido."},
                {"role": "user", "content": f"Com base nos documentos a seguir, responda à pergunta: '{query}'\n\n{text_context}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao processar consulta: {str(e)}")
        return None

def reset_system():
    st.session_state.pdf_contents = {}
    st.session_state.processed_files = []
    st.session_state.combined_text = ""
    save_state_to_file()
    st.success("Sistema resetado com sucesso.")

def save_state_to_file():
    state_data = {
        "pdf_contents": st.session_state.pdf_contents,
        "processed_files": st.session_state.processed_files,
        "combined_text": st.session_state.combined_text
    }
    
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/oraculo_state.dat', 'wb') as f:
            pickle.dump(state_data, f)
    except Exception as e:
        st.error(f"Erro ao salvar estado: {str(e)}")

def load_state_from_file():
    try:
        if os.path.exists('data/oraculo_state.dat'):
            with open('data/oraculo_state.dat', 'rb') as f:
                state_data = pickle.loads(f.read())
                
                st.session_state.pdf_contents = state_data.get("pdf_contents", {})
                st.session_state.processed_files = state_data.get("processed_files", [])
                st.session_state.combined_text = state_data.get("combined_text", "")
                
                return True
        return False
    except Exception as e:
        st.error(f"Erro ao carregar estado: {str(e)}")
        return False

# Carregar estado salvo ao iniciar o aplicativo
load_state_from_file()

# Interface principal
st.title("🔮 Oráculo - Sistema de Consulta Multi-PDF")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Toggle de modo administrador
    st.session_state.is_admin = st.checkbox("Modo Administrador", value=st.session_state.is_admin)
    
    st.divider()
    
    # Estatísticas
    st.subheader("📊 Estatísticas")
    st.write(f"Documentos carregados: {len(st.session_state.pdf_contents)}")
    
    if st.session_state.pdf_contents:
        total_pages = sum(1 for content in st.session_state.pdf_contents.values() 
                         for line in content["text"].split("\n") if line.startswith("--- Página "))
        st.write(f"Total de páginas: {total_pages}")
    
    # Gerenciamento de estado (apenas para admin)
    if st.session_state.is_admin:
        st.divider()
        st.subheader("💾 Gerenciamento de Estado")
        
        if st.button("Salvar Estado"):
            state_data = {
                "pdf_contents": st.session_state.pdf_contents,
                "processed_files": st.session_state.processed_files,
                "combined_text": st.session_state.combined_text
            }
            
            serialized = pickle.dumps(state_data)
            b64_data = base64.b64encode(serialized).decode()
            
            st.download_button(
                label="Baixar Estado do Sistema",
                data=b64_data,
                file_name=f"oraculo_state_{int(time.time())}.dat",
                mime="application/octet-stream"
            )
        
        if st.button("Resetar Sistema"):
            reset_system()

# Função para o botão de restaurar estado
def show_restore_state():
    st.subheader("Restaurar Estado")
    st.write("Envie um arquivo .dat para restaurar um estado salvo anteriormente.")
    
    # Usamos uma página separada para o upload de estado para evitar conflitos
    dat_file = st.file_uploader("Arquivo de Estado", type=["dat"], key="unique_dat_key")
    
    if dat_file is not None:
        if st.button("Restaurar Estado"):
            try:
                b64_data = dat_file.read()
                serialized = base64.b64decode(b64_data)
                state_data = pickle.loads(serialized)
                
                st.session_state.pdf_contents = state_data["pdf_contents"]
                st.session_state.processed_files = state_data["processed_files"]
                st.session_state.combined_text = state_data["combined_text"]
                
                save_state_to_file()
                st.success("Estado restaurado com sucesso!")
                st.session_state.page = "admin"
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Erro ao carregar estado: {str(e)}")

# Definir páginas para navegação (abordagem de página única)
if "page" not in st.session_state:
    st.session_state.page = "admin" if st.session_state.is_admin else "user"

# Botões de navegação para administradores
if st.session_state.is_admin:
    # Botões para navegação de páginas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Gerenciar PDFs", use_container_width=True):
            st.session_state.page = "admin"
    
    with col2:
        if st.button("💾 Restaurar Estado", use_container_width=True):
            st.session_state.page = "restore"
    
    with col3:
        if st.button("🔍 Consultar", use_container_width=True):
            st.session_state.page = "user"

# Renderizar a página correta
if st.session_state.page == "admin" and st.session_state.is_admin:
    st.header("🔧 Interface do Administrador")
    st.write("Neste modo, você pode gerenciar os documentos disponíveis para consulta.")
    
    # Upload de PDFs - COM CHAVE ÚNICA
    st.subheader("📁 Upload de Documentos PDF")
    st.write("Carregue um ou mais arquivos PDF para processamento.")
    
    pdf_files = st.file_uploader(
        "Escolha os arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload_only"  # Chave única para este uploader
    )
    
    # Processar PDFs
    if pdf_files:
        for i, pdf_file in enumerate(pdf_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Arquivo: {pdf_file.name}")
            with col2:
                if st.button(f"Processar", key=f"btn_{i}"):
                    process_pdf(pdf_file)
    
    # Documentos processados
    if st.session_state.pdf_contents:
        st.subheader("📚 Documentos Processados")
        
        for idx, (filename, content) in enumerate(st.session_state.pdf_contents.items()):
            with st.expander(f"{idx+1}. {filename}"):
                st.text_area(
                    "Amostra do texto extraído",
                    content["text"][:1000] + "..." if len(content["text"]) > 1000 else content["text"],
                    height=200,
                    key=f"preview_{idx}"
                )

elif st.session_state.page == "restore" and st.session_state.is_admin:
    # Página de restauração de estado
    show_restore_state()

else:
    # Modo usuário padrão
    st.header("🔍 Consultar Documentos")
    
    if not st.session_state.pdf_contents:
        st.warning("Nenhum documento disponível. Por favor, aguarde até que um administrador adicione documentos ao sistema.")
    else:
        st.write("Digite sua pergunta para consultar os documentos disponíveis.")
        
        # Campo de consulta com chave única
        query = st.text_input("❓ Sua pergunta:", key="unique_query")
        
        if st.button("🔎 Consultar", key="unique_query_btn") and query:
            with st.spinner("Processando consulta..."):
                answer = query_ai(query)
                
                if answer:
                    st.divider()
                    st.subheader("📝 Resposta:")
                    st.write(answer)
                    
                    st.session_state.history.append({
                        "query": query,
                        "answer": answer,
                        "timestamp": time.strftime("%d/%m/%Y %H:%M:%S")
                    })
        
        # Histórico
        if st.session_state.history:
            st.divider()
            st.subheader("📋 Histórico de Consultas")
            
            for idx, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"{item['timestamp']} - {item['query']}", key=f"hist_{idx}"):
                    st.write(item['answer'])
