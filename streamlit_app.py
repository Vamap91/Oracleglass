import streamlit as st
import os
import tempfile
import base64
import pickle
import hashlib
import time
import openai
from io import BytesIO
import pypdf  # Biblioteca mais simples para PDFs

# Configuração da página
st.set_page_config(
    page_title="Oráculo - Sistema de Consulta Multi-PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variáveis de sessão
if "pdf_contents" not in st.session_state:
    st.session_state.pdf_contents = {}  # Dicionário para armazenar o conteúdo de cada PDF
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # Lista para rastrear arquivos já processados
if "combined_text" not in st.session_state:
    st.session_state.combined_text = ""  # Texto combinado de todos os PDFs
if "history" not in st.session_state:
    st.session_state.history = []  # Histórico de consultas
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False  # Modo do usuário (admin ou comum)

# Obter a chave da API do OpenAI das secrets do Streamlit
try:
    openai_api_key = st.secrets.get("openai", {}).get("api_key", "")
    if openai_api_key:
        openai.api_key = openai_api_key
except Exception as e:
    st.sidebar.warning("Chave da API OpenAI não configurada. Alguns recursos podem não funcionar.")
    openai_api_key = ""

# Funções utilitárias
def get_file_hash(file_content):
    """Gera um hash único para o conteúdo do arquivo."""
    return hashlib.md5(file_content).hexdigest()

def extract_text_from_pdf(pdf_bytes):
    """Extrai texto de um PDF usando pypdf, uma biblioteca mais simples."""
    text_content = ""
    try:
        # Criar um arquivo temporário em memória
        pdf_file = BytesIO(pdf_bytes)
        
        # Usar pypdf para extrair texto
        reader = pypdf.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num, page in enumerate(reader.pages):
            # Extrair texto da página
            page_text = page.extract_text() or ""
            
            # Adicionar ao conteúdo total
            text_content += f"\n--- Página {page_num+1} ---\n{page_text}"
            
            # Atualizar progresso
            progress_bar.progress((page_num + 1) / total_pages)
            status_text.text(f"Processando página {page_num+1}/{total_pages}")
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
    
    return text_content

def process_pdf(uploaded_file):
    """Processa um arquivo PDF carregado."""
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)
    
    # Verificar se o arquivo já foi processado
    if file_hash in st.session_state.processed_files:
        st.info(f"O arquivo '{uploaded_file.name}' já foi processado.")
        return
    
    with st.spinner(f"Processando '{uploaded_file.name}'..."):
        try:
            text_content = extract_text_from_pdf(file_bytes)
            
            # Armazenar o conteúdo extraído
            st.session_state.pdf_contents[uploaded_file.name] = {
                "text": text_content,
                "hash": file_hash
            }
            
            # Adicionar o hash à lista de arquivos processados
            st.session_state.processed_files.append(file_hash)
            
            # Atualizar o texto combinado
            update_combined_text()
            
            # Salvar automaticamente o estado após processar um PDF
            save_state_to_file()
            
            st.success(f"Arquivo '{uploaded_file.name}' processado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")

def update_combined_text():
    """Atualiza o texto combinado de todos os PDFs processados."""
    combined = ""
    
    for filename, content in st.session_state.pdf_contents.items():
        combined += f"\n\n=== DOCUMENTO: {filename} ===\n\n"
        combined += content["text"]
    
    st.session_state.combined_text = combined

def query_ai(query):
    """Processa uma consulta usando a API da OpenAI."""
    if not openai_api_key:
        st.error("A chave da API OpenAI não está configurada. Por favor, contate o administrador.")
        return None
    
    if not st.session_state.combined_text:
        st.warning("Nenhum documento foi processado ainda. Por favor, aguarde até que o administrador adicione documentos ao sistema.")
        return None
    
    try:
        # Limitar o texto para evitar exceder os limites da API
        text_context = st.session_state.combined_text[:7500]  # Ajuste conforme necessário
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # ou outro modelo compatível
            messages=[
                {"role": "system", "content": """Você é um assistente especializado em analisar documentos e 
                 responder perguntas com base no conteúdo fornecido. Responda apenas com informações presentes 
                 nos documentos. Se a informação não estiver nos documentos, indique claramente."""},
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
    """Reseta o sistema, limpando todos os dados processados."""
    st.session_state.pdf_contents = {}
    st.session_state.processed_files = []
    st.session_state.combined_text = ""
    
    # Salvar o estado vazio
    save_state_to_file()
    
    st.success("Sistema resetado com sucesso. Todos os dados foram limpos.")

def save_state_to_file():
    """Salva o estado atual do sistema em um arquivo."""
    state_data = {
        "pdf_contents": st.session_state.pdf_contents,
        "processed_files": st.session_state.processed_files,
        "combined_text": st.session_state.combined_text
    }
    
    try:
        # Criar diretório 'data' se não existir
        os.makedirs('data', exist_ok=True)
        
        # Salvar estado em arquivo
        with open('data/oraculo_state.dat', 'wb') as f:
            pickle.dump(state_data, f)
    except Exception as e:
        st.error(f"Erro ao salvar estado: {str(e)}")

def load_state_from_file():
    """Carrega o estado salvo anteriormente a partir de um arquivo."""
    try:
        # Verificar se o arquivo existe
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

def download_state():
    """Permite baixar o estado atual do sistema."""
    state_data = {
        "pdf_contents": st.session_state.pdf_contents,
        "processed_files": st.session_state.processed_files,
        "combined_text": st.session_state.combined_text
    }
    
    try:
        serialized = pickle.dumps(state_data)
        b64_data = base64.b64encode(serialized).decode()
        
        st.download_button(
            label="Baixar Estado do Sistema",
            data=b64_data,
            file_name=f"oraculo_state_{int(time.time())}.dat",
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"Erro ao preparar download: {str(e)}")

def load_state_from_upload(uploaded_file):
    """Carrega um estado a partir de um arquivo carregado."""
    try:
        b64_data = uploaded_file.read()
        serialized = base64.b64decode(b64_data)
        state_data = pickle.loads(serialized)
        
        st.session_state.pdf_contents = state_data["pdf_contents"]
        st.session_state.processed_files = state_data["processed_files"]
        st.session_state.combined_text = state_data["combined_text"]
        
        # Salvar o estado carregado
        save_state_to_file()
        
        st.success("Estado restaurado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar estado: {str(e)}")

# Carregar estado salvo ao iniciar o aplicativo
load_state_from_file()

# Interface principal
def main():
    st.title("🔮 Oráculo - Sistema de Consulta Multi-PDF")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Alternar entre modo admin e usuário comum
        admin_mode = st.checkbox("Modo Administrador", value=st.session_state.is_admin)
        if admin_mode != st.session_state.is_admin:
            st.session_state.is_admin = admin_mode
            # Usar st.rerun() em vez de st.experimental_rerun()
            st.rerun()
        
        st.divider()
        
        # Estatísticas
        st.subheader("📊 Estatísticas")
        st.write(f"Documentos carregados: {len(st.session_state.pdf_contents)}")
        
        if st.session_state.pdf_contents:
            total_pages = sum(1 for content in st.session_state.pdf_contents.values() 
                             for line in content["text"].split("\n") if line.startswith("--- Página "))
            st.write(f"Total de páginas: {total_pages}")
        
        st.divider()
        
        # Gerenciamento de estado (apenas para admin)
        if st.session_state.is_admin:
            st.subheader("💾 Gerenciamento de Estado")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Salvar Estado", use_container_width=True):
                    download_state()
            
            with col2:
                if st.button("Resetar Sistema", use_container_width=True):
                    reset_system()
            
            # Uploader para arquivo de estado (.dat) - completamente separado
            st.write("Carregar Estado Salvo")
            state_uploader = st.file_uploader(
                "Arquivo de Estado (.dat)",
                type=["dat"],
                key="estado_dat"
            )
            
            if state_uploader is not None:
                if st.button("Restaurar Estado", key="btn_restore"):
                    load_state_from_upload(state_uploader)
    
    # Modo Administrador
    if st.session_state.is_admin:
        admin_interface()
    else:
        # Modo Usuário Comum
        user_interface()

def admin_interface():
    """Interface para administradores."""
    st.header("🔧 Interface do Administrador")
    st.write("Neste modo, você pode gerenciar os documentos disponíveis para consulta.")
    
    # Upload de documentos
    st.subheader("📁 Upload de Documentos PDF")
    st.write("Carregue um ou mais arquivos PDF para processamento.")
    
    # Upload de PDFs com configurações completamente diferentes
    pdf_uploader = st.file_uploader(
        "Escolha os arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="admin_pdf_uploader"
    )
    
    # Processar PDFs carregados
    if pdf_uploader:
        for i, uploaded_file in enumerate(pdf_uploader):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Arquivo: {uploaded_file.name}")
            with col2:
                process_button = st.button(
                    "Processar",
                    key=f"btn_process_{i}_{uploaded_file.name.replace('.', '_')}"
                )
                if process_button:
                    process_pdf(uploaded_file)
    
    # Exibir documentos processados
    if st.session_state.pdf_contents:
        st.subheader("📚 Documentos Processados")
        
        for idx, (filename, content) in enumerate(st.session_state.pdf_contents.items()):
            with st.expander(f"{idx+1}. {filename}", key=f"doc_expander_{idx}"):
                st.text_area(
                    "Amostra do texto extraído",
                    content["text"][:1000] + "..." if len(content["text"]) > 1000 else content["text"],
                    height=200,
                    key=f"text_preview_{idx}_{filename.replace('.', '_')}"
                )

def user_interface():
    """Interface para usuários comuns."""
    # Interface simplificada sem abas para maior clareza
    st.header("🔍 Consultar Documentos")
    
    if not st.session_state.pdf_contents:
        st.warning("Nenhum documento disponível. Por favor, aguarde até que um administrador adicione documentos ao sistema.")
    else:
        st.write("Digite sua pergunta para consultar os documentos disponíveis.")
        
        # Campo de consulta com chave única e identificação clara
        query = st.text_input(
            "❓ Sua pergunta sobre os documentos:",
            key="query_text_input"
        )
        
        # Botão de consulta destacado
        if st.button("🔎 Enviar Consulta", key="send_query_button", use_container_width=True) and query:
            with st.spinner("Processando sua consulta..."):
                answer = query_ai(query)
                
                if answer:
                    st.divider()
                    st.subheader("📝 Resposta:")
                    st.markdown(answer)
                    
                    # Adicionar ao histórico
                    st.session_state.history.append({
                        "query": query,
                        "answer": answer,
                        "timestamp": time.strftime("%d/%m/%Y %H:%M:%S")
                    })
        
        # Mostrar o histórico diretamente abaixo
        if st.session_state.history:
            st.divider()
            st.subheader("📋 Histórico de Consultas")
            
            for idx, item in enumerate(reversed(st.session_state.history)):
                with st.expander(
                    f"{item['timestamp']} - {item['query']}",
                    key=f"history_item_{idx}"
                ):
                    st.markdown(item['answer'])

if __name__ == "__main__":
    main()
