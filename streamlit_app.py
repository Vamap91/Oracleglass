import streamlit as st
import os
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import openai
import pickle
import hashlib
import time
import base64

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

# Funções utilitárias
def configure_openai_api():
    """Configura a API da OpenAI com a chave fornecida."""
    api_key = st.session_state.get("openai_api_key", "")
    if api_key:
        openai.api_key = api_key
        return True
    return False

def get_file_hash(file_content):
    """Gera um hash único para o conteúdo do arquivo."""
    return hashlib.md5(file_content).hexdigest()

def extract_text_from_pdf(pdf_path):
    """Extrai texto de um PDF, usando OCR quando necessário."""
    doc = fitz.open(pdf_path)
    text_content = ""
    total_pages = len(doc)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Se o texto extraído for muito curto, aplicar OCR
        if len(text.strip()) < 100:
            status_text.text(f"Aplicando OCR na página {page_num+1}/{total_pages}...")
            
            # Converter a página para imagem
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Usar OCR para extrair texto
            text = pytesseract.image_to_string(img, lang='por')
        
        text_content += f"\n--- Página {page_num+1} ---\n{text}"
        progress_bar.progress((page_num + 1) / total_pages)
        status_text.text(f"Processando página {page_num+1}/{total_pages}")
    
    progress_bar.empty()
    status_text.empty()
    
    return text_content

def save_uploaded_file(uploaded_file):
    """Salva o arquivo carregado em um diretório temporário."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def process_pdf(uploaded_file):
    """Processa um arquivo PDF carregado."""
    file_hash = get_file_hash(uploaded_file.getvalue())
    
    # Verificar se o arquivo já foi processado
    if file_hash in st.session_state.processed_files:
        st.info(f"O arquivo '{uploaded_file.name}' já foi processado.")
        return
    
    # Salvar e processar o arquivo
    temp_path = save_uploaded_file(uploaded_file)
    
    with st.spinner(f"Processando '{uploaded_file.name}'..."):
        try:
            text_content = extract_text_from_pdf(temp_path)
            
            # Armazenar o conteúdo extraído
            st.session_state.pdf_contents[uploaded_file.name] = {
                "text": text_content,
                "hash": file_hash
            }
            
            # Adicionar o hash à lista de arquivos processados
            st.session_state.processed_files.append(file_hash)
            
            # Atualizar o texto combinado
            update_combined_text()
            
            st.success(f"Arquivo '{uploaded_file.name}' processado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")
        finally:
            # Remover o arquivo temporário
            try:
                os.unlink(temp_path)
            except:
                pass

def update_combined_text():
    """Atualiza o texto combinado de todos os PDFs processados."""
    combined = ""
    
    for filename, content in st.session_state.pdf_contents.items():
        combined += f"\n\n=== DOCUMENTO: {filename} ===\n\n"
        combined += content["text"]
    
    st.session_state.combined_text = combined

def query_ai(query):
    """Processa uma consulta usando a API da OpenAI."""
    if not configure_openai_api():
        st.error("Por favor, insira uma chave de API válida da OpenAI.")
        return None
    
    if not st.session_state.combined_text:
        st.warning("Nenhum documento foi processado ainda. Por favor, carregue pelo menos um PDF.")
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
    st.success("Sistema resetado com sucesso. Todos os dados foram limpos.")

def save_state():
    """Salva o estado atual do sistema."""
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
        
        st.success("Estado pronto para download. Clique no botão para baixar.")
    except Exception as e:
        st.error(f"Erro ao salvar estado: {str(e)}")

def load_state(uploaded_file):
    """Carrega um estado salvo anteriormente."""
    try:
        b64_data = uploaded_file.read()
        serialized = base64.b64decode(b64_data)
        state_data = pickle.loads(serialized)
        
        st.session_state.pdf_contents = state_data["pdf_contents"]
        st.session_state.processed_files = state_data["processed_files"]
        st.session_state.combined_text = state_data["combined_text"]
        
        st.success("Estado restaurado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar estado: {str(e)}")

# Interface principal
def main():
    st.title("🔮 Oráculo - Sistema de Consulta Multi-PDF")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Configuração da API OpenAI
        st.subheader("API OpenAI")
        api_key = st.text_input("Chave da API OpenAI", 
                               value=st.session_state.get("openai_api_key", ""),
                               type="password")
        
        if api_key:
            st.session_state.openai_api_key = api_key
        
        st.divider()
        
        # Estatísticas
        st.subheader("📊 Estatísticas")
        st.write(f"Documentos carregados: {len(st.session_state.pdf_contents)}")
        
        if st.session_state.pdf_contents:
            total_pages = sum(1 for content in st.session_state.pdf_contents.values() 
                             for line in content["text"].split("\n") if line.startswith("--- Página "))
            st.write(f"Total de páginas: {total_pages}")
        
        st.divider()
        
        # Gerenciamento de estado
        st.subheader("💾 Gerenciamento de Estado")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Salvar Estado", use_container_width=True):
                save_state()
        
        with col2:
            if st.button("Resetar Sistema", use_container_width=True):
                reset_system()
        
        state_file = st.file_uploader("Carregar Estado Salvo", type=["dat"])
        if state_file is not None:
            if st.button("Restaurar Estado"):
                load_state(state_file)
    
    # Abas principais
    tab1, tab2, tab3 = st.tabs(["📁 Upload de Documentos", "🔍 Consultar", "📋 Histórico"])
    
    # Aba de Upload
    with tab1:
        st.header("Upload de Documentos PDF")
        st.write("Carregue um ou mais arquivos PDF para processamento.")
        
        uploaded_files = st.file_uploader("Escolha os arquivos PDF", 
                                         type="pdf", 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Processar '{uploaded_file.name}'"):
                    process_pdf(uploaded_file)
        
        st.divider()
        
        # Exibir documentos processados
        if st.session_state.pdf_contents:
            st.subheader("Documentos Processados")
            
            for idx, (filename, content) in enumerate(st.session_state.pdf_contents.items()):
                with st.expander(f"{idx+1}. {filename}"):
                    st.text_area(
                        "Amostra do texto extraído",
                        content["text"][:1000] + "..." if len(content["text"]) > 1000 else content["text"],
                        height=200
                    )
                    
                    if st.button(f"Remover '{filename}'", key=f"remove_{idx}"):
                        # Remover o documento
                        file_hash = content["hash"]
                        if file_hash in st.session_state.processed_files:
                            st.session_state.processed_files.remove(file_hash)
                        
                        del st.session_state.pdf_contents[filename]
                        update_combined_text()
                        st.experimental_rerun()
    
    # Aba de Consulta
    with tab2:
        st.header("Consultar Documentos")
        
        if not st.session_state.pdf_contents:
            st.warning("Nenhum documento processado. Por favor, carregue e processe pelo menos um PDF.")
        else:
            st.write("Digite sua pergunta para consultar os documentos carregados.")
            
            query = st.text_input("Sua pergunta:")
            
            if st.button("Consultar") and query:
                with st.spinner("Processando consulta..."):
                    answer = query_ai(query)
                    
                    if answer:
                        st.divider()
                        st.subheader("Resposta:")
                        st.write(answer)
                        
                        # Adicionar ao histórico
                        st.session_state.history.append({
                            "query": query,
                            "answer": answer,
                            "timestamp": time.strftime("%d/%m/%Y %H:%M:%S")
                        })
    
    # Aba de Histórico
    with tab3:
        st.header("Histórico de Consultas")
        
        if not st.session_state.history:
            st.info("Nenhuma consulta realizada ainda.")
        else:
            for idx, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"{item['timestamp']} - {item['query']}"):
                    st.write(item['answer'])

if __name__ == "__main__":
    main()
