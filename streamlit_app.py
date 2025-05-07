import streamlit as st
import os
from rag_engine import RAGEngine # Supondo que rag_engine.py está no mesmo diretório ou no PYTHONPATH

# Diretório para salvar os PDFs enviados, se necessário, ou para listar PDFs existentes
UPLOAD_DIR = "data/uploads"
AVAILABLE_PDFS_DIR = "data" # Diretório onde o "Guia Rápido.pdf" e outros podem estar
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AVAILABLE_PDFS_DIR, exist_ok=True)

# Função para listar arquivos PDF em um diretório
def list_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

# Inicializar o RAG Engine no estado da sessão para persistência
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_pdf_path" not in st.session_state:
    st.session_state.current_pdf_path = None

st.set_page_config(layout="wide", page_title="OracleGlass - Q&A com Documentos")

st.title("OracleGlass :mag_right::page_facing_up:")
st.markdown("Faça perguntas sobre o conteúdo do seu documento PDF.")

# Sidebar para seleção e upload de PDF
st.sidebar.header("Configuração do Documento")

# Opção para selecionar um PDF existente ou fazer upload
pdf_source_option = st.sidebar.radio(
    "Escolha a fonte do PDF:",
    ("Selecionar PDF existente", "Fazer upload de novo PDF")
)

selected_pdf_path = None

if pdf_source_option == "Selecionar PDF existente":
    available_pdfs = list_pdf_files(AVAILABLE_PDFS_DIR)
    if not available_pdfs:
        st.sidebar.warning(f"Nenhum PDF encontrado em `{AVAILABLE_PDFS_DIR}`. Faça upload de um arquivo.")
    else:
        selected_pdf_name = st.sidebar.selectbox("Selecione um PDF:", available_pdfs)
        if selected_pdf_name:
            selected_pdf_path = os.path.join(AVAILABLE_PDFS_DIR, selected_pdf_name)
else: # Fazer upload de novo PDF
    uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo PDF aqui", type=["pdf"])
    if uploaded_file is not None:
        # Salvar o arquivo upload para um local persistente se necessário
        # ou processá-lo diretamente do buffer de memória (se RAGEngine suportar)
        # Por simplicidade, vamos salvá-lo e usar o caminho
        upload_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        try:
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_pdf_path = upload_path
            st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao salvar o arquivo: {e}")

if selected_pdf_path:
    # Botão para processar o PDF selecionado/carregado
    if st.sidebar.button("Processar PDF Selecionado") or st.session_state.current_pdf_path != selected_pdf_path:
        if st.session_state.current_pdf_path != selected_pdf_path or not st.session_state.pdf_processed:
            st.session_state.current_pdf_path = selected_pdf_path
            st.session_state.pdf_processed = False # Resetar status
            st.session_state.rag_engine = None # Resetar engine para novo PDF
            try:
                with st.spinner(f"Processando '{os.path.basename(selected_pdf_path)}'... Isso pode levar alguns minutos."):
                    st.session_state.rag_engine = RAGEngine() # Inicializa sem PDF ainda
                    st.session_state.rag_engine.load_and_process_pdf(selected_pdf_path) # Carrega e processa
                
                if st.session_state.rag_engine and st.session_state.rag_engine.vector_store:
                    st.session_state.pdf_processed = True
                    st.sidebar.success(f"PDF '{os.path.basename(selected_pdf_path)}' processado e pronto para perguntas!")
                else:
                    st.sidebar.error("Falha ao processar o PDF. Verifique os logs ou o arquivo.")
                    st.session_state.pdf_processed = False
            except Exception as e:
                st.sidebar.error(f"Erro durante o processamento do PDF: {e}")
                st.session_state.pdf_processed = False
                st.session_state.rag_engine = None

# Se um PDF foi processado com sucesso, mostrar a interface de Q&A
if st.session_state.pdf_processed and st.session_state.rag_engine:
    st.subheader(f"Perguntando sobre: {os.path.basename(st.session_state.current_pdf_path)}")
    
    # Inicializar histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Qual sua pergunta sobre o documento?"):
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar resposta do assistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Pensando..."):
                    # Aqui chamaria a lógica de busca e geração de resposta do RAG Engine
                    # Por enquanto, vamos usar a busca semântica simples
                    search_results = st.session_state.rag_engine.search(prompt, k=3) # Pega os 3 chunks mais relevantes
                    
                    if search_results:
                        context = "\n\n---\n\n".join([chunk for chunk, score in search_results])
                        # Simular uma resposta baseada no contexto (idealmente, usar um LLM aqui)
                        # full_response = f"Com base nos trechos encontrados:\n\n{context}\n\n(Esta é uma resposta simulada baseada na busca. Para uma resposta completa, um modelo de linguagem seria necessário.)"
                        
                        # Para uma resposta mais direta, apenas mostrar os chunks relevantes
                        full_response = "**Trechos relevantes encontrados no documento:**\n\n"
                        for i, (chunk, score) in enumerate(search_results):
                            full_response += f"**Trecho {i+1} (Relevância: {score:.2f}):**\n{chunk}\n\n"
                        if not full_response.strip():
                             full_response = "Não encontrei informações diretamente relevantes para sua pergunta no documento."
                    else:
                        full_response = "Desculpe, não consegui encontrar informações relevantes para sua pergunta no documento processado."
            except Exception as e:
                full_response = f"Ocorreu um erro ao processar sua pergunta: {e}"
            
            message_placeholder.markdown(full_response)
        # Adicionar resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif st.session_state.current_pdf_path:
    st.info("Por favor, clique em 'Processar PDF Selecionado' na barra lateral para começar.")
else:
    st.info("Por favor, selecione ou faça upload de um arquivo PDF e clique em 'Processar PDF' na barra lateral para começar.")

# Adicionar um rodapé ou informações adicionais
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por Manus IA")

# Para executar: streamlit run streamlit_app.py
