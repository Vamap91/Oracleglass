import streamlit as st
import os
import pypdf
from io import BytesIO
import time
from datetime import datetime
import base64
import pandas as pd
import tiktoken
from rag_engine import RAGEngine

# Configuração da página
st.set_page_config(
    page_title="Oráculo - Sistema de Consulta Inteligente",
    page_icon="🚗",
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
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "index_status" not in st.session_state:
    st.session_state.index_status = "Não inicializado"

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
    
    # 4. Verificar dependências básicas
    try:
        import pypdf
        import openai
        VALIDATION_MESSAGES["dependencies"] = "OK"
    except ImportError as e:
        VALIDATION_OK = False
        VALIDATION_MESSAGES["dependencies"] = f"Dependência faltando: {str(e)}"
    
    # 5. Verificar dependências RAG
    try:
        import faiss
        import sentence_transformers
        import numpy
        VALIDATION_MESSAGES["rag_dependencies"] = "OK"
    except ImportError as e:
        VALIDATION_OK = False
        VALIDATION_MESSAGES["rag_dependencies"] = f"Dependência faltando para RAG: {str(e)}"
    
    return VALIDATION_OK

def extract_text_from_pdf(pdf_path):
    """Extrai texto de um arquivo PDF com método melhorado para preservar estrutura"""
    text_content = ""
    try:
        # Abrir o arquivo PDF
        reader = pypdf.PdfReader(pdf_path)
        
        # Iterar por todas as páginas
        for page_num, page in enumerate(reader.pages):
            # Extrair o texto com configurações para preservar layout
            page_text = page.extract_text() or ""
            
            # Adicionar número da página e texto
            text_content += f"\n--- Página {page_num+1} ---\n{page_text}\n"
            
            # Tentar extrair tabelas e conteúdo estruturado (abordagem simples)
            lines = page_text.split('\n')
            for i, line in enumerate(lines):
                # Detectar possíveis números de telefone
                if ('0800' in line or '4004' in line or 
                    'telefone' in line.lower() or 'contato' in line.lower() or
                    '-' in line and any(c.isdigit() for c in line)):
                    # Adicionar linhas ao redor para contexto
                    start_idx = max(0, i-2)
                    end_idx = min(len(lines), i+3)
                    context_lines = lines[start_idx:end_idx]
                    text_content += "\n--- INFORMAÇÃO DE CONTATO DETECTADA ---\n"
                    text_content += "\n".join(context_lines) + "\n"
                    text_content += "--- FIM DA INFORMAÇÃO DE CONTATO ---\n"
    
    except Exception as e:
        st.error(f"Erro ao processar PDF: {str(e)}")
    
    return text_content

def initialize_rag_engine():
    """Inicializa ou carrega o motor RAG"""
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine()
        
    # Tentar carregar um índice existente
    if st.session_state.rag_engine.load_index():
        st.session_state.index_status = "Índice carregado com sucesso"
        return True
    
    # Se não conseguir carregar, criar um novo índice
    try:
        with st.spinner("Criando índice vetorial do documento... Isso pode levar alguns minutos."):
            start_time = time.time()
            st.session_state.rag_engine.create_index(st.session_state.pdf_text)
            elapsed_time = time.time() - start_time
            st.session_state.index_status = f"Índice criado em {elapsed_time:.2f} segundos"
        return True
    except Exception as e:
        st.session_state.index_status = f"Erro ao criar índice: {str(e)}"
        return False

def estimate_tokens(text, model="gpt-3.5-turbo"):
    """Estima o número de tokens em um texto"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Estimativa simples se tiktoken falhar
        return len(text) // 4

def query_ai(query):
    """
    Processa uma consulta usando RAG (Retrieval Augmented Generation)
    """
    try:
        # Importar OpenAI dentro da função para evitar erros de escopo
        import openai
        
        # Cliente OpenAI para v1.0+
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Verificar se o RAG engine está inicializado
        if st.session_state.rag_engine is None or st.session_state.index_status.startswith("Erro"):
            st.warning("O motor RAG não foi inicializado corretamente. Tentando inicializar novamente...")
            if not initialize_rag_engine():
                return "Não foi possível inicializar o motor RAG para processar sua consulta. Tente novamente mais tarde."
        
        # Instrução do sistema refinada para melhor processamento de termos específicos
        system_prompt = """
        Você é um assistente de IA especializado em analisar o conteúdo de um documento PDF fornecido e responder perguntas exclusivamente com base nesse conteúdo.
        
        Instruções Importantes:
        1. Sua resposta DEVE ser estritamente baseada nas informações contidas no texto do documento fornecido.
        2. NÃO utilize conhecimento externo ou informações que não estejam presentes no documento.
        3. Se a pergunta se referir a um tópico ou termo específico (ex: 'undercar', 'telefone', '0800'), procure cuidadosamente por todas as menções desse tópico no documento completo antes de responder.
        4. Se a informação solicitada estiver presente, forneça-a de forma clara e cite a parte relevante do texto, se possível.
        5. Se a informação solicitada NÃO estiver presente no documento, declare explicitamente que a informação não foi encontrada no documento fornecido.
        6. Se partes do documento parecerem incompletas ou ambíguas em relação à pergunta, mencione isso.
        """
        
        st.session_state.processing_status = "Buscando informações relevantes no documento..."
        
        # Número de chunks a recuperar (ajuste conforme necessário)
        top_k = 5
        
        # Usar o motor RAG para consulta
        response = st.session_state.rag_engine.query_with_context(
            client=client,
            query=query,
            model=st.session_state.model,
            system_prompt=system_prompt,
            top_k=top_k
        )
        
        st.session_state.processing_status = "Consulta processada com sucesso."
        return response
        
    except Exception as e:
        st.error(f"Erro inesperado ao processar consulta: {str(e)}")
        st.session_state.processing_status = f"Erro: {str(e)}"
        return f"Ocorreu um erro inesperado: {str(e)}"

def verificar_termos_no_pdf(termos, texto_pdf=None):
    """
    Verifica se determinados termos estão presentes no texto do PDF
    e retorna suas posições no texto.
    
    Args:
        termos (list): Lista de termos a serem verificados
        texto_pdf (str, optional): Texto do PDF. Se None, usa st.session_state.pdf_text
        
    Returns:
        dict: Dicionário com os termos encontrados e suas posições
    """
    if texto_pdf is None:
        if "pdf_text" not in st.session_state:
            return {"erro": "Texto do PDF não disponível"}
        texto_pdf = st.session_state.pdf_text
    
    resultados = {}
    
    for termo in termos:
        termo = termo.lower()
        posicoes = []
        texto_lower = texto_pdf.lower()
        
        # Encontrar todas as ocorrências
        pos = texto_lower.find(termo)
        while pos != -1:
            # Adicionar contexto (50 caracteres antes e depois)
            inicio = max(0, pos - 50)
            fim = min(len(texto_pdf), pos + len(termo) + 50)
            contexto = texto_pdf[inicio:fim]
            
            posicoes.append({
                "posicao": pos,
                "contexto": contexto
            })
            
            # Procurar próxima ocorrência
            pos = texto_lower.find(termo, pos + 1)
        
        if posicoes:
            resultados[termo] = posicoes
    
    return resultados

def diagnosticar_reconhecimento(query, texto_pdf=None):
    """
    Função de diagnóstico para ajudar a identificar problemas de reconhecimento
    de termos específicos no PDF.
    
    Args:
        query (str): A consulta do usuário
        texto_pdf (str, optional): Texto do PDF. Se None, usa st.session_state.pdf_text
        
    Returns:
        dict: Resultados do diagnóstico
    """
    if texto_pdf is None:
        if "pdf_text" not in st.session_state:
            return {"erro": "Texto do PDF não disponível"}
        texto_pdf = st.session_state.pdf_text
    
    # Extrair palavras-chave da consulta (palavras com mais de 3 letras)
    palavras = [p for p in query.lower().split() if len(p) > 3]
    
    # Verificar presença das palavras-chave no texto
    resultados = verificar_termos_no_pdf(palavras, texto_pdf)
    
    # Adicionar estatísticas gerais
    diagnostico = {
        "tamanho_texto": len(texto_pdf),
        "palavras_analisadas": palavras,
        "palavras_encontradas": list(resultados.keys()),
        "detalhes": resultados
    }
    
    return diagnostico

def export_to_csv():
    """Exporta o histórico para CSV"""
    if not st.session_state.history:
        st.warning("Não há consultas para exportar.")
        return None, None
    
    # Criar DataFrame com o histórico
    data = []
    for item in st.session_state.history:
        data.append({
            "Data e Hora": item.get("timestamp", ""),
            "Pergunta": item.get("query", ""),
            "Resposta": item.get("answer", ""),
            "Modelo": item.get("model", "")
        })
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"oraculo_historico_{timestamp}.csv"
    
    return csv, filename

def show_extracted_text():
    """Exibe o texto extraído para depuração"""
    if st.session_state.pdf_text:
        st.text_area("Texto Completo Extraído do PDF", st.session_state.pdf_text, height=300)
        
        # Adicionar ferramenta de diagnóstico para termos específicos
        st.subheader("Diagnóstico de Termos")
        termo_busca = st.text_input("Digite um termo para verificar no PDF:")
        if termo_busca and st.button("Verificar Termo"):
            resultados = verificar_termos_no_pdf([termo_busca])
            if termo_busca.lower() in resultados:
                st.success(f"Termo '{termo_busca}' encontrado {len(resultados[termo_busca.lower()])} vezes no documento!")
                for i, ocorrencia in enumerate(resultados[termo_busca.lower()]):
                    st.markdown(f"**Ocorrência {i+1}:** (posição {ocorrencia['posicao']})")
                    st.text(f"...{ocorrencia['contexto']}...")
            else:
                st.error(f"Termo '{termo_busca}' não encontrado no documento.")
                
        # Diagnóstico do RAG
        st.subheader("Diagnóstico do RAG")
        if st.session_state.rag_engine is not None and st.session_state.rag_engine.chunks:
            stats = st.session_state.rag_engine.get_chunks_stats()
            st.write(f"Total de chunks: {stats['total_chunks']}")
            st.write(f"Tamanho médio: {stats['avg_chunk_size']:.0f} caracteres")
            st.write(f"Tamanho total: {stats['total_content_size']} caracteres")
            
            if st.button("Testar Busca RAG"):
                termo_teste = termo_busca or "veículo"
                st.write(f"Testando busca por: '{termo_teste}'")
                try:
                    resultados = st.session_state.rag_engine.search(termo_teste, top_k=3)
                    st.write(f"Encontrados {len(resultados)} resultados relevantes:")
                    for i, (idx, texto, score) in enumerate(resultados):
                        with st.expander(f"Resultado {i+1} (score: {score:.4f})"):
                            st.text(texto[:300] + "..." if len(texto) > 300 else texto)
                except Exception as e:
                    st.error(f"Erro ao testar RAG: {str(e)}")
        else:
            st.warning("Sistema RAG não inicializado. Inicialize-o primeiro.")
            if st.button("Inicializar RAG"):
                initialize_rag_engine()
    else:
        st.warning("Nenhum texto extraído ainda.")

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
    st.write(f"{dep_status} **Dependências Básicas**: {VALIDATION_MESSAGES.get('dependencies')}")
    
    # Status de dependências RAG
    rag_dep_status = "✅" if VALIDATION_MESSAGES.get("rag_dependencies") == "OK" else "❌"
    st.write(f"{rag_dep_status} **Dependências RAG**: {VALIDATION_MESSAGES.get('rag_dependencies')}")
    
    # Informações do PDF
    if VALIDATION_OK:
        st.divider()
        st.subheader("📄 Informações do PDF")
        st.write(f"Arquivo: {os.path.basename(PDF_PATH)}")
        st.write(f"Tamanho do texto: {len(st.session_state.pdf_text)} caracteres")
        st.write(f"Tokens estimados: {estimate_tokens(st.session_state.pdf_text)}")
        
        # Status do sistema RAG
        st.divider()
        st.subheader("🧠 Sistema RAG")
        st.write(f"Status: {st.session_state.index_status}")
        
        if st.button("Inicializar/Reconstruir Índice RAG"):
            initialize_rag_engine()
        
        # Configurações de modelo
        st.divider()
        st.subheader("⚙️ Configurações")
        
        model = st.selectbox(
            "Modelo OpenAI:",
            options=[
                "gpt-3.5-turbo",      # Modelo básico, bom custo-benefício
                "gpt-4",              # Melhor qualidade, mais caro
                "gpt-4-turbo",        # Melhor performance que GPT-4 com custo menor
                "gpt-4o",             # GPT-4 Omni - modelo mais recente
                "gpt-3.5-turbo-16k",  # Versão com contexto maior
                "gpt-4-32k",          # GPT-4 com contexto de 32k tokens
            ],
            index=0,  # Default para o modelo mais econômico
            help="Selecione o modelo da OpenAI. GPT-3.5 é mais rápido e econômico, GPT-4 é mais preciso, modelos com número maior suportam mais contexto."
        )
        
        # Atualizar o modelo selecionado
        if model != st.session_state.model:
            st.session_state.model = model
            st.info(f"Modelo alterado para {model}")
        
        # Botão para exportar histórico
        st.divider()
        st.subheader("📊 Exportar Histórico")
        
        export_button = st.button("📥 Exportar Consultas para CSV")
        if export_button:
            csv_data, filename = export_to_csv()
            if csv_data is not None and filename is not None:
                st.download_button(
                    label="⬇️ Baixar CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                )
        
        # Modo de depuração para visualizar o texto extraído
        st.divider()
        st.subheader("🔍 Depuração")
        if st.button("Visualizar Texto Extraído"):
            show_extracted_text()

# Conteúdo principal - Sempre exibir, independente da validação
st.write("Digite sua pergunta sobre veículos e clique em 'Consultar'.")

# Campo de consulta
query = st.text_input("❓ Sua pergunta:", key="query_input")

# Status do processamento
if st.session_state.processing_status:
    st.info(st.session_state.processing_status)

# Inicializar o RAG engine se o ambiente estiver validado
if VALIDATION_OK and "pdf_text" in st.session_state and st.session_state.pdf_text and st.session_state.rag_engine is None:
    st.info("Inicializando sistema RAG para processamento eficiente de documentos grandes...")
    initialize_rag_engine()

# Botão de consulta
consult_button = st.button("🔍 Consultar", key="query_button", disabled=not VALIDATION_OK)
if consult_button and query:
    st.session_state.processing_status = "Iniciando processamento com RAG..."
    with st.spinner("Processando consulta..."):
        answer = query_ai(query)
        
        if answer:
            st.divider()
            st.subheader("📝 Resposta:")
            st.markdown(answer)
            
            # Adicionar ao histórico com timestamp
            st.session_state.history.append({
                "query": query,
                "answer": answer,
                "model": st.session_state.model,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            })
            
            # Resetar status após conclusão
            st.session_state.processing_status = None

# Histórico de consultas - sempre mostrar se houver itens
if st.session_state.history:
    st.divider()
    st.subheader("📋 Histórico de Consultas")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        question = item.get("query", "")
        with st.expander(f"Pergunta ({i+1}): {question}"):
            st.markdown(f"**Data e Hora:** {item.get('timestamp', '')}")
            st.markdown(f"**Modelo:** {item.get('model', '')}")
            st.markdown("**Resposta:**")
            st.markdown(item.get('answer', ''))
