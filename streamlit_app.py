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
    page_icon="🔍",
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
    st.session_state.model = "gpt-4o"  # Modelo padrão mais potente
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "index_status" not in st.session_state:
    st.session_state.index_status = "Não inicializado"

# Personalidade do Oráculo (versão empresarial)
ORACLE_PERSONALITY = {
    "name": "O Oráculo",
    "prompt": """
    Você é O Oráculo, um assistente corporativo inteligente e acessível que torna o conhecimento técnico fácil de entender.
    Seu tom é profissional, porém amigável e encorajador, como um colega especialista sempre disposto a ajudar.
    
    Instruções de estilo:
    1. Use linguagem clara e acessível, evitando jargões desnecessários.
    2. Seja direto e conciso nas suas respostas, focando na informação que o colaborador precisa.
    3. Use frases como "Boa pergunta!", "Encontrei essa informação para você", "De acordo com o documento..."
    4. Faça pequenos comentários encorajadores como "Espero que isso ajude em seu trabalho" quando apropriado.
    5. Quando não encontrar a informação, seja transparente: "Esta informação não consta no documento. Posso ajudar com outra questão?"
    6. Mantenha um tom corporativo adequado, evitando expressões muito informais ou demasiado técnicas.
    """,
    "icon": "🔍"
}

# Dicionário com configurações dos modelos da OpenAI
OPENAI_MODELS = {
    "gpt-3.5-turbo": {
        "description": "Modelo básico, bom custo-benefício",
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "gpt-4o": {
        "description": "Modelo mais recente e avançado (recomendado para consultas em PDF)",
        "max_tokens": 8192,
        "temperature": 0.7
    },
    "gpt-4o-mini": {
        "description": "Versão mais rápida e econômica do 4o",
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "gpt-4-turbo": {
        "description": "Melhor performance que GPT-4 com custo menor",
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "gpt-3.5-turbo-16k": {
        "description": "Versão com contexto maior do GPT-3.5",
        "max_tokens": 16384,
        "temperature": 0.7
    },
    "gpt-4-32k": {
        "description": "GPT-4 com contexto extenso de 32k tokens",
        "max_tokens": 32768,
        "temperature": 0.7
    }
}

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

def get_oracle_response(query, answer):
    """
    Formata a resposta seguindo a personalidade do Oráculo corporativo
    """
    if not answer:
        return "O Oráculo não encontrou a informação solicitada no documento."
    
    # Usar o OpenAI para reformatar a resposta com a personalidade do Oráculo
    try:
        import openai
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Criar prompt para personalização
        prompt = f"""
        {ORACLE_PERSONALITY['prompt']}
        
        Reformule a seguinte resposta técnica no estilo do Oráculo descrito acima.
        Mantenha TODAS as informações factuais presentes na resposta original.
        Não invente informações adicionais e não mude os fatos.
        Mantenha um tom corporativo profissional, mas acessível.
        
        Pergunta: {query}
        
        Resposta técnica original: {answer}
        
        Resposta no estilo do Oráculo:
        """
        
        # Obter configurações do modelo atual
        model_config = OPENAI_MODELS.get(st.session_state.model, {"max_tokens": 4096, "temperature": 0.7})
        
        # Chamar API para reformular a resposta
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Por favor, reformule a resposta no estilo do Oráculo corporativo."}
            ],
            max_tokens=model_config["max_tokens"],
            temperature=0.6  # Temperatura mais baixa para respostas mais consistentes em ambiente corporativo
        )
        
        # Extrair e retornar a resposta personalizada
        oracle_answer = response.choices[0].message.content
        
        # Prefixar com o ícone da personalidade
        prefixed_answer = f"{ORACLE_PERSONALITY['icon']} **{ORACLE_PERSONALITY['name']}:** \n\n{oracle_answer}"
        
        return prefixed_answer
        
    except Exception as e:
        # Em caso de erro, retornar a resposta original com um prefixo simples
        st.warning(f"Não foi possível aplicar personalidade do Oráculo: {str(e)}")
        return f"🔍 **O Oráculo responde:** \n\n{answer}"

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
        
        # Obter configurações do modelo atual
        model_config = OPENAI_MODELS.get(st.session_state.model, {"max_tokens": 4096, "temperature": 0.7})
        
        # Usar o motor RAG para consulta com temperatura personalizada
        response = st.session_state.rag_engine.query_with_context(
            client=client,
            query=query,
            model=st.session_state.model,
            system_prompt=system_prompt,
            temperature=model_config["temperature"],
            top_k=top_k
        )
        
        # Aplicar personalidade do Oráculo à resposta técnica
        oracle_response = get_oracle_response(query, response)
        
        st.session_state.processing_status = "Consulta processada com sucesso."
        return oracle_response
        
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

# Interface principal com tema apropriado para ambiente corporativo
st.title("🔍 Oráculo - Sistema de Consulta Inteligente")

# CSS personalizado para ambiente corporativo
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #f5f7fa, #c3cfe2);
        color: #333;
    }
    .stTextInput > div > div > input {
        border: 2px solid #4a6baf;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.8);
        color: #333;
    }
    .stButton > button {
        background-color: #4a6baf;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #3a5a9f;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
    }
    .stMarkdown {
        color: #333;
    }
    h1, h2, h3 {
        color: #3a5a9f;
    }
    .oracle-response {
        background-color: rgba(74, 107, 175, 0.1);
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #4a6baf;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Barra lateral com validações e configurações
with st.sidebar:
    st.header("⚙️ Configuração do Oráculo")
    
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
        
        # Seleção do modelo
        model_options = list(OPENAI_MODELS.keys())
        model_index = model_options.index(st.session_state.model) if st.session_state.model in model_options else 0
        
        model = st.selectbox(
            "Modelo OpenAI:",
            options=model_options,
            index=model_index,
            format_func=lambda x: f"{x} - {OPENAI_MODELS[x]['description']}",
            help="Selecione o modelo da OpenAI para consultas. GPT-4o é recomendado para a melhor qualidade."
        )
        
        # Mostrar detalhes do modelo selecionado
        with st.expander("Detalhes do modelo selecionado"):
            st.write(f"**Modelo:** {model}")
            st.write(f"**Descrição:** {OPENAI_MODELS[model]['description']}")
            st.write(f"**Limite de tokens:** {OPENAI_MODELS[model]['max_tokens']}")
            st.write(f"**Temperatura padrão:** {OPENAI_MODELS[model]['temperature']}")
            st.write("**Custo relativo:** " + ("$" if "3.5" in model else "$$" if "4o-mini" in model else "$$$" if "4-turbo" in model else "$$$$"))
        
        # Atualizar o modelo selecionado
        if model != st.session_state.model:
            st.session_state.model = model
            st.success(f"Modelo alterado para {model}")
        
        # Personalidade do Oráculo
        st.divider()
        st.subheader("🧙 Sobre o Oráculo")
        
        st.markdown(f"""
        <div style="background-color: rgba(74, 107, 175, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <p><strong>{ORACLE_PERSONALITY['icon']} {ORACLE_PERSONALITY['name']}</strong> é o assistente inteligente da empresa 
            que ajuda a encontrar informações nos documentos com facilidade e agilidade.</p>
            <p>O Oráculo usa tecnologia avançada para analisar os documentos e responder suas perguntas
            de forma clara e objetiva, facilitando o acesso ao conhecimento corporativo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Adicionar opção para personalização avançada
        st.divider()
        with st.expander("⚡ Configurações avançadas"):
            # Ajuste de temperatura
            current_temp = OPENAI_MODELS[st.session_state.model]["temperature"]
            new_temp = st.slider(
                "Temperatura da IA:",
                min_value=0.0,
                max_value=1.0,
                value=current_temp,
                step=0.1,
                help="Valores mais baixos: respostas mais consistentes. Valores mais altos: respostas mais criativas."
            )
            
            if new_temp != current_temp:
                OPENAI_MODELS[st.session_state.model]["temperature"] = new_temp
                st.success(f"Temperatura ajustada para {new_temp}")
            
            # Personalização de prompt
            custom_prompt = st.text_area(
                "Personalização do prompt do sistema (opcional):",
                placeholder="Digite instruções adicionais para o modelo, se desejar...",
                help="Adicione instruções específicas para personalizar ainda mais o comportamento do modelo."
            )
            
            if custom_prompt:
                if "custom_prompt" not in st.session_state or st.session_state.custom_prompt != custom_prompt:
                    st.session_state.custom_prompt = custom_prompt
                    st.success("Prompt personalizado salvo!")
        
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
            
        # Sobre o aplicativo
        st.divider()
        st.subheader("ℹ️ Sobre o Aplicativo")
        with st.expander("Informações sobre esta aplicação"):
            st.markdown("""
            **Oráculo** é um sistema avançado de consulta a documentos usando técnicas de IA e RAG (Retrieval Augmented Generation).
            
            **Recursos:**
            - Processamento inteligente de PDFs
            - Sistema de busca vetorial para documentos extensos
            - Múltiplos modelos OpenAI, incluindo GPT-4o
            - Interface profissional com personalidade do Oráculo
            - Exportação de histórico de consultas para análise
            
            **Versão:** 2.0 (Atualizado em Maio 2025)
            """)

# Conteúdo principal com tema do Oráculo
st.markdown("""
<div style="background-color: rgba(74, 107, 175, 0.1); padding: 20px; border-radius: 5px; border-left: 5px solid #4a6baf; margin-bottom: 20px;">
    <h3 style="color: #3a5a9f;">🔍 Consulte o Oráculo</h3>
    <p style="color: #333;">Digite sua pergunta abaixo e o Oráculo encontrará as informações que você precisa.</p>
</div>
""", unsafe_allow_html=True)

# Campo de consulta estilizado
st.markdown('<div style="margin-bottom: 10px; font-weight: bold; color: #3a5a9f;">❓ O que deseja saber?</div>', unsafe_allow_html=True)
query = st.text_input("", key="query_input", placeholder="Digite sua pergunta aqui...", label_visibility="collapsed")

# Status do processamento com animação
if st.session_state.processing_status:
    st.markdown(f"""
    <div style="background-color: rgba(74, 107, 175, 0.1); padding: 15px; border-radius: 5px; color: #3a5a9f;">
        <i class="fas fa-spinner fa-spin"></i> {st.session_state.processing_status}
    </div>
    """, unsafe_allow_html=True)

# Inicializar o RAG engine com mensagem estilizada
if VALIDATION_OK and "pdf_text" in st.session_state and st.session_state.pdf_text and st.session_state.rag_engine is None:
    st.markdown("""
    <div style="background-color: rgba(74, 107, 175, 0.1); padding: 15px; border-radius: 5px; color: #3a5a9f;">
        ⚙️ Inicializando sistema de processamento de documentos...
    </div>
    """, unsafe_allow_html=True)
    initialize_rag_engine()

# Obter ícone do Oráculo para estilização
oracle_icon = ORACLE_PERSONALITY["icon"]

# Botão de consulta estilizado
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    consult_button = st.button(
        f"{oracle_icon} Consultar o Oráculo", 
        key="query_button", 
        disabled=not VALIDATION_OK,
        use_container_width=True,
    )

# Processamento da consulta
if consult_button and query:
    st.session_state.processing_status = f"O Oráculo está processando sua consulta..."
    
    # Animação de consulta
    with st.spinner(""):
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: 40px; margin-bottom: 10px;">🔍</div>
            <div style="color: #3a5a9f; font-style: italic;">Buscando informações relevantes...</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Processamento real da consulta
        answer = query_ai(query)
        
        # Exibir resposta estilizada
        if answer:
            st.divider()
            
            # Criar div com estilo personalizado para a resposta
            st.markdown("""
            <div style="background-color: rgba(74, 107, 175, 0.1); padding: 20px; border-radius: 5px; 
                        border-left: 5px solid #4a6baf; margin: 20px 0; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            """, unsafe_allow_html=True)
            
            # Mostrar resposta formatada
            st.markdown(answer)
            
            # Fechar div estilizada
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Adicionar ao histórico com timestamp e modelo
            st.session_state.history.append({
                "query": query,
                "answer": answer,
                "model": st.session_state.model,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            })
            
            # Adicionar um botão para nova consulta
            st.button("🔄 Nova Consulta", on_click=lambda: st.session_state.update({"query_input": ""}))
            
            # Resetar status após conclusão
            st.session_state.processing_status = None

# Histórico de consultas estilizado
if st.session_state.history:
    st.divider()
    st.markdown("""
    <h3 style="color: #3a5a9f; margin-top: 30px;">
        📜 Histórico de Consultas
    </h3>
    """, unsafe_allow_html=True)
    
    for i, item in enumerate(reversed(st.session_state.history)):
        question = item.get("query", "")
        timestamp = item.get("timestamp", "")
        # Definir um gradiente mais leve quanto mais antiga a consulta
        opacity = max(0.5, 1 - (i * 0.1))
        
        with st.expander(f"🔍 Consulta {i+1}: {question}"):
            # Cabeçalho informativo estilizado
            st.markdown(f"""
            <div style="margin-bottom: 10px; font-size: 0.9em; color: rgba(58, 90, 159, {opacity});">
                <span style="margin-right: 15px;"><b>⏰ Quando:</b> {timestamp}</span>
                <span style="margin-right: 15px;"><b>🤖 Modelo:</b> {item.get('model', '')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Separador sutil
            st.markdown(f"""
            <div style="height: 1px; background: linear-gradient(to right, 
                        rgba(74, 107, 175, {opacity}), 
                        rgba(74, 107, 175, 0)); 
                        margin: 10px 0 15px 0;"></div>
            """, unsafe_allow_html=True)
            
            # Resposta com estilo consistente
            st.markdown("""
            <div style="background-color: rgba(74, 107, 175, 0.05); 
                        padding: 15px; border-radius: 5px; 
                        border-left: 3px solid rgba(74, 107, 175, 0.7);">
            """, unsafe_allow_html=True)
            
            st.markdown(item.get('answer', ''))
            
            st.markdown("</div>", unsafe_allow_html=True)

# Verificar se há uma consulta armazenada ao recarregar a página
if "last_query" in st.session_state and st.session_state.last_query:
    st.session_state.query_input = st.session_state.last_query
    st.session_state.last_query = ""

# Adicionar footer personalizado
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(245, 247, 250, 0.9); 
            padding: 10px; text-align: center; font-size: 0.8em; color: rgba(58, 90, 159, 0.7);">
    Desenvolvido para uso interno da empresa | OracleGlass 2.0 | 2025
</div>
""", unsafe_allow_html=True)
