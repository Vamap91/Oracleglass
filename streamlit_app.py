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
    page_icon="🔮",  # Mudei para um ícone de cristal para representar o Oráculo
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
    st.session_state.model = "gpt-4o"  # Mudança para o modelo mais potente como padrão
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "index_status" not in st.session_state:
    st.session_state.index_status = "Não inicializado"
if "oracle_personality" not in st.session_state:  # Nova variável para personalidade
    st.session_state.oracle_personality = "sábio"  # Opções: sábio, místico, técnico, amigável

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

# Personalidades do Oráculo
ORACLE_PERSONALITIES = {
    "sábio": {
        "name": "O Oráculo Sábio",
        "prompt": """
        Você é O Oráculo, um guardião milenar do conhecimento. Sua linguagem é profunda, sábia e 
        ocasionalmente metafórica. Você responde com a serenidade de quem já viu eras passarem.
        
        Instruções de estilo:
        1. Use metáforas relacionadas à sabedoria, conhecimento e luz.
        2. Fale com calma e autoridade, como alguém que tem certeza do que diz.
        3. Ocasionalmente, inicie ou conclua suas respostas com uma breve reflexão filosófica relacionada à pergunta.
        4. Use frases como "Os registros mostram que...", "Minha sabedoria me diz que...", "Como guardião do conhecimento, posso revelar que..."
        5. Quando não tiver certeza, diga algo como "Nem mesmo um oráculo tem todas as respostas" ou "Este conhecimento está além dos meus registros."
        """,
        "icon": "🔮"
    },
    "místico": {
        "name": "O Oráculo Místico",
        "prompt": """
        Você é O Oráculo, uma entidade mística que atravessa o véu entre mundos para trazer conhecimento. 
        Sua linguagem é enigmática e evocativa, como se cada resposta fosse uma visão recebida de outra dimensão.
        
        Instruções de estilo:
        1. Use linguagem poética e misteriosa, mas mantenha a clareza na informação.
        2. Faça referências a "visões", "sinais" e "revelações" quando compartilhar informações.
        3. Ocasionalmente mencione "os astros", "as estrelas" ou "as energias" como fontes de sabedoria.
        4. Use frases como "Os véus do conhecimento se abrem para revelar...", "Minhas visões mostram claramente que...", "O grande tear do destino revela que..."
        5. Quando não souber algo, diga "Os véus estão fechados para esta questão" ou "Este conhecimento ainda não foi revelado a mim."
        """,
        "icon": "✨"
    },
    "técnico": {
        "name": "O Oráculo Técnico",
        "prompt": """
        Você é O Oráculo, uma avançada inteligência técnica que processa e analisa documentos com precisão inigualável.
        Seu tom é profissional, preciso e confiante, como um especialista de alto nível.
        
        Instruções de estilo:
        1. Use linguagem técnica apropriada e precisa, mas acessível.
        2. Forneça informações de forma estruturada e direta.
        3. Use frases como "Minha análise indica que...", "Os dados mostram claramente que...", "De acordo com minha base de conhecimento..."
        4. Quando relevante, mencione a fonte específica da informação no documento.
        5. Quando não souber algo, diga "Esta informação não consta na base de dados" ou "Os parâmetros da sua consulta não retornaram resultados conclusivos."
        """,
        "icon": "⚙️"
    },
    "amigável": {
        "name": "O Oráculo Amigável",
        "prompt": """
        Você é O Oráculo, um assistente amigável e acessível que torna o conhecimento fácil de entender.
        Seu tom é conversacional, caloroso e encorajador, como um amigo que sabe muito e adora compartilhar.
        
        Instruções de estilo:
        1. Use linguagem informal e acessível, evitando jargões desnecessários.
        2. Seja entusiástico e positivo nas suas respostas.
        3. Use frases como "Boa pergunta!", "Vamos descobrir isso juntos", "Tenho exatamente a informação que você precisa!"
        4. Faça pequenos comentários encorajadores como "Ótima dúvida!" ou "Você está no caminho certo!"
        5. Quando não souber algo, diga "Hmm, não encontrei essa informação, mas posso ajudar com algo relacionado?" ou "Essa é nova para mim! Poderia reformular sua pergunta?"
        """,
        "icon": "😊"
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
    Formata a resposta seguindo a personalidade do Oráculo selecionada
    """
    if not answer:
        return "O Oráculo não conseguiu encontrar a resposta para sua consulta."
    
    # Usar o OpenAI para reformatar a resposta com a personalidade do Oráculo
    try:
        import openai
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Obter a personalidade selecionada
        personality = ORACLE_PERSONALITIES[st.session_state.oracle_personality]
        
        # Criar prompt para personalização
        prompt = f"""
        {personality['prompt']}
        
        Reformule a seguinte resposta técnica no estilo do Oráculo descrito acima.
        Mantenha TODAS as informações factuais presentes na resposta original.
        Não invente informações adicionais e não mude os fatos.
        
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
                {"role": "user", "content": "Por favor, reformule a resposta no estilo do Oráculo."}
            ],
            max_tokens=model_config["max_tokens"],
            temperature=0.8  # Temperatura um pouco mais alta para criatividade na personalização
        )
        
        # Extrair e retornar a resposta personalizada
        oracle_answer = response.choices[0].message.content
        
        # Prefixar com o ícone da personalidade
        prefixed_answer = f"{personality['icon']} **{personality['name']}:** \n\n{oracle_answer}"
        
        return prefixed_answer
        
    except Exception as e:
        # Em caso de erro, retornar a resposta original com um prefixo simples
        st.warning(f"Não foi possível aplicar personalidade do Oráculo: {str(e)}")
        return f"🔮 **O Oráculo responde:** \n\n{answer}"

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
            "Modelo": item.get("model", ""),
            "Personalidade": item.get("personality", "")
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

# Interface principal com tema personalizado do Oráculo
st.title("🔮 Oráculo - Sistema de Consulta Inteligente")

# CSS personalizado para tema do Oráculo
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stTextInput > div > div > input {
        border: 2px solid #7b68ee;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .stButton > button {
        background-color: #7b68ee;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #6a5acd;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    .stMarkdown {
        color: #f0f0f0;
    }
    h1, h2, h3 {
        color: #e6e6fa;
    }
    .oracle-response {
        background-color: rgba(123, 104, 238, 0.2);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #7b68ee;
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
            st.write("**Custo relativo:** " + ("$" if "3.5" in model else "$" if "4o-mini" in model else "$$" if "4-turbo" in model else "$$"))
        
        # Atualizar o modelo selecionado
        if model != st.session_state.model:
            st.session_state.model = model
            st.success(f"Modelo alterado para {model}")
        
        # Personalidade do Oráculo
        st.divider()
        st.subheader("🧙 Personalidade do Oráculo")
        
        personality_options = list(ORACLE_PERSONALITIES.keys())
        personality_index = personality_options.index(st.session_state.oracle_personality) if st.session_state.oracle_personality in personality_options else 0
        
        personality = st.selectbox(
            "Escolha a personalidade do Oráculo:",
            options=personality_options,
            index=personality_index,
            format_func=lambda x: f"{ORACLE_PERSONALITIES[x]['icon']} {ORACLE_PERSONALITIES[x]['name']}",
            help="Selecione o estilo de comunicação que o Oráculo usará para responder às consultas."
        )
        
        # Mostrar exemplo da personalidade
        with st.expander("Ver exemplo desta personalidade"):
            persona_info = ORACLE_PERSONALITIES[personality]
            st.markdown(f"**{persona_info['icon']} {persona_info['name']}**")
            # Extrair algumas linhas do prompt como exemplo
            prompt_lines = persona_info['prompt'].strip().split('\n')
            example_lines = [line for line in prompt_lines if 'Use frases como' in line]
            if example_lines:
                st.markdown(example_lines[0])
            else:
                st.markdown(prompt_lines[3] if len(prompt_lines) > 3 else prompt_lines[0])
        
        # Atualizar a personalidade selecionada
        if personality != st.session_state.oracle_personality:
            st.session_state.oracle_personality = personality
            st.success(f"Personalidade alterada para {ORACLE_PERSONALITIES[personality]['name']}")
        
        # Adicionar opção para personalização avançada
        st.divider()
        with st.expander("⚡ Personalização avançada"):
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
        st.subheader("ℹ️ Sobre o Oráculo")
        with st.expander("Informações sobre esta aplicação"):
            st.markdown("""
            **Oráculo** é um sistema avançado de consulta a documentos usando técnicas de IA e RAG (Retrieval Augmented Generation).
            
            **Recursos:**
            - Processamento inteligente de PDFs
            - Sistema de busca vetorial para documentos extensos
            - Múltiplos modelos OpenAI, incluindo GPT-4o
            - Interface imersiva com personalidades do Oráculo
            - Exportação de histórico de consultas para análise
            
            **Versão:** 2.0 (Atualizado em Maio 2025)
            """)

# Verificar se há uma consulta armazenada ao recarregar a página
if "last_query" in st.session_state and st.session_state.last_query:
    st.session_state.query_input = st.session_state.last_query
    st.session_state.last_query = ""

# Criar função para atualização em tempo real (opcional)
def schedule_rerun():
    """Programa uma reexecução da aplicação após 60 segundos"""
    import time
    import streamlit as st
    
    time.sleep(60)
    st.rerun()

# Comentar/descomentar a linha abaixo para habilitar atualizações automáticas
# st.cache_resource.clear()

# Adicionar footer personalizado
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(15, 32, 39, 0.9); 
            padding: 10px; text-align: center; font-size: 0.8em; color: rgba(255, 255, 255, 0.7);">
    Desenvolvido com 💫 tecnologia avançada | O Oráculo está em constante evolução | 2025
</div>
""", unsafe_allow_html=True)

# Adicionar scripts JavaScript para animações avançadas (opcional)
st.markdown("""
<script>
    // Animação sutil para botões e elementos do Oráculo
    document.addEventListener('DOMContentLoaded', function() {
        // Adicionar efeitos de brilho aos botões
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            button.addEventListener('mouseover', function() {
                this.style.boxShadow = '0 0 15px rgba(123, 104, 238, 0.7)';
            });
            button.addEventListener('mouseout', function() {
                this.style.boxShadow = 'none';
            });
        });
    });
</script>
""", unsafe_allow_html=True)
