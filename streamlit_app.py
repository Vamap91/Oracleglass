import streamlit as st
import os
import pypdf
from io import BytesIO
import time
from datetime import datetime
import base64
import pandas as pd
import tiktoken

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
        import tiktoken
        VALIDATION_MESSAGES["dependencies"] = "OK"
    except ImportError as e:
        VALIDATION_OK = False
        VALIDATION_MESSAGES["dependencies"] = f"Dependência faltando: {str(e)}"
    
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

def estimate_tokens(text, model="gpt-3.5-turbo"):
    """Estima o número de tokens em um texto"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Estimativa simples se tiktoken falhar
        return len(text) // 4

def split_text_into_chunks(text, max_chunk_tokens=6000, overlap=500):
    """Divide o texto em chunks menores com sobreposição"""
    # Dividir por páginas primeiro (respeitando os marcadores "--- Página X ---")
    pages = []
    current_page = ""
    lines = text.split("\n")
    
    for line in lines:
        if line.startswith("--- Página "):
            if current_page:
                pages.append(current_page)
            current_page = line + "\n"
        else:
            current_page += line + "\n"
    
    if current_page:
        pages.append(current_page)
    
    # Agora dividir as páginas em chunks se necessário
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    
    for page in pages:
        page_tokens = estimate_tokens(page)
        
        # Se a página inteira couber no chunk atual
        if current_chunk_tokens + page_tokens <= max_chunk_tokens:
            current_chunk += page
            current_chunk_tokens += page_tokens
        # Se a página for maior que o limite por si só
        elif page_tokens > max_chunk_tokens:
            # Dividir a página em parágrafos
            paragraphs = page.split("\n\n")
            for para in paragraphs:
                para_tokens = estimate_tokens(para)
                
                # Se o parágrafo couber no chunk atual
                if current_chunk_tokens + para_tokens <= max_chunk_tokens:
                    current_chunk += para + "\n\n"
                    current_chunk_tokens += para_tokens
                # Se o parágrafo for muito grande
                else:
                    # Se tivermos conteúdo no chunk atual, salve-o
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Se o parágrafo for maior que o limite, divida-o em sentenças
                    if para_tokens > max_chunk_tokens:
                        sentences = para.replace(". ", ".\n").split("\n")
                        current_chunk = ""
                        current_chunk_tokens = 0
                        
                        for sentence in sentences:
                            sentence_tokens = estimate_tokens(sentence)
                            if current_chunk_tokens + sentence_tokens <= max_chunk_tokens:
                                current_chunk += sentence + " "
                                current_chunk_tokens += sentence_tokens
                            else:
                                chunks.append(current_chunk)
                                current_chunk = sentence + " "
                                current_chunk_tokens = sentence_tokens
                    else:
                        current_chunk = para + "\n\n"
                        current_chunk_tokens = para_tokens
        else:
            # Salvar o chunk atual e começar um novo com esta página
            chunks.append(current_chunk)
            current_chunk = page
            current_chunk_tokens = page_tokens
    
    # Adicionar o último chunk se tiver conteúdo
    if current_chunk:
        chunks.append(current_chunk)
    
    # Adicionar sobreposição para garantir continuidade
    if overlap > 0 and len(chunks) > 1:
        chunks_with_overlap = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            this_chunk = chunks[i]
            
            # Adicionar as últimas linhas do chunk anterior
            prev_lines = prev_chunk.split("\n")
            overlap_text = "\n".join(prev_lines[-min(len(prev_lines), overlap//10):]) + "\n"
            
            chunks_with_overlap.append(overlap_text + this_chunk)
        
        chunks = chunks_with_overlap
    
    return chunks

def query_ai(query):
    """
    Processa uma consulta usando a API da OpenAI com suporte a documentos grandes
    através de chunking (divisão em partes menores).
    """
    try:
        # Importar OpenAI dentro da função para evitar erros de escopo
        import openai
        
        # Cliente OpenAI para v1.0+
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Obter o texto completo do PDF
        full_text = st.session_state.pdf_text
        
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
        7. Você receberá o documento em partes. Considere todas as partes ao formular sua resposta final.
        """
        
        # Calcula o limite de tokens com base no modelo
        model_max_tokens = {
            "gpt-3.5-turbo": 4000,
            "gpt-3.5-turbo-16k": 16000,
            "gpt-4": 8000,
            "gpt-4-32k": 32000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        
        max_context_tokens = model_max_tokens.get(st.session_state.model, 4000)
        max_chunk_tokens = max(max_context_tokens // 2, 2000)  # Use half of the model's capacity for each chunk
        
        # Estimar tokens no texto completo
        total_tokens = estimate_tokens(full_text, st.session_state.model)
        
        # Tentar primeiro com o documento completo se for pequeno o suficiente
        if total_tokens < max_context_tokens * 0.7:  # Deixar margem de 30%
            st.session_state.processing_status = "Processando documento completo..."
            try:
                response = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Com base EXCLUSIVAMENTE no seguinte documento, responda à pergunta: '{query}'\n\nConteúdo do Documento:\n{full_text}"}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            except openai.BadRequestError as single_error:
                if "context_length_exceeded" not in str(single_error):
                    raise single_error
                # Se der erro de contexto, continuar com chunking
                st.info("O documento é grande demais para processar de uma só vez. Dividindo em partes menores...")
        else:
            st.info(f"Documento grande detectado ({total_tokens} tokens estimados). Dividindo em partes menores para processamento...")
        
        # Dividir o texto em chunks
        chunks = split_text_into_chunks(full_text, max_chunk_tokens=max_chunk_tokens)
        
        st.session_state.processing_status = f"Documento dividido em {len(chunks)} partes para processamento."
        
        # Lista para armazenar os resultados parciais
        partial_results = []
        
        # Processar cada chunk
        for i, chunk in enumerate(chunks):
            st.session_state.processing_status = f"Processando parte {i+1} de {len(chunks)}..."
            
            try:
                # Chamada da API para cada chunk
                partial_response = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Com base EXCLUSIVAMENTE na seguinte PARTE {i+1} de {len(chunks)} do documento, analise se há informações relevantes para responder à pergunta: '{query}'\n\nSe encontrar informações relevantes, forneça-as. Se não encontrar, apenas diga 'Não encontrei informações relevantes nesta parte'. Não faça suposições ou use conhecimento externo.\n\nConteúdo da Parte {i+1}:\n{chunk}"}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                
                result = partial_response.choices[0].message.content
                
                # Ignorar resultados sem informações relevantes
                if "Não encontrei informações relevantes nesta parte" not in result:
                    partial_results.append({
                        "part": i+1,
                        "content": result
                    })
                    
            except Exception as chunk_error:
                st.warning(f"Erro ao processar parte {i+1}. Continuando com as próximas partes. Erro: {str(chunk_error)}")
        
        # Se não encontrou informações relevantes em nenhuma parte
        if not partial_results:
            st.session_state.processing_status = "Concluído. Nenhuma informação relevante encontrada."
            return f"Após analisar todas as {len(chunks)} partes do documento, não encontrei informações relevantes para responder à pergunta: '{query}'. A informação solicitada não parece estar presente no documento fornecido."
        
        # Combinar os resultados parciais para uma resposta final
        st.session_state.processing_status = "Sintetizando resultados das partes analisadas..."
        
        synthesis_prompt = f"""
        Eu analisei um documento em {len(chunks)} partes diferentes procurando informações para responder à pergunta: '{query}'
        
        Encontrei informações relevantes nas seguintes partes do documento:
        
        {'\n\n'.join([f"PARTE {r['part']}:\n{r['content']}" for r in partial_results])}
        
        Com base APENAS nestas informações encontradas no documento, forneça uma resposta completa, coerente e concisa para a pergunta original. Se houver conflitos ou ambiguidades nas diferentes partes, mencione-os. Se as informações encontradas forem insuficientes, indique isso claramente.
        """
        
        # Chamada final da API para sintetizar os resultados
        try:
            final_response = client.chat.completions.create(
                model=st.session_state.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            st.session_state.processing_status = "Concluído com sucesso."
            return final_response.choices[0].message.content
            
        except Exception as synthesis_error:
            # Se a síntese falhar, retornar os resultados parciais formatados
            st.session_state.processing_status = "Erro na síntese final. Retornando resultados parciais."
            combined_results = f"Encontrei as seguintes informações relevantes no documento para responder à pergunta '{query}':\n\n"
            for r in partial_results:
                combined_results += f"--- Da parte {r['part']} do documento ---\n{r['content']}\n\n"
            
            return combined_results

    except openai.BadRequestError as e:
        # Capturar erro específico de contexto muito longo
        if "context_length_exceeded" in str(e):
            st.error(f"Erro: O documento ainda é muito longo mesmo após a divisão em partes. Tente usar um documento menor ou selecionar um modelo com maior capacidade de contexto.")
            st.session_state.processing_status = "Erro: documento muito grande."
            return "Erro: O documento excede a capacidade de processamento mesmo com a técnica de divisão em partes."
        else:
            st.error(f"Erro na API OpenAI: {str(e)}")
            st.session_state.processing_status = f"Erro na API: {str(e)}"
            return f"Ocorreu um erro ao processar sua consulta com a OpenAI: {str(e)}"
            
    except Exception as e:
        st.error(f"Erro inesperado ao processar consulta: {str(e)}")
        st.session_state.processing_status = f"Erro inesperado: {str(e)}"
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
        
        # Diagnóstico de chunking
        st.subheader("Diagnóstico de Chunking")
        if st.button("Testar Divisão em Chunks"):
            total_tokens = estimate_tokens(st.session_state.pdf_text)
            st.write(f"Tamanho estimado do documento: {total_tokens} tokens")
            
            # Testar divisão com diferentes limites
            for max_tokens in [2000, 4000, 6000]:
                chunks = split_text_into_chunks(st.session_state.pdf_text, max_chunk_tokens=max_tokens)
                st.write(f"Com limite de {max_tokens} tokens por chunk: {len(chunks)} chunks gerados")
                
                # Mostrar amostra do primeiro chunk
                if chunks:
                    with st.expander(f"Amostra do primeiro chunk ({estimate_tokens(chunks[0])} tokens)"):
                        st.text(chunks[0][:500] + "...")
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
    st.write(f"{dep_status} **Dependências**: {VALIDATION_MESSAGES.get('dependencies')}")
    
    # Informações do PDF
    if VALIDATION_OK:
        st.divider()
        st.subheader("📄 Informações do PDF")
        st.write(f"Arquivo: {os.path.basename(PDF_PATH)}")
        st.write(f"Tamanho do texto: {len(st.session_state.pdf_text)} caracteres")
        st.write(f"Tokens estimados: {estimate_tokens(st.session_state.pdf_text)}")
        
        # Configurações de modelo - ATUALIZADO COM MAIS OPÇÕES
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

# Botão de consulta
consult_button = st.button("🔍 Consultar", key="query_button", disabled=not VALIDATION_OK)
if consult_button and query:
    st.session_state.processing_status = "Iniciando processamento..."
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
