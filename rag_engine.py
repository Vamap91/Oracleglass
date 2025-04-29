# Aqui a Magica Acontece, entrando com RAG na operação
# rag_engine.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import string
from collections import Counter

class RAGEngine:
    def __init__(self):
        """
        Inicializa o motor RAG com abordagem otimizada para extração de informações estruturadas.
        """
        self.chunks = []
        self.index_path = "data/chunks_index.pkl"
        self.chunks_path = "data/document_chunks.pkl"
        self.processed_doc = None
        
        # Palavras de parada (stopwords) em português
        self.stopwords = {"a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", "com", "como", 
                          "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do", "dos", "e", "ela", "elas", 
                          "ele", "eles", "em", "entre", "era", "eram", "éramos", "essa", "essas", "esse", "esses", "esta", 
                          "estas", "este", "estes", "eu", "foi", "fomos", "for", "foram", "fossem", "há", "isso", "isto", 
                          "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", 
                          "na", "nas", "não", "no", "nos", "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", 
                          "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem", 
                          "são", "se", "seja", "sem", "seu", "seus", "só", "somos", "sua", "suas", "também", "te", 
                          "tem", "temos", "tenho", "teu", "teus", "tu", "tua", "tuas", "um", "uma", "você", "vocês", "vos"}
        
        # Termos importantes para busca de informações específicas
        self.important_terms = {
            "telefone": ["telefone", "tel", "tel.", "contato", "fone", "telefônico", "número", "ligar"],
            "nome": ["nome", "responsável", "representante", "gerente", "contato", "pessoa"],
            "responsável": ["responsável", "responsavel", "representante", "gerente", "encarregado", "supervisor"],
            "comercial": ["comercial", "comerciais", "vendas", "venda", "negócios", "atendimento", "cliente", "clientes"],
            "cdf": ["cdf", "carrefour", "empresa", "loja", "organização", "instituição", "companhia"],
            "contato": ["contato", "comunicação", "comunicar", "comunicar-se", "contatar", "falar", "ligar"],
            "undercar": ["undercar", "under", "car", "carroceria", "inferior", "embaixo", "chassi"]
        }
        
        # Criar pasta data se não existir
        if not os.path.exists("data"):
            os.makedirs("data")
    
    def tokenize_and_normalize(self, text):
        """
        Tokeniza e normaliza o texto removendo pontuação e convertendo para minúsculas.
        """
        # Remover pontuação
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Converter para minúsculas e dividir em tokens
        tokens = text.lower().split()
        
        # Remover stopwords
        tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_sections(self, text):
        """
        Extrai seções específicas do texto que podem conter informações importantes.
        """
        sections = []
        
        # Padrões de informações importantes
        patterns = [
            # Telefones
            r'\b(?:telefone|tel|fone|contato)(?:[:\s]+)([0-9\-\(\)\s\.]{7,})',
            # Emails 
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Nomes de pessoas (padrão aproximado)
            r'(?:responsável|responsavel|gerente|representante)\s*(?:comercial|de vendas|de atendimento)?[\s:]+([A-Z][a-z]+\s+(?:[A-Z][a-z]+\s*)+)',
            # Empresas ou entidades iniciando com maiúscula
            r'\b((?:[A-Z][a-z]*\s*)+)\s*[-–]\s*',
            # Padrões de formulário
            r'(?:^|\n)([A-Z][a-zA-Z\s]+):\s*([^\n]+)',
        ]
        
        # Buscar todos os padrões no texto
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                # Extrair contexto ao redor do match (50 caracteres antes e depois)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                section = text[start:end].strip()
                sections.append(section)
        
        return sections
    
    def preprocess_text(self, text, chunk_size=500, overlap=100):
        """
        Processa o texto em chunks com atenção especial a informações estruturadas.
        """
        chunks = []
        extracted_sections = self.extract_sections(text)
        
        # Adicionar todas as seções extraídas como chunks especiais
        for section in extracted_sections:
            chunks.append(("SECTION", section))
        
        # Processar o texto completo em chunks por página
        pages = []
        current_page = ""
        for line in text.split("\n"):
            if line.startswith("--- Página "):
                if current_page:
                    pages.append(current_page)
                current_page = line + "\n"
            else:
                current_page += line + "\n"
        
        if current_page:
            pages.append(current_page)
        
        # Processar cada página
        for page_num, page in enumerate(pages):
            # Dividir página em parágrafos
            paragraphs = page.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    if current_chunk:
                        chunks.append(("PAGE", current_chunk, page_num))
                    
                    # Se o parágrafo for muito grande, dividi-lo
                    if len(para) > chunk_size:
                        sentences = para.replace(". ", ".\n").split("\n")
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) <= chunk_size:
                                if current_chunk:
                                    current_chunk += " " + sentence
                                else:
                                    current_chunk = sentence
                            else:
                                chunks.append(("PAGE", current_chunk, page_num))
                                current_chunk = sentence
                    else:
                        current_chunk = para
            
            if current_chunk:
                chunks.append(("PAGE", current_chunk, page_num))
        
        # Processar os chunks para indexação
        processed_chunks = []
        for chunk_data in chunks:
            chunk_type = chunk_data[0]
            text = chunk_data[1] if chunk_type == "SECTION" else chunk_data[1]
            
            # Extrair termos e calcular frequências
            tokens = self.tokenize_and_normalize(text)
            term_freq = Counter(tokens)
            
            # Para cada termo importante, verificar sinônimos
            important_matches = {}
            for category, terms in self.important_terms.items():
                for term in terms:
                    if term in text.lower():
                        if category not in important_matches:
                            important_matches[category] = 0
                        important_matches[category] += 1
            
            processed_chunks.append({
                "text": text,
                "type": chunk_type,
                "tokens": tokens,
                "term_freq": term_freq,
                "important_matches": important_matches,
                "page": chunk_data[2] if chunk_type == "PAGE" else None
            })
        
        return processed_chunks
    
    def create_index(self, text):
        """
        Cria um índice de busca a partir do texto do documento.
        """
        self.processed_doc = text
        self.chunks = self.preprocess_text(text)
        print(f"Documento processado em {len(self.chunks)} chunks")
        
        # Salvar os chunks para uso futuro
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Criar índice invertido simples
        inverted_index = {}
        for i, chunk in enumerate(self.chunks):
            for term in chunk["term_freq"]:
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((i, chunk["term_freq"][term]))
        
        # Salvar o índice
        with open(self.index_path, 'wb') as f:
            pickle.dump(inverted_index, f)
        
        print(f"Índice criado com {len(inverted_index)} termos")
        return True
    
    def load_index(self):
        """
        Carrega o índice e chunks salvos.
        """
        try:
            if os.path.exists(self.chunks_path) and os.path.exists(self.index_path):
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"Chunks carregados: {len(self.chunks)}")
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar índice: {e}")
            return False
    
    def expand_query(self, query):
        """
        Expande a consulta com termos relacionados.
        """
        query_tokens = self.tokenize_and_normalize(query)
        expanded_tokens = set(query_tokens)
        
        # Adicionar sinônimos e termos relacionados
        for token in query_tokens:
            for category, terms in self.important_terms.items():
                if token in terms:
                    expanded_tokens.update(terms)
        
        return list(expanded_tokens)
    
    def search(self, query, top_k=7):
        """
        Realiza busca avançada combinando correspondência de termos e análise de contexto.
        """
        if not self.chunks:
            raise ValueError("Os chunks não foram carregados")
        
        # Expandir query com termos relacionados
        expanded_query = self.expand_query(query)
        
        # Primeira etapa: buscar chunks por termos importantes
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            score = 0
            
            # Verificar correspondências de termos importantes
            for term in expanded_query:
                if term in chunk["tokens"]:
                    # Dar peso maior para termos da query original
                    weight = 3 if term in query.lower() else 1
                    score += chunk["term_freq"].get(term, 0) * weight
            
            # Dar peso extra para chunks do tipo SECTION (informações estruturadas)
            if chunk["type"] == "SECTION":
                score *= 2
            
            # Verificar correspondências de categorias importantes
            query_categories = set()
            for term in expanded_query:
                for category, terms in self.important_terms.items():
                    if term in terms:
                        query_categories.add(category)
            
            # Aumentar pontuação para chunks com categorias importantes
            for category in query_categories:
                if category in chunk["important_matches"]:
                    score += chunk["important_matches"][category] * 5
            
            # Adicionar à lista se tiver pontuação positiva
            if score > 0:
                chunk_scores.append((i, chunk["text"], score))
        
        # Se não encontrou nada relevante, buscar por similaridade aproximada
        if not chunk_scores:
            query_str = " ".join(expanded_query)
            for i, chunk in enumerate(self.chunks):
                chunk_str = " ".join(chunk["tokens"])
                
                # Calcular similaridade simples
                common_words = set(expanded_query) & set(chunk["tokens"])
                if common_words:
                    score = len(common_words) / (len(expanded_query) + len(chunk["tokens"]))
                    chunk_scores.append((i, chunk["text"], score))
        
        # Se ainda não encontrou nada, retornar algumas seções
        if not chunk_scores:
            for i, chunk in enumerate(self.chunks):
                if chunk["type"] == "SECTION":
                    chunk_scores.append((i, chunk["text"], 0.1))
                    if len(chunk_scores) >= top_k:
                        break
        
        # Ordenar por relevância
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Remover duplicatas mantendo a ordem
        seen = set()
        unique_chunks = []
        for idx, text, score in chunk_scores:
            if text not in seen:
                seen.add(text)
                unique_chunks.append((idx, text, score))
        
        # Retornar os top_k mais relevantes
        return unique_chunks[:top_k]
    
    def query_with_context(self, client, query, model, system_prompt, top_k=7):
        """
        Processa consulta com contexto aprimorado para maior precisão.
        """
        # Buscar chunks relevantes
        relevant_chunks = self.search(query, top_k)
        
        # Verificar se é uma busca por informações específicas
        specific_info_query = False
        specific_categories = []
        
        for category, terms in self.important_terms.items():
            for term in terms:
                if term in query.lower():
                    specific_info_query = True
                    if category not in specific_categories:
                        specific_categories.append(category)
        
        # Preparar contexto diferenciado baseado no tipo de consulta
        if specific_info_query:
            # Para consultas de informações específicas, estruturar o contexto
            context_parts = []
            
            # Extrair e adicionar informações estruturadas primeiro
            for _, chunk_text, _ in relevant_chunks:
                # Para cada categoria específica na consulta, extrair informações relevantes
                for category in specific_categories:
                    # Usar regex para extrair informações baseadas na categoria
                    if category == "telefone":
                        matches = re.finditer(r'(?:telefone|tel|fone|contato)(?:[:\s]+)([0-9\-\(\)\s\.]{7,})', chunk_text, re.IGNORECASE)
                        for match in matches:
                            context_parts.append(f"Telefone encontrado: {match.group(0)}")
                    
                    elif category in ["nome", "responsável", "comercial"]:
                        matches = re.finditer(r'(?:responsável|responsavel|gerente|representante)\s*(?:comercial|de vendas|de atendimento)?[\s:]+([A-Z][a-z]+\s+(?:[A-Z][a-z]+\s*)+)', chunk_text, re.IGNORECASE)
                        for match in matches:
                            context_parts.append(f"Responsável comercial: {match.group(0)}")
                    
                    elif category == "cdf":
                        matches = re.finditer(r'\b(CDF|Carrefour|CDF\s*-\s*Carrefour)\b', chunk_text, re.IGNORECASE)
                        for match in matches:
                            context_parts.append(f"Empresa: {match.group(0)}")
            
            # Adicionar chunks completos após as informações estruturadas
            for _, chunk_text, _ in relevant_chunks:
                context_parts.append(f"Trecho do documento:\n{chunk_text}")
            
            context = "\n\n".join(context_parts)
        else:
            # Para consultas gerais, usar contexto simples
            context = "\n\n---\n\n".join([chunk_text for _, chunk_text, _ in relevant_chunks])
        
        # Construir prompt para a IA
        user_prompt = f"""Com base EXCLUSIVAMENTE nos seguintes trechos do documento, responda à pergunta: '{query}'

Trechos do documento:
{context}

Responda apenas com base nas informações contidas nesses trechos. Se a informação não estiver presente nos trechos fornecidos, diga claramente que não foi possível encontrar a informação.
"""
        
        # Para buscas específicas, adicionar instruções especiais
        if specific_info_query:
            categories_str = ", ".join(specific_categories)
            user_prompt += f"\n\nObservação: Esta pergunta busca informações específicas sobre {categories_str}. Caso encontre essas informações, forneça-as de forma clara e direta."
        
        # Realizar a consulta à API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro ao processar consulta: {str(e)}"
    
    def get_chunks_stats(self):
        """
        Retorna estatísticas sobre os chunks.
        """
        if not self.chunks:
            return {"error": "Nenhum chunk disponível"}
        
        section_chunks = sum(1 for chunk in self.chunks if chunk["type"] == "SECTION")
        page_chunks = sum(1 for chunk in self.chunks if chunk["type"] == "PAGE")
        
        return {
            "total_chunks": len(self.chunks),
            "section_chunks": section_chunks,
            "page_chunks": page_chunks,
            "important_terms": len(self.important_terms)
        }
