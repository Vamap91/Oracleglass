# Aqui a Magica Acontece, entrando com RAG na operação
# rag_engine.py
import os
import pickle
import re
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional

class RAGEngine:
    def __init__(self):
        """
        Motor RAG otimizado para documentação técnica de seguros automotivos.
        Implementa reconhecimento de entidades, chunking semântico e busca contextual.
        """
        self.chunks = []
        self.entities = {}  # Dicionário de entidades (seguradoras, assistências)
        self.entity_chunks = defaultdict(list)  # Chunks por entidade
        self.category_chunks = defaultdict(list)  # Chunks por categoria
        self.index_path = "data/chunks_index.pkl"
        self.chunks_path = "data/document_chunks.pkl"
        self.entities_path = "data/entities.pkl"
        
        # Lista de entidades (seguradoras/assistências) extraídas do documento
        self.known_entities = set([
            "ald comfort", "carbank", "assístia automob", "cdf", "carrefour", 
            "ezze seguros", "bradesco", "assistência", "automob"
        ])
        
        # Stopwords em português
        self.stopwords = {
            "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", 
            "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", 
            "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", 
            "eram", "éramos", "essa", "essas", "esse", "esses", "esta", "estas", "este", 
            "estes", "eu", "foi", "fomos", "for", "foram", "fossem", "há", "isso", "isto", 
            "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", 
            "minhas", "muito", "na", "nas", "não", "no", "nos", "nós", "nossa", "nossas", 
            "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", 
            "pelo", "pelos", "por", "qual", "quando", "que", "quem", "são", "se", "seja", 
            "sem", "seu", "seus", "só", "somos", "sua", "suas", "também", "te", "tem", 
            "temos", "tenho", "teu", "teus", "tu", "tua", "tuas", "um", "uma", "você", "vocês", "vos"
        }
        
        # Termos importantes e seus sinônimos/variações
        self.important_terms = {
            "telefone": ["telefone", "tel", "tel.", "contato", "ligar", "0800", "4003", "2699", "704", "fone", "número"],
            "responsável": ["responsável", "responsavel", "representante", "gerente", "encarregado", "supervisor", "fagner", "osório", "osorio", "gabriela", "tulio", "renata", "rampazzo", "matheus"],
            "comercial": ["comercial", "comerciais", "vendas", "venda", "negócios", "atendimento", "cliente", "clientes"],
            "seguradora": ["seguradora", "seguro", "assistência", "assistencia", "automob", "comfort", "carbank", "bradesco", "ezze", "cdf", "carrefour"],
            "undercar": ["undercar", "pneus", "suspensão", "suspensao", "roda", "rodas", "pneu", "under", "car", "matheus"],
            "cobertura": ["cobertura", "coberturas", "contrato", "plano", "planos", "vigência", "vigencia", "assistência", "assistencia"],
            "fluxo": ["fluxo", "atendimento", "procedimento", "script", "passo", "etapa", "processo"],
            "exclusao": ["exclusão", "exclusoes", "exclusao", "não", "exceto", "limitações", "restrições"]
        }
        
        # Categorias de perguntas para classificação
        self.query_types = {
            "info_pessoal": ["telefone", "responsável", "nome", "contato", "email", "fone", "quem"],
            "fluxo": ["como", "procedimento", "passo", "etapa", "fluxo", "fazer", "processo", "script"],
            "cobertura": ["cobre", "cobertura", "plano", "inclui", "incluído", "valor", "limite", "máximo", "vidros", "faróis", "para-brisa"],
            "excecao": ["não cobre", "exclusão", "excluído", "limitação", "restrição", "quando não", "exceção", "reembolso"]
        }
        
        # Criar pasta data se não existir
        if not os.path.exists("data"):
            os.makedirs("data")
    
    def normalize_text(self, text: str) -> str:
        """Normaliza texto removendo acentos e convertendo para minúsculas"""
        # Remover acentos
        import unicodedata
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        return text.lower()
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokeniza e normaliza o texto removendo pontuação e stopwords"""
        # Remover pontuação
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Converter para minúsculas e dividir em tokens
        tokens = text.lower().split()
        
        # Remover stopwords
        tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extrai entidades (seguradoras/assistências) e suas informações relacionadas
        """
        entities = {}
        
        # Padrões para extrair blocos de entidades
        entity_patterns = [
            # Padrão para nomes de seguradoras/assistências
            r'(?:^|\n)([A-Z][A-Za-z\s]+(?:Seguros|Seguradora|Comfort|Assistência|Automob|Carrefour|Bradesco|Ezze|ALD|CDF))(?:\n|:|\s*-\s*)',
            # Padrão alternativo para nomes mais simples
            r'(?:^|\n)([A-Z][A-Za-z\s]{2,20})(?:\n|:|\s*-\s*)'
        ]
        
        # Extrair potenciais entidades
        potential_entities = set()
        for pattern in entity_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                entity_name = match.group(1).strip()
                potential_entities.add(entity_name.lower())
        
        # Adicionar entidades conhecidas que possam estar no texto
        for known_entity in self.known_entities:
            if known_entity in text.lower():
                potential_entities.add(known_entity)
        
        # Para cada entidade potencial, extrair informações relacionadas
        for entity_name in potential_entities:
            # Ignorar entidades muito genéricas
            if len(entity_name) < 3 or entity_name.lower() in self.stopwords:
                continue
                
            # Buscar contexto em torno da entidade
            context_pattern = r'(?:^|\n)(?:[^\n]*?' + re.escape(entity_name) + r'[^\n]*?)(?:\n|$)((?:.+?\n){0,20})'
            context_matches = re.finditer(context_pattern, text, re.IGNORECASE | re.MULTILINE)
            
            entity_info = {
                "name": entity_name,
                "telefones": [],
                "responsavel": "",
                "fluxo": "",
                "cobertura": [],
                "context": ""
            }
            
            for context_match in context_matches:
                context = context_match.group(0)
                entity_info["context"] += context + "\n\n"
                
                # Extrair telefones
                phone_matches = re.finditer(r'(?:telefone|tel|contato|fone)[^\d]*((?:\d{4,5}[-\s]?\d{4}|\d{3,4}[-\s]?\d{3,4}[-\s]?\d{4}|0800[-\s]?\d{3}[-\s]?\d{4}))', context, re.IGNORECASE)
                for phone_match in phone_matches:
                    phone = phone_match.group(1).strip()
                    if phone and phone not in entity_info["telefones"]:
                        entity_info["telefones"].append(phone)
                
                # Extrair responsável
                resp_matches = re.finditer(r'(?:responsável|responsavel|gerente)[^\n:]*[:•]\s*([A-Z][a-zÀ-ú]+\s+[A-Z][a-zÀ-ú\s]+)', context, re.IGNORECASE)
                for resp_match in resp_matches:
                    responsavel = resp_match.group(1).strip()
                    if responsavel:
                        entity_info["responsavel"] = responsavel
                
                # Extrair fluxo de atendimento
                flow_matches = re.finditer(r'(?:fluxo|procedimento|atendimento)[^\n]*(?:\n|:)((?:.+\n){1,10})', context, re.IGNORECASE)
                for flow_match in flow_matches:
                    flow = flow_match.group(1).strip()
                    if flow:
                        entity_info["fluxo"] += flow + "\n"
                
                # Extrair coberturas
                coverage_matches = re.finditer(r'(?:cobertura|plano|vidros|acessórios|undercar|pneus)[^\n]*(?:\n|:)((?:.+\n){1,10})', context, re.IGNORECASE)
                for cov_match in coverage_matches:
                    coverage = cov_match.group(1).strip()
                    if coverage:
                        entity_info["cobertura"].append(coverage)
            
            # Adicionar à lista de entidades apenas se tiver informações relevantes
            if entity_info["telefones"] or entity_info["responsavel"] or entity_info["fluxo"]:
                entities[entity_name.lower()] = entity_info
        
        return entities
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrai seções semânticas do documento baseadas em padrões de formatação e conteúdo.
        """
        sections = []
        
        # 1. Identificar cabeçalhos potenciais (em maiúsculas ou com formatação especial)
        header_patterns = [
            # Padrão para cabeçalhos em maiúsculas
            r'(?:^|\n)([A-Z][A-Z\s]{2,30})(?:\n|$)',
            # Padrão para títulos com formatação especial
            r'(?:^|\n)(?:[•★✓✦✧➢➤➥]\s*)([A-Z][A-Za-z\s]{2,30})(?::|$)',
            # Padrão para seções numeradas
            r'(?:^|\n)(?:\d+[\.\)]\s+)([A-Z][A-Za-z\s]{2,30})(?::|$)'
        ]
        
        potential_headers = []
        for pattern in header_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                header = match.group(1).strip()
                position = match.start()
                potential_headers.append((header, position))
        
        # Ordenar por posição no texto
        potential_headers.sort(key=lambda x: x[1])
        
        # 2. Extrair conteúdo entre cabeçalhos consecutivos
        for i in range(len(potential_headers)):
            header, start_pos = potential_headers[i]
            
            # Determinar o fim da seção
            if i < len(potential_headers) - 1:
                end_pos = potential_headers[i+1][1]
            else:
                end_pos = len(text)
            
            # Extrair conteúdo
            content = text[start_pos:end_pos].strip()
            
            # Categorizar a seção
            category = self.categorize_section(header, content)
            
            # Criar objeto de seção
            section = {
                "title": header,
                "category": category,
                "content": content,
                "position": start_pos,
                "entities": self.extract_section_entities(content)
            }
            
            sections.append(section)
        
        # 3. Capturar seções especiais baseadas em padrões específicos
        special_patterns = [
            # Telefones
            (r'(?:telefone|tel|contato)(?:[:\s]+)([0-9\-\(\)\s\.]{7,})', "telefone"),
            # Responsáveis
            (r'(?:responsável|responsavel)[^:]*:([^,$\n]{3,40})', "responsavel"),
            # Fluxo de atendimento
            (r'(?:fluxo de atendimento|procedimento)(?:[^$]{10,500})', "fluxo"),
            # Undercar
            (r'(?:undercar|pneus)(?:[^$]{10,500})', "undercar"),
            # Exclusões
            (r'(?:exclusões|exclusao|não cobre)(?:[^$]{10,500})', "exclusao")
        ]
        
        for pattern, category in special_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                # Extrair mais contexto ao redor do match
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 400)
                content = text[start:end].strip()
                
                # Criar seção especial
                section = {
                    "title": f"{category.upper()}",
                    "category": category,
                    "content": content,
                    "position": match.start(),
                    "entities": self.extract_section_entities(content)
                }
                
                sections.append(section)
        
        return sections
    
    def categorize_section(self, title: str, content: str) -> str:
        """
        Categoriza uma seção com base em seu título e conteúdo.
        """
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Mapear palavras-chave para categorias
        category_keywords = {
            "telefone": ["telefone", "tel", "contato", "0800", "4003", "2699", "704"],
            "responsavel": ["responsável", "responsavel", "representante", "gerente"],
            "fluxo": ["fluxo", "atendimento", "procedimento", "etapa", "script"],
            "cobertura": ["cobertura", "plano", "vidros", "acessórios", "inclui"],
            "exclusao": ["exclusão", "exclusao", "não cobre", "restrição", "limite"],
            "undercar": ["undercar", "pneus", "suspensão", "suspensao", "roda"],
            "entidade": ["seguradora", "assistência", "assistencia", "comfort", "bradesco", "ezze"],
            "valor": ["valor", "coparticipação", "coparticipacao", "preço", "custo", "vmd"]
        }
        
        # Verificar título primeiro
        for category, keywords in category_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        
        # Verificar conteúdo se o título não for conclusivo
        for category, keywords in category_keywords.items():
            if any(keyword in content_lower[:500] for keyword in keywords):
                return category
        
        # Categoria padrão se nenhuma correspondência for encontrada
        return "geral"
    
    def extract_section_entities(self, content: str) -> List[str]:
        """
        Extrai nomes de entidades (seguradoras/assistências) mencionadas no conteúdo.
        """
        entities = []
        
        # Verificar entidades conhecidas
        for entity in self.known_entities:
            if entity.lower() in content.lower():
                entities.append(entity)
        
        # Buscar por padrões de nomes de empresa
        company_patterns = [
            r'(?:^|\n|\s)([A-Z][a-zÀ-ú]+(?:\s+[A-Z][a-zÀ-ú]+){1,3})(?:\s+Seguros|\s+Seguradora|\s+Automob|\s+Comfort)',
            r'(?:empresa|seguradora|assistência):?\s+([A-Z][a-zÀ-ú]+(?:\s+[A-Z][a-zÀ-ú]+){0,3})'
        ]
        
        for pattern in company_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                entity = match.group(1).strip()
                if entity and entity.lower() not in [e.lower() for e in entities]:
                    entities.append(entity)
        
        return entities
    
    def create_semantic_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cria chunks semânticos baseados nas seções extraídas.
        """
        chunks = []
        
        # 1. Criar chunks para cada seção
        for section in sections:
            content = section["content"]
            category = section["category"]
            title = section["title"]
            
            # Se o conteúdo for muito grande, dividi-lo em partes menores
            if len(content) > 1000:
                # Dividir por parágrafos
                paragraphs = [p for p in re.split(r'\n\s*\n', content) if p.strip()]
                
                # Se não houver parágrafos claros, dividir por linhas
                if not paragraphs or len(paragraphs) == 1:
                    paragraphs = [p for p in content.split('\n') if p.strip()]
                
                # Criar chunks a partir dos parágrafos
                current_chunk = title + "\n"
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= 1000:
                        current_chunk += para + "\n\n"
                    else:
                        # Finalizar chunk atual
                        chunk = self.create_chunk_object(current_chunk, title, category, section["entities"])
                        chunks.append(chunk)
                        
                        # Iniciar novo chunk
                        current_chunk = title + "\n" + para + "\n\n"
                
                # Adicionar último chunk se não estiver vazio
                if current_chunk.strip() != title:
                    chunk = self.create_chunk_object(current_chunk, title, category, section["entities"])
                    chunks.append(chunk)
            else:
                # Para seções menores, criar um único chunk
                chunk = self.create_chunk_object(content, title, category, section["entities"])
                chunks.append(chunk)
        
        # 2. Criar chunks especiais para entidades
        for entity_name, entity_info in self.entities.items():
            # Chunk com todas as informações da entidade
            entity_content = f"{entity_info['name'].upper()}\n"
            
            if entity_info["telefones"]:
                entity_content += f"Telefone(s): {', '.join(entity_info['telefones'])}\n"
            
            if entity_info["responsavel"]:
                entity_content += f"Responsável: {entity_info['responsavel']}\n"
            
            if entity_info["fluxo"]:
                entity_content += f"Fluxo de Atendimento: {entity_info['fluxo']}\n"
            
            if entity_info["cobertura"]:
                entity_content += f"Coberturas: {', '.join(entity_info['cobertura'])}\n"
            
            # Adicionar contexto completo
            entity_content += f"\nContexto completo:\n{entity_info['context']}"
            
            # Criar chunk da entidade
            chunk = self.create_chunk_object(
                entity_content, 
                entity_info['name'].upper(), 
                "entidade", 
                [entity_info['name']]
            )
            
            # Definir alta prioridade para chunks de entidade
            chunk["priority"] = 10
            chunks.append(chunk)
            
            # Chunks específicos para telefone e responsável
            if entity_info["telefones"]:
                tel_content = f"{entity_info['name'].upper()}\nTelefone(s): {', '.join(entity_info['telefones'])}"
                tel_chunk = self.create_chunk_object(tel_content, "TELEFONE", "telefone", [entity_info['name']])
                tel_chunk["priority"] = 20  # Prioridade ainda maior para informações críticas
                chunks.append(tel_chunk)
            
            if entity_info["responsavel"]:
                resp_content = f"{entity_info['name'].upper()}\nResponsável: {entity_info['responsavel']}"
                resp_chunk = self.create_chunk_object(resp_content, "RESPONSÁVEL", "responsavel", [entity_info['name']])
                resp_chunk["priority"] = 20
                chunks.append(resp_chunk)
        
        # 3. Indexar chunks por entidade e categoria para pesquisa mais eficiente
        for i, chunk in enumerate(chunks):
            # Indexar por entidade
            for entity in chunk["entities"]:
                self.entity_chunks[entity.lower()].append(i)
            
            # Indexar por categoria
            self.category_chunks[chunk["category"]].append(i)
        
        return chunks
    
    def create_chunk_object(self, content: str, title: str, category: str, entities: List[str]) -> Dict[str, Any]:
        """
        Cria um objeto de chunk estruturado com metadados.
        """
        # Tokenizar para análise semântica
        tokens = self.tokenize_text(content)
        
        # Calcular frequência de termos
        term_freq = Counter(tokens)
        
        # Verificar correspondência com termos importantes
        important_matches = {}
        for term_category, terms in self.important_terms.items():
            for term in terms:
                if term in content.lower():
                    if term_category not in important_matches:
                        important_matches[term_category] = 0
                    important_matches[term_category] += 1
        
        # Criar objeto de chunk
        chunk = {
            "text": content,
            "title": title,
            "category": category,
            "entities": entities,
            "tokens": tokens,
            "term_freq": term_freq,
            "important_matches": important_matches,
            "priority": 5  # Prioridade padrão
        }
        
        # Ajustar prioridade com base na categoria
        if category in ["telefone", "responsavel", "undercar"]:
            chunk["priority"] += 3
        
        # Ajustar prioridade com base em correspondências importantes
        if len(important_matches) > 2:
            chunk["priority"] += 2
        
        return chunk
    
    def preprocess_text(self, text: str) -> None:
        """
        Realiza o pré-processamento completo do texto do documento.
        """
        # 1. Extrair entidades (seguradoras/assistências)
        self.entities = self.extract_entities(text)
        print(f"Extraídas {len(self.entities)} entidades do documento")
        
        # 2. Extrair seções semânticas
        sections = self.extract_sections(text)
        print(f"Extraídas {len(sections)} seções do documento")
        
        # 3. Criar chunks semânticos
        self.chunks = self.create_semantic_chunks(sections)
        print(f"Criados {len(self.chunks)} chunks semânticos")
        
        # 4. Salvar resultados
        with open(self.chunks_path, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "entity_chunks": dict(self.entity_chunks),
                "category_chunks": dict(self.category_chunks)
            }, f)
        
        with open(self.entities_path, 'wb') as f:
            pickle.dump(self.entities, f)
        
        print(f"Índices e chunks salvos com sucesso!")
    
    def create_index(self, text: str) -> bool:
        """
        Cria índices a partir do texto do documento.
        """
        self.preprocess_text(text)
        return True
    
    def load_index(self) -> bool:
        """
        Carrega índices e chunks salvos.
        """
        try:
            if os.path.exists(self.chunks_path) and os.path.exists(self.entities_path):
                # Carregar chunks e índices
                with open(self.chunks_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data["chunks"]
                    self.entity_chunks = defaultdict(list, data["entity_chunks"])
                    self.category_chunks = defaultdict(list, data["category_chunks"])
                
                # Carregar entidades
                with open(self.entities_path, 'rb') as f:
                    self.entities = pickle.load(f)
                
                print(f"Carregados {len(self.chunks)} chunks e {len(self.entities)} entidades")
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar índice: {e}")
            return False
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classifica a consulta para direcionar a busca.
        """
        query_lower = query.lower()
        
        # 1. Detectar entidades mencionadas
        detected_entities = []
        for entity in self.known_entities:
            if entity.lower() in query_lower:
                detected_entities.append(entity.lower())
        
        # Buscar por entidades desconhecidas também
        for entity_name in self.entities.keys():
            if entity_name.lower() in query_lower and entity_name.lower() not in detected_entities:
                detected_entities.append(entity_name.lower())
        
        # 2. Classificar tipo de consulta
        query_type = "geral"
        for q_type, keywords in self.query_types.items():
            if any(keyword in query_lower for keyword in keywords):
                query_type = q_type
                break
        
        # 3. Detectar termos importantes
        important_categories = []
        for category, terms in self.important_terms.items():
            if any(term in query_lower for term in terms):
                important_categories.append(category)
        
        # 4. Verificar menção específica a "undercar"
        has_undercar = "undercar" in query_lower or any(term in query_lower for term in self.important_terms["undercar"])
        
        return {
            "entities": detected_entities,
            "type": query_type,
            "important_categories": important_categories,
            "has_undercar": has_undercar
        }
    
    def expand_query(self, query: str) -> Set[str]:
        """
        Expande a consulta com termos relacionados.
        """
        query_lower = query.lower()
        expanded_terms = set(self.tokenize_text(query_lower))
        
        # Adicionar sinônimos e termos relacionados
        for term in list(expanded_terms):
            for category, terms in self.important_terms.items():
                if term in terms:
                    expanded_terms.update(terms)
        
        # Adicionar termos importantes se mencionados na consulta
        for category, terms in self.important_terms.items():
            if any(term in query_lower for term in terms):
                expanded_terms.update(terms)
        
        return expanded_terms
    
    def search(self, query: str, top_k: int = 7) -> List[Tuple[int, str, float]]:
        """
        Realiza busca semântica guiada por entidades e categorias.
        """
        if not self.chunks:
            raise ValueError("Os chunks não foram carregados")
        
        # 1. Classificar e expandir a consulta
        query_info = self.classify_query(query)
        expanded_terms = self.expand_query(query)
        
        # 2. Preparar conjuntos de chunks a considerar
        candidate_chunks = set()
        
        # 2.1 Priorizar chunks relacionados às entidades detectadas
        for entity in query_info["entities"]:
            candidate_chunks.update(self.entity_chunks.get(entity.lower(), []))
        
        # 2.2 Considerar chunks de categorias importantes
        if query_info["important_categories"]:
            for category in query_info["important_categories"]:
                candidate_chunks.update(self.category_chunks.get(category, []))
        
        # 2.3 Busca especial para undercar
        if query_info["has_undercar"]:
            candidate_chunks.update(self.category_chunks.get("undercar", []))
        
        # 2.4 Se não houver candidatos específicos, considerar todos os chunks
        if not candidate_chunks:
            candidate_chunks = set(range(len(self.chunks)))
        
        # 3. Pontuar chunks candidatos
        chunk_scores = []
        
        for idx in candidate_chunks:
            chunk = self.chunks[idx]
            score = 0
            
            # 3.1 Pontuação base da prioridade do chunk
            score += chunk["priority"]
            
            # 3.2 Pontuação por correspondência de termos
            for term in expanded_terms:
                if term in chunk["tokens"]:
                    # Peso maior para termos da consulta original
                    weight = 3 if term in query.lower() else 1
                    score += chunk["term_freq"].get(term, 0) * weight
            
            # 3.3 Pontuação adicional para correspondências importantes
            for category, count in chunk["important_matches"].items():
                if category in query_info["important_categories"]:
                    score += count * 5
            
            # 3.4 Bônus para chunks que mencionam entidades da consulta
            for entity in query_info["entities"]:
                if entity in [e.lower() for e in chunk["entities"]]:
                    score += 15
            
            # 3.5 Bônus especial para undercar se mencionado
            if query_info["has_undercar"] and "undercar" in chunk["text"].lower():
                score += 25
            
            # Adicionar à lista de resultados se tiver pontuação positiva
            if score > 0:
                chunk_scores.append((idx, chunk["text"], score))
        
        # 4. Se não encontrou nada relevante, usar busca de fallback
        if not chunk_scores:
            # Buscar em todos os chunks pela correspondência simples de palavras
            for i, chunk in enumerate(self.chunks):
                score = 0
                chunk_text = chunk["text"].lower()
                
                # Verificar correspondência de termos expandidos
                for term in expanded_terms:
                    if term in chunk_text:
                        score += 1
                
                # Priorizar chunks de categorias específicas
                if chunk["category"] in ["telefone", "responsavel", "undercar", "fluxo"]:
                    score += 2
                
                if score > 0:
                    chunk_scores.append((i, chunk["text"], score))
        
        # 5. Se ainda não encontrou nada, pegar chunks com prioridades maiores
        if not chunk_scores:
            high_priority_chunks = [(i, chunk["text"], chunk["priority"]) 
                                   for i, chunk in enumerate(self.chunks) 
                                   if chunk["priority"] >= 5]
            chunk_scores.extend(high_priority_chunks[:top_k])
        
        # 6. Ordenar por relevância
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 7. Remover duplicatas mantendo a ordem
        seen = set()
        unique_chunks = []
        for idx, text, score in chunk_scores:
            normalized_text = text[:100]  # Usar início do texto para comparação
            if normalized_text not in seen:
                seen.add(normalized_text)
                unique_chunks.append((idx, text, score))
        
        # 8. Retornar os top_k mais relevantes
        return unique_chunks[:top_k]
    
    def query_with_context(self, client, query: str, model: str, system_prompt: str, temperature: float = 0.2, top_k: int = 7) -> str:
        """
        Processa consulta com contexto enriquecido baseado na classificação.
        
        Args:
            client: Cliente OpenAI
            query: A consulta do usuário
            model: Modelo a ser usado
            system_prompt: Prompt de sistema para instruir o modelo
            temperature: Temperatura para controlar aleatoriedade das respostas (0.0 a 1.0)
            top_k: Número de chunks relevantes a recuperar
            
        Returns:
            str: Resposta gerada pelo modelo
        """
        # 1. Classificar a consulta
        query_info = self.classify_query(query)
        
        # 2. Buscar chunks relevantes
        relevant_chunks = self.search(query, top_k)
        
        # 3. Verificar se temos resultados relevantes
        if not relevant_chunks:
            return f"Não foi possível encontrar informações específicas sobre '{query}' no documento fornecido."
        
        # 4. Preparar contexto
        context_parts = []
        
        # 4.1 Adicionar seções especiais prioritárias
        if query_info["entities"]:
            entity_str = ", ".join(query_info["entities"])
            context_parts.append(f"ENTIDADES MENCIONADAS: {entity_str}")
        
        if query_info["has_undercar"]:
            for idx, text, score in relevant_chunks:
                if "undercar" in text.lower():
                    context_parts.append(f"INFORMAÇÃO ESPECÍFICA DE UNDERCAR:\n{text}")
        
        # 4.2 Adicionar chunks relevantes em ordem de pontuação
        for idx, text, score in relevant_chunks:
            chunk = self.chunks[idx]
            # Evitar duplicação
            if not any(text in part for part in context_parts):
                if chunk["title"]:
                    formatted_text = f"SEÇÃO: {chunk['title']}\n{text}"
                else:
                    formatted_text = text
                context_parts.append(formatted_text)
        
        # 5. Construir o contexto final
        context = "\n\n---\n\n".join(context_parts)
        
        # 6. Enriquecer o prompt do sistema para focar na consulta
        enriched_system_prompt = system_prompt
        
        if query_info["entities"]:
            entity_str = ", ".join(query_info["entities"])
            enriched_system_prompt += f"\n\nA consulta se refere especificamente a: {entity_str}. Priorize informações relacionadas a esta(s) entidade(s)."
        
        if query_info["important_categories"]:
            categories_str = ", ".join(query_info["important_categories"])
            enriched_system_prompt += f"\n\nA consulta busca por informações sobre: {categories_str}. Concentre-se nessas categorias de informação."
        
        # 7. Construir prompt para a IA
        user_prompt = f"""Com base nos trechos do documento abaixo, responda à pergunta: '{query}'

Trechos do documento:
{context}

IMPORTANTE: Responda APENAS com base nas informações contidas nesses trechos. Se a informação solicitada estiver presente, forneça-a de forma clara e direta. Se não estiver presente nos trechos fornecidos, diga explicitamente que não foi possível encontrar essa informação específica no documento.
"""
        
        # 8. Realizar a consulta à API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": enriched_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,  # Usar o parâmetro de temperatura passado
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro ao processar consulta: {str(e)}"
    
    def get_chunks_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre os chunks.
        """
        if not self.chunks:
            return {"error": "Nenhum chunk disponível"}
        
        # Contar chunks por categoria
        categories = Counter(chunk["category"] for chunk in self.chunks)
        
        # Contar entidades
        entity_counts = Counter()
        for chunk in self.chunks:
            for entity in chunk["entities"]:
                entity_counts[entity] += 1
        
        # Estatísticas de prioridade
        priorities = [chunk["priority"] for chunk in self.chunks]
        
        return {
            "total_chunks": len(self.chunks),
            "categories": dict(categories),
            "entities": dict(entity_counts.most_common(10)),
            "priorities": {
                "min": min(priorities) if priorities else 0,
                "max": max(priorities) if priorities else 0,
                "avg": sum(priorities)/len(priorities) if priorities else 0
            },
            "total_entities": len(self.entities)
        }
