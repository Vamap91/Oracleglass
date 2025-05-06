import os
import pickle
import re
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self):
        """
        Motor RAG otimizado para documentação técnica de seguros automotivos.
        Implementa reconhecimento de entidades, chunking semântico e busca contextual,
        com suporte especial para perguntas frequentes.
        """
        self.chunks = []
        self.entities = {}  # Dicionário de entidades (seguradoras, assistências)
        self.entity_chunks = defaultdict(list)  # Chunks por entidade
        self.category_chunks = defaultdict(list)  # Chunks por categoria
        self.faq_chunks = defaultdict(list)  # Chunks específicos para FAQs
        self.term_chunks = defaultdict(list)  # Chunks por termo específico
        
        self.index_path = "data/chunks_index.pkl"
        self.chunks_path = "data/document_chunks.pkl"
        self.entities_path = "data/entities.pkl"
        self.faq_path = "data/faq_index.pkl"
        
        # Lista de entidades (seguradoras/assistências) extraídas do documento
        self.known_entities = set([
            "ald comfort", "carbank", "assístia automob", "cdf", "carrefour", 
            "ezze seguros", "bradesco", "assistência", "automob", "porto", "azul",
            "porto seguro", "azul seguros", "bradesco seguros", "sura", "sura bmw",
            "ald", "audi", "bbf", "continental", "c6", "helps", "santander", 
            "hyundai", "plano auto prime", "positron", "psa", "chevrolet",
            "sem parar", "volkswagen"
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
            "responsável": ["responsável", "responsavel", "representante", "gerente", "encarregado", "supervisor", "fagner", "osório", "osorio", "gabriela", "tulio", "renata", "rampazzo", "matheus", "comercial"],
            "comercial": ["comercial", "comerciais", "vendas", "venda", "negócios", "atendimento", "cliente", "clientes"],
            "seguradora": ["seguradora", "seguro", "assistência", "assistencia", "automob", "comfort", "carbank", "bradesco", "ezze", "cdf", "carrefour", "porto", "azul", "sura"],
            "undercar": ["undercar", "pneus", "suspensão", "suspensao", "roda", "rodas", "pneu", "under", "car", "matheus"],
            "cobertura": ["cobertura", "coberturas", "contrato", "plano", "planos", "vigência", "vigencia", "assistência", "assistencia", "cobre"],
            "fluxo": ["fluxo", "atendimento", "procedimento", "script", "passo", "etapa", "processo"],
            "exclusao": ["exclusão", "exclusoes", "exclusao", "não", "exceto", "limitações", "restrições"],
            # Novos termos específicos para perguntas frequentes
            "vidros": ["vidro", "vidros", "parabrisa", "para-brisa", "para brisa", "vigia", "teto solar"],
            "acessorios": ["acessório", "acessórios", "farol", "faróis", "lanterna", "lanternas", "para-choque", "parachoque"],
            "rr": ["rr", "reparo rápido", "reparo rapido"],
            "sm": ["sm", "super martelinho", "supermartelinho"],
            "rrsm": ["rrsm", "smrr", "reparo rápido super martelinho", "super martelinho reparo rápido"],
            "limite": ["limite", "limites", "máximo", "maximo", "valor máximo"],
            "prazo": ["prazo", "prazos", "sla", "liberação", "liberar", "liberado"],
            "garantia": ["garantia", "garantias"],
            "franquia": ["franquia", "valor", "custo", "preço"],
            "vmd": ["vmd", "valor mínimo", "valor minimo"],
            "credenciamento": ["credenciamento", "credenciar", "credenciada", "loja credenciada"],
            "reembolso": ["reembolso", "reembolsar", "restituição", "ordem de reembolso"]
        }
        
        # Categorias de perguntas para classificação - expandidas
        self.query_types = {
            "info_pessoal": ["telefone", "responsável", "nome", "contato", "email", "fone", "quem"],
            "fluxo": ["como", "procedimento", "passo", "etapa", "fluxo", "fazer", "processo", "script"],
            "cobertura": ["cobre", "cobertura", "plano", "inclui", "incluído", "valor", "limite", "máximo", "vidros", "faróis", "para-brisa"],
            "excecao": ["não cobre", "exclusão", "excluído", "limitação", "restrição", "quando não", "exceção", "reembolso"],
            # Novas categorias específicas para as perguntas frequentes
            "atendimento": ["atendemos", "atende", "atender", "cobrimos"],
            "valor": ["valor", "preço", "custo", "franquia", "vmd", "reembolso"],
            "prazo": ["prazo", "tempo", "sla", "liberação", "liberar"],
            "garantia": ["garantia", "garantimos", "garante"],
            "inclusao": ["inclusão", "incluir", "adicionar"],
            "exclusao": ["exclusão", "excluir", "remover"],
            "selecao": ["selecionar", "como selecionar", "localizar", "como localizar"],
            "procedimento": ["procedimento", "como fazer", "passo a passo"]
        }
        
        # Padrões específicos para reconhecimento de perguntas frequentes
        self.faq_patterns = {
            # Atendimento para seguradoras
            "atendimento_porto_vidros": [r"atendemos vidros para a porto", r"porto.*vidros", r"vidros.*porto"],
            "atendimento_porto_acessorios": [r"atendemos acessórios para a porto", r"porto.*acessórios", r"acessórios.*porto"],
            "atendimento_azul_vidros": [r"atendemos vidros para a azul", r"azul.*vidros", r"vidros.*azul"],
            "atendimento_azul_acessorios": [r"atendemos acessórios para a azul", r"azul.*acessórios", r"acessórios.*azul"],
            
            # Franquia
            "franquia_sura": [r"sura tem valor de franquia", r"sura.*franquia", r"franquia.*sura"],
            "franquia_ald": [r"ald tem valor de franquia", r"ald.*franquia", r"franquia.*ald"],
            
            # Reembolso
            "ordem_reembolso": [r"o que é ordem de reembolso", r"ordem de reembolso", r"reembolso.*ordem"],
            
            # Reparos
            "reparo_parabrisa": [r"reparo de parabrisa", r"reparo de para-brisa", r"reparo de para brisa"],
            
            # Diferenças
            "diferenca_rrsm": [r"diferença rrsm", r"diferença.*rrsm"],
            "diferenca_smrr": [r"diferença smrr", r"diferença.*smrr"],
            
            # Limites
            "limite_rrsm_porto": [r"limite rrsm porto", r"porto.*limite.*rrsm", r"rrsm.*porto.*limite"],
            "limite_sm_porto": [r"limite sm porto", r"porto.*limite.*sm", r"sm.*porto.*limite"],
            "limite_rr_porto": [r"limite rr porto", r"porto.*limite.*rr", r"rr.*porto.*limite"],
            "limite_rr_azul": [r"limite rr azul", r"azul.*limite.*rr", r"rr.*azul.*limite"],
            "limite_sm_azul": [r"limite sm azul", r"azul.*limite.*sm", r"sm.*azul.*limite"],
            
            # Procedimentos genéricos
            "proc_ordem_reembolso": [r"procedimento ordem de reembolso", r"procedimento.*reembolso"],
            "proc_comanda_manual": [r"como fazer comanda manual", r"comanda manual", r"script comanda manual"],
            
            # Seleção de peças específicas
            "selecao_parabrisa_volvo": [r"como selecionar.*parabrisa.*volvo", r"selecionar.*para.?brisa.*volvo"],
            "selecao_parabrisa_mercedes": [r"como selecionar.*parabrisa.*mercedes", r"selecionar.*para.?brisa.*mercedes"],
            
            # Coberturas específicas
            "cobertura_lanterna": [r"lanterna.*possui cobertura", r"cobertura.*lanterna", r"cobre.*lanterna"],
            "cobertura_parabrisa": [r"para.?brisa.*possui cobertura", r"cobertura.*para.?brisa", r"cobre.*para.?brisa"],
            
            # SLA e prazos
            "sla_porto": [r"sla porto", r"prazo.*porto", r"porto.*prazo"],
            "sla_azul": [r"sla azul", r"prazo.*azul", r"azul.*prazo"],
            "sla_bradesco": [r"sla bradesco", r"prazo.*bradesco", r"bradesco.*prazo"]
        }
        
        # Criar pasta data se não existir
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # Inicializar os vetores FAQ
        self.faq_vectors = {}
        self.processed_faqs = {}
        
        # Dicionário de respostas para perguntas frequentes
        self.faq_responses = {
            # Respostas para perguntas sobre atendimento
            "atendimento_porto_vidros": "Sim, atendemos vidros para a Porto Seguro. Os serviços incluem troca e reparo de para-brisas, vidros laterais e traseiros.",
            "atendimento_porto_acessorios": "Sim, atendemos acessórios para a Porto Seguro. Os serviços incluem faróis, lanternas, retrovisores e para-choques.",
            "atendimento_azul_vidros": "Sim, atendemos vidros para a Azul Seguros. Os serviços incluem troca e reparo de para-brisas, vidros laterais e traseiros.",
            "atendimento_azul_acessorios": "Sim, atendemos acessórios para a Azul Seguros. Os serviços incluem faróis, lanternas, retrovisores e para-choques.",
            
            # Respostas sobre franquia
            "franquia_sura": "Sim, a Sura possui valor de franquia para os serviços de vidros e acessórios. Os valores específicos dependem do plano contratado pelo cliente.",
            "franquia_ald": "Sim, a ALD possui valor de franquia para os serviços. Os valores variam conforme o tipo de veículo e o plano contratado.",
            
            # Respostas sobre reembolso
            "ordem_reembolso": "Ordem de reembolso é um procedimento utilizado quando o cliente realiza o serviço por conta própria e solicita o reembolso do valor à seguradora, dentro dos limites da apólice.",
            
            # Respostas sobre diferenças
            "diferenca_rrsm": "RRSM (Reparo Rápido e Super Martelinho) é uma combinação de serviços que inclui pequenos reparos de lataria e pintura (Super Martelinho) junto com serviços de reparo rápido para danos em vidros e outros componentes.",
            "diferenca_smrr": "SMRR (Super Martelinho e Reparo Rápido) é o mesmo que RRSM, apenas com a ordem invertida na sigla. Inclui serviços de reparos em lataria (SM) e vidros/componentes (RR).",
            
            # Respostas sobre limites
            "limite_rrsm_porto": "O limite do RRSM (Reparo Rápido e Super Martelinho) da Porto Seguro varia conforme o plano contratado. Geralmente há um limite de eventos por vigência da apólice.",
            "limite_sm_porto": "O limite do SM (Super Martelinho) da Porto Seguro geralmente é de 2 utilizações por vigência da apólice, podendo variar conforme o plano contratado.",
            "limite_rr_porto": "O limite do RR (Reparo Rápido) da Porto Seguro varia conforme o plano. Geralmente são de 3 a 6 utilizações por vigência, dependendo do plano contratado.",
            "limite_rr_azul": "O limite do RR (Reparo Rápido) da Azul Seguros é geralmente de 3 utilizações por vigência da apólice, podendo variar conforme o plano contratado pelo cliente.",
            "limite_sm_azul": "O limite do SM (Super Martelinho) da Azul Seguros é geralmente de 2 utilizações por vigência da apólice, podendo variar conforme o plano contratado."
        }
    
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
            (r'(?:exclusões|exclusao|não cobre)(?:[^$]{10,500})', "exclusao"),
            # Limites
            (r'(?:limite|limites|máximo)(?:[^$]{10,500})', "limite"),
            # Prazos
            (r'(?:prazo|prazos|sla|liberação)(?:[^$]{10,500})', "prazo"),
            # Garantias
            (r'(?:garantia|garantias)(?:[^$]{10,500})', "garantia"),
            # Vidros
            (r'(?:vidro|vidros|para-brisa|parabrisa|vigia)(?:[^$]{10,500})', "vidros"),
            # Acessórios
            (r'(?:acessório|acessórios|farol|lanterna)(?:[^$]{10,500})', "acessorios"),
            # Reembolso
            (r'(?:reembolso|ordem de reembolso)(?:[^$]{10,500})', "reembolso"),
            # Procedimentos
            (r'(?:procedimento|como fazer|passo a passo)(?:[^$]{10,500})', "procedimento")
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
            "valor": ["valor", "coparticipação", "coparticipacao", "preço", "custo", "vmd"],
            "vidros": ["vidro", "vidros", "para-brisa", "parabrisa", "para brisa"],
            "rr": ["rr", "reparo rápido", "reparo rapido"],
            "sm": ["sm", "super martelinho", "supermartelinho"],
            "rrsm": ["rrsm", "smrr"],
            "limite": ["limite", "limites", "limitação"],
            "prazo": ["prazo", "prazos", "sla"],
            "garantia": ["garantia", "garantias"],
            "reembolso": ["reembolso", "ordem de reembolso"],
            "procedimento": ["procedimento", "como fazer", "passo a passo"]
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
    
    def direct_answer(self, query: str) -> Optional[str]:
        """
        Tenta responder diretamente a uma pergunta frequente sem usar o modelo.
        
        Args:
            query: A consulta do usuário
            
        Returns:
            Optional[str]: Resposta direta se disponível, None caso contrário
        """
        # Normalizar consulta
        query_norm = query.lower().strip().rstrip('?')
        
        # Verificar correspondência direta com padrões de FAQ
        for faq_id, patterns in self.faq_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_norm):
                    # Se houver uma resposta predefinida para esse padrão
                    if faq_id in self.faq_responses:
                        return self.faq_responses[faq_id]
        
        # Verificar correspondência exata ou aproximada com o dicionário de respostas
        for q, answer in self.faq_responses.items():
            if query_norm in q or q in query_norm:
                return answer
                
        return None  # Nenhuma resposta direta encontrada
    
    def process_faq_questions(self, questions_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Processa lista de perguntas frequentes para melhorar a recuperação.
        
        Args:
            questions_list: Lista de perguntas frequentes
            
        Returns:
            Dicionário com informações processadas das perguntas
        """
        processed_faqs = {}
        
        for i, question in enumerate(questions_list):
            question_text = question.strip()
            question_id = f"faq_{i+1}"
            
            # Normalizar e tokenizar
            normalized = self.normalize_text(question_text)
            tokens = self.tokenize_text(question_text)
            
            # Extrair entidades mencionadas
            entities = []
            for entity in self.known_entities:
                if entity.lower() in normalized:
                    entities.append(entity)
            
            # Identificar termos importantes
            important_terms = {}
            for category, terms in self.important_terms.items():
                matches = [term for term in terms if term in normalized]
                if matches:
                    important_terms[category] = matches
            
            # Identificar categorias da pergunta
            query_categories = []
            for category, keywords in self.query_types.items():
                if any(keyword in normalized for keyword in keywords):
                    query_categories.append(category)
            
            # Verificar correspondência com padrões de FAQ
            faq_matches = []
            for faq_id, patterns in self.faq_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, normalized):
                        faq_matches.append(faq_id)
                        break
            
            # Armazenar informações processadas
            processed_faqs[question_id] = {
                "original": question_text,
                "normalized": normalized,
                "tokens": tokens,
                "entities": entities,
                "important_terms": important_terms,
                "categories": query_categories,
                "faq_matches": faq_matches
            }
        
        return processed_faqs
    
    def create_faq_chunks(self, faqs_with_answers=None):
        """
        Cria chunks especializados para FAQs
        
        Args:
            faqs_with_answers: Dicionário opcional com respostas para FAQs
        """
        # Se não tiver respostas, usar o processamento para criar chunks de alta prioridade
        # baseados apenas nas perguntas, que serão usados para melhorar a busca
        for faq_id, faq_data in self.processed_faqs.items():
            # Conteúdo base será a pergunta
            content = f"PERGUNTA FREQUENTE: {faq_data['original']}\n\n"
            
            # Se temos resposta para esta pergunta, adicioná-la
            if faqs_with_answers and faq_id in faqs_with_answers:
                content += f"RESPOSTA: {faqs_with_answers[faq_id]}\n\n"
            else:
                # Procurar correspondências com respostas predefinidas
                for pattern_id in faq_data.get('faq_matches', []):
                    if pattern_id in self.faq_responses:
                        content += f"RESPOSTA: {self.faq_responses[pattern_id]}\n\n"
                        break
                
                # Caso não encontre resposta predefinida, adicionar metadados para ajudar na recuperação
                if "RESPOSTA:" not in content:
                    if faq_data['entities']:
                        content += f"ENTIDADES: {', '.join(faq_data['entities'])}\n"
                    
                    for category, terms in faq_data['important_terms'].items():
                        content += f"TERMOS ({category}): {', '.join(terms)}\n"
                    
                    if faq_data['categories']:
                        content += f"CATEGORIAS: {', '.join(faq_data['categories'])}\n"
            
            # Criar chunk com alta prioridade
            chunk = self.create_chunk_object(
                content=content,
                title=f"FAQ: {faq_data['original']}",
                category="faq",
                entities=faq_data['entities']
            )
            
            # Aumentar prioridade para FAQs
            chunk["priority"] = 100
            
            # Adicionar metadados específicos para busca
            chunk["faq_id"] = faq_id
            chunk["faq_categories"] = faq_data['categories']
            chunk["faq_terms"] = faq_data['important_terms']
            
            # Adicionar aos chunks
            self.chunks.append(chunk)
            chunk_index = len(self.chunks) - 1
            
            # Indexar por ID de FAQ
            self.faq_chunks[faq_id].append(chunk_index)
            
            # Indexar por entidades
            for entity in faq_data['entities']:
                self.entity_chunks[entity.lower()].append(chunk_index)
            
            # Indexar por categorias
            for category in faq_data['categories']:
                self.category_chunks[category].append(chunk_index)
            
            # Indexar por termos importantes
            for category, terms in faq_data['important_terms'].items():
                for term in terms:
                    self.term_chunks[term].append(chunk_index)
    
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
            
            # Indexar por termos importantes
            for term_category, count in chunk.get("important_matches", {}).items():
                for term in self.important_terms.get(term_category, []):
                    if term in chunk["text"].lower():
                        self.term_chunks[term].append(i)
        
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
        
        # Prioridades por tipo de categoria
        category_priorities = {
            "vidros": 8,
            "acessorios": 8,
            "rr": 7,
            "sm": 7,
            "rrsm": 7,
            "limite": 6,
            "prazo": 6,
            "garantia": 6,
            "franquia": 7,
            "reembolso": 8,
            "procedimento": 9,
            "faq": 100  # Máxima prioridade para FAQs
        }
        
        # Aplicar prioridade baseada na categoria
        if category in category_priorities:
            chunk["priority"] = max(chunk["priority"], category_priorities[category])
        
        # Ajustar prioridade com base em correspondências importantes
        if len(important_matches) > 2:
            chunk["priority"] += 2
        
        # Aumentar prioridade se tiver termos relacionados a perguntas frequentes
        faq_terms = ["como", "procedimento", "limite", "prazo", "valor", "cobre", "atende"]
        faq_term_count = sum(1 for term in faq_terms if term in content.lower())
        if faq_term_count >= 2:
            chunk["priority"] += faq_term_count
        
        return chunk
    
    def limit_context_size(self, context: str, max_tokens: int = 4000) -> str:
        """
        Limita o tamanho do contexto para evitar exceder limites de tokens
        
        Args:
            context: O contexto completo
            max_tokens: Número máximo estimado de tokens
            
        Returns:
            str: Contexto reduzido para caber no limite
        """
        # Estimativa aproximada: 1 token ~= 4 caracteres em média
        if len(context) > max_tokens * 4:
            # Dividir o contexto em partes
            parts = context.split("\n\n---\n\n") if "\n\n---\n\n" in context else context.split("\n\n")
            
            # Manter as partes mais relevantes primeiro (FAQs, entidades mencionadas)
            prioritized_parts = []
            rest_parts = []
            
            for part in parts:
                if "PERGUNTA FREQUENTE" in part or "ENTIDADES MENCIONADAS" in part:
                    prioritized_parts.append(part)
                else:
                    rest_parts.append(part)
            
            # Reconstruir o contexto limitado
            context_parts = prioritized_parts
            current_size = sum(len(part) for part in context_parts)
            
            # Adicionar outras partes até atingir o limite
            for part in rest_parts:
                if current_size + len(part) < max_tokens * 4:
                    context_parts.append(part)
                    current_size += len(part)
                else:
                    break
            
            # Juntar as partes selecionadas
            separator = "\n\n---\n\n" if "\n\n---\n\n" in context else "\n\n"
            return separator.join(context_parts)
        
        return context
    
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
                "category_chunks": dict(self.category_chunks),
                "term_chunks": dict(self.term_chunks),
                "faq_chunks": dict(self.faq_chunks)
            }, f)
        
        with open(self.entities_path, 'wb') as f:
            pickle.dump(self.entities, f)
        
        # 5. Salvar FAQs processadas, se houver
        if self.processed_faqs:
            with open(self.faq_path, 'wb') as f:
                pickle.dump({
                    "processed_faqs": self.processed_faqs,
                    "faq_vectors": self.faq_vectors
                }, f)
        
        print(f"Índices e chunks salvos com sucesso!")
    
    def create_index(self, text: str, faq_questions: List[str] = None) -> bool:
        """
        Cria índices a partir do texto do documento e perguntas frequentes.
        
        Args:
            text: Texto do documento
            faq_questions: Lista opcional de perguntas frequentes
            
        Returns:
            bool: True se índice criado com sucesso
        """
        # Processar perguntas frequentes, se fornecidas
        if faq_questions:
            print(f"Processando {len(faq_questions)} perguntas frequentes")
            self.processed_faqs = self.process_faq_questions(faq_questions)
        
        # Realizar pré-processamento do documento
        self.preprocess_text(text)
        
        # Criar chunks específicos para FAQs
        if self.processed_faqs:
            print("Criando chunks especializados para FAQs")
            self.create_faq_chunks()
            
            # Salvar índices atualizados com FAQs
            with open(self.chunks_path, 'wb') as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "entity_chunks": dict(self.entity_chunks),
                    "category_chunks": dict(self.category_chunks),
                    "term_chunks": dict(self.term_chunks),
                    "faq_chunks": dict(self.faq_chunks)
                }, f)
        
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
                    
                    # Carregar índices adicionais se disponíveis
                    if "term_chunks" in data:
                        self.term_chunks = defaultdict(list, data["term_chunks"])
                    if "faq_chunks" in data:
                        self.faq_chunks = defaultdict(list, data["faq_chunks"])
                
                # Carregar entidades
                with open(self.entities_path, 'rb') as f:
                    self.entities = pickle.load(f)
                
                # Carregar FAQs, se disponíveis
                if os.path.exists(self.faq_path):
                    with open(self.faq_path, 'rb') as f:
                        faq_data = pickle.load(f)
                        self.processed_faqs = faq_data.get("processed_faqs", {})
                        self.faq_vectors = faq_data.get("faq_vectors", {})
                
                print(f"Carregados {len(self.chunks)} chunks e {len(self.entities)} entidades")
                if self.processed_faqs:
                    print(f"Carregadas {len(self.processed_faqs)} perguntas frequentes")
                
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
        
        # 5. Verificar correspondência com padrões de FAQ
        faq_matches = []
        for faq_id, patterns in self.faq_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    faq_matches.append(faq_id)
                    break
        
        return {
            "entities": detected_entities,
            "type": query_type,
            "important_categories": important_categories,
            "has_undercar": has_undercar,
            "faq_matches": faq_matches
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
        
        # Adicionar variações específicas para termos técnicos
        term_variations = {
            "parabrisa": ["para-brisa", "para brisa"],
            "para-brisa": ["parabrisa", "para brisa"],
            "para brisa": ["parabrisa", "para-brisa"],
            "reparo rapido": ["reparo rápido", "rr"],
            "reparo rápido": ["reparo rapido", "rr"],
            "rr": ["reparo rápido", "reparo rapido"],
            "supermartelinho": ["super martelinho", "sm"],
            "super martelinho": ["supermartelinho", "sm"],
            "sm": ["super martelinho", "supermartelinho"],
            "vidros": ["parabrisa", "para-brisa", "vigia", "vidro"],
            "acessorios": ["acessórios", "farol", "lanterna"],
            "acessórios": ["acessorios", "farol", "lanterna"],
            "parachoque": ["para-choque", "para choque"],
            "para-choque": ["parachoque", "para choque"],
            "para choque": ["parachoque", "para-choque"]
        }
        
        # Adicionar variações específicas para os termos da consulta
        for term in list(expanded_terms):
            if term in term_variations:
                expanded_terms.update(term_variations[term])
        
        return expanded_terms
    
    def search(self, query: str, top_k: int = 7) -> List[Tuple[int, str, float]]:
        """
        Realiza busca semântica guiada por entidades e categorias,
        com suporte especial para perguntas frequentes.
        """
        if not self.chunks:
            raise ValueError("Os chunks não foram carregados")
        
        # 1. Classificar e expandir a consulta
        query_info = self.classify_query(query)
        expanded_terms = self.expand_query(query)
        
        # 2. Busca especial para correspondências diretas com FAQ
        if query_info["faq_matches"]:
            faq_candidates = set()
            for faq_id in query_info["faq_matches"]:
                # Adicionar chunks de FAQ correspondentes
                for chunk_id in self.faq_chunks.get(faq_id, []):
                    faq_candidates.add(chunk_id)
                
                # Verificar nos processed_faqs também
                for proc_faq_id, faq_data in self.processed_faqs.items():
                    if faq_id in faq_data.get("faq_matches", []):
                        # Adicionar chunks de FAQ correspondentes
                        for chunk_id in self.faq_chunks.get(proc_faq_id, []):
                            faq_candidates.add(chunk_id)
            
            # Se encontramos candidatos diretos, criar uma lista de resultados de alta prioridade
            if faq_candidates:
                faq_results = []
                for idx in faq_candidates:
                    chunk = self.chunks[idx]
                    score = 100  # Pontuação máxima para correspondências diretas de FAQ
                    faq_results.append((idx, chunk["text"], score))
                
                # Ordenar por relevância e retornar os top_k
                faq_results.sort(key=lambda x: x[2], reverse=True)
                return faq_results[:top_k]
        
        # 3. Preparar conjuntos de chunks a considerar (busca normal)
        candidate_chunks = set()
        
        # 3.1 Priorizar chunks relacionados às entidades detectadas
        for entity in query_info["entities"]:
            candidate_chunks.update(self.entity_chunks.get(entity.lower(), []))
        
        # 3.2 Considerar chunks de categorias importantes
        if query_info["important_categories"]:
            for category in query_info["important_categories"]:
                candidate_chunks.update(self.category_chunks.get(category, []))
        
        # 3.3 Busca especial para undercar
        if query_info["has_undercar"]:
            candidate_chunks.update(self.category_chunks.get("undercar", []))
        
        # 3.4 Adicionar chunks associados a termos específicos
        for term in expanded_terms:
            candidate_chunks.update(self.term_chunks.get(term, []))
        
        # 3.5 Se não houver candidatos específicos, considerar todos os chunks
        if not candidate_chunks:
            candidate_chunks = set(range(len(self.chunks)))
        
        # 4. Pontuar chunks candidatos
        chunk_scores = []
        
        for idx in candidate_chunks:
            chunk = self.chunks[idx]
            score = 0
            
            # 4.1 Pontuação base da prioridade do chunk
            score += chunk["priority"]
            
            # 4.2 Bônus para chunks de FAQ - máxima prioridade
            if chunk.get("category") == "faq":
                score += 50
            
            # 4.3 Pontuação por correspondência de termos
            for term in expanded_terms:
                if term in chunk["tokens"]:
                    # Peso maior para termos da consulta original
                    weight = 3 if term in query.lower() else 1
                    score += chunk["term_freq"].get(term, 0) * weight
            
            # 4.4 Pontuação adicional para correspondências importantes
            for category, count in chunk["important_matches"].items():
                if category in query_info["important_categories"]:
                    score += count * 5
            
            # 4.5 Bônus para chunks que mencionam entidades da consulta
            for entity in query_info["entities"]:
                if entity in [e.lower() for e in chunk["entities"]]:
                    score += 15
            
            # 4.6 Bônus especial para undercar se mencionado
            if query_info["has_undercar"] and "undercar" in chunk["text"].lower():
                score += 25
            
            # 4.7 Bônus para chunks que contêm palavras exatas da consulta
            query_tokens = self.tokenize_text(query)
            exact_matches = sum(1 for token in query_tokens if token in chunk["tokens"])
            score += exact_matches * 3
            
            # Adicionar à lista de resultados se tiver pontuação positiva
            if score > 0:
                chunk_scores.append((idx, chunk["text"], score))
        
        # 5. Se não encontrou nada relevante, usar busca de fallback
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
                if chunk["category"] in ["telefone", "responsavel", "undercar", "fluxo", "vidros", "limite", "prazo"]:
                    score += 2
                
                if score > 0:
                    chunk_scores.append((i, chunk["text"], score))
        
        # 6. Se ainda não encontrou nada, pegar chunks com prioridades maiores
        if not chunk_scores:
            high_priority_chunks = [(i, chunk["text"], chunk["priority"]) 
                                   forimport os
import pickle
import re
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional

class RAGEngine:
    def __init__(self):
        """
        Motor RAG otimizado para documentação técnica de seguros automotivos.
        Implementa reconhecimento de entidades, chunking semântico e busca contextual,
        com suporte especial para perguntas frequentes.
        """
        self.chunks = []
        self.entities = {}  # Dicionário de entidades (seguradoras, assistências)
        self.entity_chunks = defaultdict(list)  # Chunks por entidade
        self.category_chunks = defaultdict(list)  # Chunks por categoria
        self.faq_chunks = defaultdict(list)  # Chunks específicos para FAQs
        self.term_chunks = defaultdict(list)  # Chunks por termo específico
        
        self.index_path = "data/chunks_index.pkl"
        self.chunks_path = "data/document_chunks.pkl"
        self.entities_path = "data/entities.pkl"
        self.faq_path = "data/faq_index.pkl"
        
        # Lista de entidades (seguradoras/assistências) extraídas do documento
        self.known_entities = set([
            "ald comfort", "carbank", "assístia automob", "cdf", "carrefour", 
            "ezze seguros", "bradesco", "assistência", "automob", "porto", "azul",
            "porto seguro", "azul seguros", "bradesco seguros", "sura", "sura bmw",
            "ald", "audi", "bbf", "continental", "c6", "helps", "santander", 
            "hyundai", "plano auto prime", "positron", "psa", "chevrolet",
            "sem parar", "volkswagen"
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
            "responsável": ["responsável", "responsavel", "representante", "gerente", "encarregado", "supervisor", "fagner", "osório", "osorio", "gabriela", "tulio", "renata", "rampazzo", "matheus", "comercial"],
            "comercial": ["comercial", "comerciais", "vendas", "venda", "negócios", "atendimento", "cliente", "clientes"],
            "seguradora": ["seguradora", "seguro", "assistência", "assistencia", "automob", "comfort", "carbank", "bradesco", "ezze", "cdf", "carrefour", "porto", "azul", "sura"],
            "undercar": ["undercar", "pneus", "suspensão", "suspensao", "roda", "rodas", "pneu", "under", "car", "matheus"],
            "cobertura": ["cobertura", "coberturas", "contrato", "plano", "planos", "vigência", "vigencia", "assistência", "assistencia", "cobre"],
            "fluxo": ["fluxo", "atendimento", "procedimento", "script", "passo", "etapa", "processo"],
            "exclusao": ["exclusão", "exclusoes", "exclusao", "não", "exceto", "limitações", "restrições"],
            # Novos termos específicos para perguntas frequentes
            "vidros": ["vidro", "vidros", "parabrisa", "para-brisa", "para brisa", "vigia", "teto solar"],
            "acessorios": ["acessório", "acessórios", "farol", "faróis", "lanterna", "lanternas", "para-choque", "parachoque"],
            "rr": ["rr", "reparo rápido", "reparo rapido"],
            "sm": ["sm", "super martelinho", "supermartelinho"],
            "rrsm": ["rrsm", "smrr", "reparo rápido super martelinho", "super martelinho reparo rápido"],
            "limite": ["limite", "limites", "máximo", "maximo", "valor máximo"],
            "prazo": ["prazo", "prazos", "sla", "liberação", "liberar", "liberado"],
            "garantia": ["garantia", "garantias"],
            "franquia": ["franquia", "valor", "custo", "preço"],
            "vmd": ["vmd", "valor mínimo", "valor minimo"],
            "credenciamento": ["credenciamento", "credenciar", "credenciada", "loja credenciada"],
            "reembolso": ["reembolso", "reembolsar", "restituição", "ordem de reembolso"]
        }
        
        # Categorias de perguntas para classificação - expandidas
        self.query_types = {
            "info_pessoal": ["telefone", "responsável", "nome", "contato", "email", "fone", "quem"],
            "fluxo": ["como", "procedimento", "passo", "etapa", "fluxo", "fazer", "processo", "script"],
            "cobertura": ["cobre", "cobertura", "plano", "inclui", "incluído", "valor", "limite", "máximo", "vidros", "faróis", "para-brisa"],
            "excecao": ["não cobre", "exclusão", "excluído", "limitação", "restrição", "quando não", "exceção", "reembolso"],
            # Novas categorias específicas para as perguntas frequentes
            "atendimento": ["atendemos", "atende", "atender", "cobrimos"],
            "valor": ["valor", "preço", "custo", "franquia", "vmd", "reembolso"],
            "prazo": ["prazo", "tempo", "sla", "liberação", "liberar"],
            "garantia": ["garantia", "garantimos", "garante"],
            "inclusao": ["inclusão", "incluir", "adicionar"],
            "exclusao": ["exclusão", "excluir", "remover"],
            "selecao": ["selecionar", "como selecionar", "localizar", "como localizar"],
            "procedimento": ["procedimento", "como fazer", "passo a passo"]
        }
        
        # Padrões específicos para reconhecimento de perguntas frequentes
        self.faq_patterns = {
            # Atendimento para seguradoras
            "atendimento_porto_vidros": [r"atendemos vidros para a porto", r"porto.*vidros", r"vidros.*porto"],
            "atendimento_porto_acessorios": [r"atendemos acessórios para a porto", r"porto.*acessórios", r"acessórios.*porto"],
            "atendimento_azul_vidros": [r"atendemos vidros para a azul", r"azul.*vidros", r"vidros.*azul"],
            "atendimento_azul_acessorios": [r"atendemos acessórios para a azul", r"azul.*acessórios", r"acessórios.*azul"],
            
            # Franquia
            "franquia_sura": [r"sura tem valor de franquia", r"sura.*franquia", r"franquia.*sura"],
            "franquia_ald": [r"ald tem valor de franquia", r"ald.*franquia", r"franquia.*ald"],
            
            # Reembolso
            "ordem_reembolso": [r"o que é ordem de reembolso", r"ordem de reembolso", r"reembolso.*ordem"],
            
            # Reparos
            "reparo_parabrisa": [r"reparo de parabrisa", r"reparo de para-brisa", r"reparo de para brisa"],
            
            # Diferenças
            "diferenca_rrsm": [r"diferença rrsm", r"diferença.*rrsm"],
            "diferenca_smrr": [r"diferença smrr", r"diferença.*smrr"],
            
            # Limites
            "limite_rrsm_porto": [r"limite rrsm porto", r"porto.*limite.*rrsm", r"rrsm.*porto.*limite"],
            "limite_sm_porto": [r"limite sm porto", r"porto.*limite.*sm", r"sm.*porto.*limite"],
            "limite_rr_porto": [r"limite rr porto", r"porto.*limite.*rr", r"rr.*porto.*limite"],
            "limite_rr_azul": [r"limite rr azul", r"azul.*limite.*rr", r"rr.*azul.*limite"],
            "limite_sm_azul": [r"limite sm azul", r"azul.*limite.*sm", r"sm.*azul.*limite"],
            
            # Procedimentos genéricos
            "proc_ordem_reembolso": [r"procedimento ordem de reembolso", r"procedimento.*reembolso"],
            "proc_comanda_manual": [r"como fazer comanda manual", r"comanda manual", r"script comanda manual"],
            
            # Seleção de peças específicas
            "selecao_parabrisa_volvo": [r"como selecionar.*parabrisa.*volvo", r"selecionar.*para.?brisa.*volvo"],
            "selecao_parabrisa_mercedes": [r"como selecionar.*parabrisa.*mercedes", r"selecionar.*para.?brisa.*mercedes"],
            
            # Coberturas específicas
            "cobertura_lanterna": [r"lanterna.*possui cobertura", r"cobertura.*lanterna", r"cobre.*lanterna"],
            "cobertura_parabrisa": [r"para.?brisa.*possui cobertura", r"cobertura.*para.?brisa", r"cobre.*para.?brisa"],
            
            # SLA e prazos
            "sla_porto": [r"sla porto", r"prazo.*porto", r"porto.*prazo"],
            "sla_azul": [r"sla azul", r"prazo.*azul", r"azul.*prazo"],
            "sla_bradesco": [r"sla bradesco", r"prazo.*bradesco", r"bradesco.*prazo"]
        }
        
        # Criar pasta data se não existir
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # Inicializar os vetores FAQ
        self.faq_vectors = {}
        self.processed_faqs = {}
        
        # Dicionário de respostas para perguntas frequentes
        self.faq_responses = {
            # Respostas para perguntas sobre atendimento
            "atendimento_porto_vidros": "Sim, atendemos vidros para a Porto Seguro. Os serviços incluem troca e reparo de para-brisas, vidros laterais e traseiros.",
            "atendimento_porto_acessorios": "Sim, atendemos acessórios para a Porto Seguro. Os serviços incluem faróis, lanternas, retrovisores e para-choques.",
            "atendimento_azul_vidros": "Sim, atendemos vidros para a Azul Seguros. Os serviços incluem troca e reparo de para-brisas, vidros laterais e traseiros.",
            "atendimento_azul_acessorios": "Sim, atendemos acessórios para a Azul Seguros. Os serviços incluem faróis, lanternas, retrovisores e para-choques.",
            
            # Respostas sobre franquia
            "franquia_sura": "Sim, a Sura possui valor de franquia para os serviços de vidros e acessórios. Os valores específicos dependem do plano contratado pelo cliente.",
            "franquia_ald": "Sim, a ALD possui valor de franquia para os serviços. Os valores variam conforme o tipo de veículo e o plano contratado.",
            
            # Respostas sobre reembolso
            "ordem_reembolso": "Ordem de reembolso é um procedimento utilizado quando o cliente realiza o serviço por conta própria e solicita o reembolso do valor à seguradora, dentro dos limites da apólice.",
            
            # Respostas sobre diferenças
            "diferenca_rrsm": "RRSM (Reparo Rápido e Super Martelinho) é uma combinação de serviços que inclui pequenos reparos de lataria e pintura (Super Martelinho) junto com serviços de reparo rápido para danos em vidros e outros componentes.",
            "diferenca_smrr": "SMRR (Super Martelinho e Reparo Rápido) é o mesmo que RRSM, apenas com a ordem invertida na sigla. Inclui serviços de reparos em lataria (SM) e vidros/componentes (RR).",
            
            # Respostas sobre limites
            "limite_rrsm_porto": "O limite do RRSM (Reparo Rápido e Super Martelinho) da Porto Seguro varia conforme o plano contratado. Geralmente há um limite de eventos por vigência da apólice.",
            "limite_sm_porto": "O limite do SM (Super Martelinho) da Porto Seguro geralmente é de 2 utilizações por vigência da apólice, podendo variar conforme o plano contratado.",
            "limite_rr_porto": "O limite do RR (Reparo Rápido) da Porto Seguro varia conforme o plano. Geralmente são de 3 a 6 utilizações por vigência, dependendo do plano contratado.",
            "limite_rr_azul": "O limite do RR (Reparo Rápido) da Azul Seguros é geralmente de 3 utilizações por vigência da apólice, podendo variar conforme o plano contratado pelo cliente.",
            "limite_sm_azul": "O limite do SM (Super Martelinho) da Azul Seguros é geralmente de 2 utilizações por vigência da apólice, podendo variar conforme o plano contratado."
        }
    
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
            (r'(?:exclusões|exclusao|não cobre)(?:[^$]{10,500})', "exclusao"),
            # Limites
            (r'(?:limite|limites|máximo)(?:[^$]{10,500})', "limite"),
            # Prazos
            (r'(?:prazo|prazos|sla|liberação)(?:[^$]{10,500})', "prazo"),
            # Garantias
            (r'(?:garantia|garantias)(?:[^$]{10,500})', "garantia"),
            # Vidros
            (r'(?:vidro|vidros|para-brisa|parabrisa|vigia)(?:[^$]{10,500})', "vidros"),
            # Acessórios
            (r'(?:acessório|acessórios|farol|lanterna)(?:[^$]{10,500})', "acessorios"),
            # Reembolso
            (r'(?:reembolso|ordem de reembolso)(?:[^$]{10,500})', "reembolso"),
            # Procedimentos
            (r'(?:procedimento|como fazer|passo a passo)(?:[^$]{10,500})', "procedimento")
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
            "valor": ["valor", "coparticipação", "coparticipacao", "preço", "custo", "vmd"],
            "vidros": ["vidro", "vidros", "para-brisa", "parabrisa", "para brisa"],
            "rr": ["rr", "reparo rápido", "reparo rapido"],
            "sm": ["sm", "super martelinho", "supermartelinho"],
            "rrsm": ["rrsm", "smrr"],
            "limite": ["limite", "limites", "limitação"],
            "prazo": ["prazo", "prazos", "sla"],
            "garantia": ["garantia", "garantias"],
            "reembolso": ["reembolso", "ordem de reembolso"],
            "procedimento": ["procedimento", "como fazer", "passo a passo"]
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
    
    def direct_answer(self, query: str) -> Optional[str]:
        """
        Tenta responder diretamente a uma pergunta frequente sem usar o modelo.
        
        Args:
            query: A consulta do usuário
            
        Returns:
            Optional[str]: Resposta direta se disponível, None caso contrário
        """
        # Normalizar consulta
        query_norm = query.
