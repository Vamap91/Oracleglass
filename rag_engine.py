import os
import json
import pickle
import re
import string
import unicodedata
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

CONFIG_DIR = "data/config"
PROCESSED_DIR = "data/processed"

class RAGEngine:
    def __init__(self, pdf_path: Optional[str] = None):
        """
        Motor RAG otimizado para documentação técnica.
        Implementa reconhecimento de entidades, chunking semântico e busca contextual,
        com suporte especial para perguntas frequentes.
        """
        self.pdf_path = pdf_path
        self.document_text = ""
        self.chunks = []
        self.vector_store = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # Modelo padrão, pode ser configurável

        # Caminhos para arquivos processados (relativos ao PROCESSED_DIR)
        # O nome do arquivo processado pode depender do PDF original para evitar colisões
        self.base_processed_filename = "" # Será definido após o carregamento do PDF
        self.chunks_path = ""
        self.vector_store_path = ""
        self.entities_path = ""
        self.faq_path = ""

        # Cria diretórios de configuração e processados se não existirem
        os.makedirs(CONFIG_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        # Carregar configurações externas
        self.stopwords = self._load_json_config(os.path.join(CONFIG_DIR, "stopwords_pt.json"), default_value={"stopwords": []})["stopwords"]
        self.known_entities = self._load_json_config(os.path.join(CONFIG_DIR, "known_entities.json"), default_value={"entities": []})["entities"]
        self.important_terms = self._load_json_config(os.path.join(CONFIG_DIR, "important_terms.json"), default_value={})
        self.query_types = self._load_json_config(os.path.join(CONFIG_DIR, "query_types.json"), default_value={})
        self.faq_patterns = self._load_json_config(os.path.join(CONFIG_DIR, "faq_patterns.json"), default_value={})
        self.faq_responses = self._load_json_config(os.path.join(CONFIG_DIR, "faq_responses.json"), default_value={})

        self.entities_extracted = {}  # Dicionário de entidades (seguradoras, assistências) extraídas do doc atual
        self.entity_chunks = defaultdict(list)  # Chunks por entidade
        self.category_chunks = defaultdict(list)  # Chunks por categoria
        self.faq_chunks = defaultdict(list)  # Chunks específicos para FAQs
        self.term_chunks = defaultdict(list)  # Chunks por termo específico

        if self.pdf_path:
            self.load_and_process_pdf(self.pdf_path)

    def _load_json_config(self, file_path: str, default_value: Any = None) -> Any:
        """Carrega uma configuração JSON de um arquivo."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON de {file_path}: {e}")
                return default_value
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")
                return default_value
        return default_value

    def _save_pickle(self, data: Any, file_path: str):
        """Salva dados em um arquivo pickle."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Dados salvos em {file_path}")
        except Exception as e:
            print(f"Erro ao salvar pickle em {file_path}: {e}")

    def _load_pickle(self, file_path: str) -> Any:
        """Carrega dados de um arquivo pickle."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Erro ao carregar pickle de {file_path}: {e}")
        return None

    def _set_processed_paths(self, pdf_filename: str):
        """Define os caminhos para os arquivos processados com base no nome do PDF."""
        self.base_processed_filename = os.path.splitext(pdf_filename)[0]
        self.chunks_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_chunks.pkl")
        self.vector_store_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_faiss.index")
        self.entities_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_entities.pkl")
        # FAQ path pode ser genérico ou específico, dependendo da estratégia
        self.faq_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_faq_index.pkl") 

    def load_pdf_text(self, pdf_path: str) -> str:
        """Carrega o texto de um arquivo PDF."""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            print(f"Texto extraído de {pdf_path} com sucesso.")
            return text
        except FileNotFoundError:
            print(f"Erro: Arquivo PDF não encontrado em {pdf_path}")
            return ""
        except Exception as e:
            print(f"Erro ao ler o arquivo PDF {pdf_path}: {e}")
            return ""

    def normalize_text(self, text: str) -> str:
        """Normaliza texto removendo acentos e convertendo para minúsculas."""
        if not text: return ""
        try:
            # Remover acentos
            nfkd_form = unicodedata.normalize('NFKD', text)
            only_ascii = nfkd_form.encode('ASCII', 'ignore')
            text = only_ascii.decode('ASCII')
        except Exception as e:
            print(f"Erro ao normalizar texto (remoção de acentos): {e}")
        return text.lower()

    def tokenize_text(self, text: str) -> List[str]:
        """Tokeniza e normaliza o texto removendo pontuação e stopwords."""
        if not text: return []
        normalized_text = self.normalize_text(text)
        # Remover pontuação - mantendo hífens em palavras compostas e números
        # Esta regex é um pouco mais permissiva com pontuação interna que pode ser relevante.
        processed_text = re.sub(r"[\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~]", " ", normalized_text) # Não remove hífens ou exclamações/interrogações
        processed_text = re.sub(r"\s+", " ", processed_text).strip() # Normaliza espaços múltiplos
        
        tokens = processed_text.split()
        
        # Remover stopwords
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        else:
            tokens = [token for token in tokens if len(token) > 1]
            
        return tokens

    # ... (restante das funções de RAGEngine: semantic_chunking, build_vector_store, search, etc. a serem adicionadas/refatoradas)
    # ... (funções de extração de entidades, processamento de FAQ, etc. a serem adaptadas)

    def semantic_chunking(self, text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
        """Divide o texto em chunks semânticos com sobreposição."""
        if not text: return []
        # Uma abordagem simples de chunking por tokens, pode ser melhorada com sentence-splitting mais inteligente
        tokens = self.tokenize_text(text) # Usar a tokenização da classe
        if not tokens: return []

        chunks = []
        current_pos = 0
        while current_pos < len(tokens):
            end_pos = min(current_pos + chunk_size, len(tokens))
            chunk = " ".join(tokens[current_pos:end_pos])
            chunks.append(chunk)
            if end_pos == len(tokens):
                break
            current_pos += (chunk_size - overlap)
            if current_pos >= len(tokens):
                 # Evitar loop infinito se overlap for muito grande ou chunk_size pequeno
                 # Adiciona o restante se houver e não foi coberto
                 if end_pos < len(tokens):
                     last_chunk_tokens = tokens[end_pos:]
                     if last_chunk_tokens:
                         chunks.append(" ".join(last_chunk_tokens))
                 break
        return chunks

    def build_vector_store(self, chunks: List[str]):
        """Constrói o vector store (FAISS index) a partir dos chunks."""
        if not chunks:
            print("Nenhum chunk fornecido para construir o vector store.")
            self.vector_store = None
            return
        try:
            print(f"Gerando embeddings para {len(chunks)} chunks...")
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(np.array(embeddings, dtype=np.float32))
            print(f"Vector store construído com {self.vector_store.ntotal} vetores.")
            # Salvar o índice e os chunks
            if self.vector_store_path and self.chunks_path:
                faiss.write_index(self.vector_store, self.vector_store_path)
                self._save_pickle(chunks, self.chunks_path)
                print(f"Índice FAISS salvo em: {self.vector_store_path}")
                print(f"Chunks salvos em: {self.chunks_path}")
        except Exception as e:
            print(f"Erro ao construir o vector store: {e}")
            self.vector_store = None

    def load_processed_data(self) -> bool:
        """Carrega chunks e vector store processados, se existirem."""
        if not self.base_processed_filename: # Garante que os caminhos foram definidos
            print("Nome base do arquivo processado não definido. Não é possível carregar dados.")
            return False
            
        if os.path.exists(self.vector_store_path) and os.path.exists(self.chunks_path):
            try:
                print(f"Carregando vector store de {self.vector_store_path}...")
                self.vector_store = faiss.read_index(self.vector_store_path)
                print(f"Carregando chunks de {self.chunks_path}...")
                self.chunks = self._load_pickle(self.chunks_path)
                if self.vector_store and self.chunks and self.vector_store.ntotal == len(self.chunks):
                    print("Dados processados carregados com sucesso.")
                    return True
                else:
                    print("Falha ao carregar dados processados ou dados inconsistentes.")
                    self.vector_store = None
                    self.chunks = []
                    return False
            except Exception as e:
                print(f"Erro ao carregar dados processados: {e}")
                self.vector_store = None
                self.chunks = []
                return False
        return False

    def load_and_process_pdf(self, pdf_path: str, force_reprocess: bool = False):
        """Carrega, processa o PDF e constrói o vector store."""
        self.pdf_path = pdf_path
        pdf_filename = os.path.basename(pdf_path)
        self._set_processed_paths(pdf_filename) # Define os caminhos baseados no nome do PDF

        if not force_reprocess and self.load_processed_data():
            print(f"Usando dados processados existentes para {pdf_filename}.")
            # Carregar outras informações processadas como entidades, FAQs, se necessário
            # self.entities_extracted = self._load_pickle(self.entities_path) or {}
            # self.faq_data_processed = self._load_pickle(self.faq_path) or {}
            return

        print(f"Processando PDF: {pdf_path}...")
        self.document_text = self.load_pdf_text(pdf_path)
        if not self.document_text:
            print("Falha ao carregar texto do PDF. Abortando processamento.")
            return

        # Aqui podem entrar as lógicas de extração de entidades, FAQs, etc., antes do chunking geral
        # self.entities_extracted = self.extract_entities(self.document_text) # Exemplo
        # self._save_pickle(self.entities_extracted, self.entities_path)

        self.chunks = self.semantic_chunking(self.document_text)
        if not self.chunks:
            print("Nenhum chunk gerado a partir do PDF. Abortando.")
            return
        
        self.build_vector_store(self.chunks)
        print("Processamento do PDF concluído.")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Realiza uma busca semântica no vector store."""
        if not self.vector_store or not self.chunks:
            print("Vector store não está pronto ou não há chunks carregados.")
            return []
        if not query:
            return []
        try:
            print(f"Buscando por: '{query}'")
            query_embedding = self.model.encode([self.normalize_text(query)])
            distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), k)
            
            results = []
            for i in range(len(indices[0])):
                chunk_index = indices[0][i]
                if 0 <= chunk_index < len(self.chunks):
                    results.append((self.chunks[chunk_index], float(distances[0][i])))
            return results
        except Exception as e:
            print(f"Erro durante a busca: {e}")
            return []

    # TODO: Implementar/Adaptar as funções:
    # - extract_entities (para usar self.known_entities e salvar em self.entities_path)
    # - process_faqs (para usar self.faq_patterns, self.faq_responses e salvar em self.faq_path)
    # - classify_query (para usar self.query_types)
    # - get_answer (lógica principal que combina busca semântica, FAQs, etc.)

# Exemplo de uso (para teste)
if __name__ == "__main__":
    # Criar arquivos de configuração de exemplo se não existirem
    if not os.path.exists(os.path.join(CONFIG_DIR, "known_entities.json")):
        with open(os.path.join(CONFIG_DIR, "known_entities.json"), "w") as f:
            json.dump({"entities": ["porto seguro", "azul seguros", "bradesco seguros"]}, f)
    
    if not os.path.exists(os.path.join(CONFIG_DIR, "stopwords_pt.json")):
         with open(os.path.join(CONFIG_DIR, "stopwords_pt.json"), "w") as f:
            json.dump({"stopwords": ["de", "a", "o", "que", "e", "do", "da", "em", "um"]}, f) # Exemplo mínimo

    # Supondo que você tenha um PDF de teste em data/Guia Rapido.pdf
    TEST_PDF_DIR = "data"
    TEST_PDF_NAME = "Guia Rapido.pdf" # Coloque seu PDF de teste aqui
    TEST_PDF_PATH = os.path.join(TEST_PDF_DIR, TEST_PDF_NAME)

    os.makedirs(TEST_PDF_DIR, exist_ok=True)
    # Crie um PDF de teste simples se não existir
    if not os.path.exists(TEST_PDF_PATH):
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(TEST_PDF_PATH)
            c.drawString(100, 750, "Bem-vindo ao Guia Rápido da Porto Seguro.")
            c.drawString(100, 730, "Este documento contém informações sobre seguros.")
            c.drawString(100, 710, "Para mais detalhes, contate a Azul Seguros.")
            c.save()
            print(f"PDF de teste criado em {TEST_PDF_PATH}")
        except ImportError:
            print("Por favor, instale reportlab para criar o PDF de teste: pip install reportlab")
        except Exception as e:
            print(f"Não foi possível criar o PDF de teste: {e}")

    if os.path.exists(TEST_PDF_PATH):
        print(f"--- Testando RAGEngine com {TEST_PDF_PATH} ---")
        engine = RAGEngine()
        engine.load_and_process_pdf(TEST_PDF_PATH, force_reprocess=False) # Mude para True para forçar o reprocessamento

        if engine.vector_store:
            print("--- Testando busca ---")
            search_results = engine.search("informações porto seguro")
            if search_results:
                for chunk, score in search_results:
                    print(f"Score: {score:.4f} - Chunk: {chunk[:100]}...")
            else:
                print("Nenhum resultado encontrado.")
        else:
            print("Motor não inicializado corretamente, busca não pode ser realizada.")
    else:
        print(f"Arquivo PDF de teste {TEST_PDF_PATH} não encontrado. Crie-o ou ajuste o caminho.")


