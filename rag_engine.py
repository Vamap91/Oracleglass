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
        self.pdf_path = pdf_path
        self.document_text = ""
        self.chunks = []
        self.vector_store = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.base_processed_filename = ""
        self.chunks_path = ""
        self.vector_store_path = ""
        self.entities_path = ""
        self.faq_index_path = "" # Para o antigo faq_index.pkl, pode ser removido ou adaptado
        self.faq_pairs_path = "" # Novo caminho para o faq_pairs.json

        os.makedirs(CONFIG_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        self.stopwords = self._load_json_config(os.path.join(CONFIG_DIR, "stopwords_pt.json"), default_value={"stopwords": []})["stopwords"]
        self.known_entities = self._load_json_config(os.path.join(CONFIG_DIR, "known_entities.json"), default_value={"entities": []})["entities"]
        self.important_terms = self._load_json_config(os.path.join(CONFIG_DIR, "important_terms.json"), default_value={})
        self.query_types = self._load_json_config(os.path.join(CONFIG_DIR, "query_types.json"), default_value={})
        # Os faq_patterns e faq_responses originais podem ser mantidos para uma lógica de FAQ mais complexa no futuro
        self.faq_patterns = self._load_json_config(os.path.join(CONFIG_DIR, "faq_patterns.json"), default_value={})
        self.faq_responses_config = self._load_json_config(os.path.join(CONFIG_DIR, "faq_responses.json"), default_value={})
        
        self.faq_data = {} # Dicionário para armazenar os pares de FAQ carregados do faq_pairs.json

        self.entities_extracted = {}
        self.entity_chunks = defaultdict(list)
        self.category_chunks = defaultdict(list)
        self.faq_chunks = defaultdict(list)
        self.term_chunks = defaultdict(list)

        if self.pdf_path:
            self.load_and_process_pdf(self.pdf_path)

    def _load_json_config(self, file_path: str, default_value: Any = None) -> Any:
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
        print(f"Arquivo de configuração não encontrado: {file_path}. Usando valor padrão.")
        return default_value

    def _save_pickle(self, data: Any, file_path: str):
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Dados salvos em {file_path}")
        except Exception as e:
            print(f"Erro ao salvar pickle em {file_path}: {e}")

    def _load_pickle(self, file_path: str) -> Any:
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Erro ao carregar pickle de {file_path}: {e}")
        return None

    def _set_processed_paths(self, pdf_filename: str):
        self.base_processed_filename = os.path.splitext(pdf_filename)[0]
        self.chunks_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_chunks.pkl")
        self.vector_store_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_faiss.index")
        self.entities_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_entities.pkl")
        self.faq_index_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_faq_index.pkl")
        self.faq_pairs_path = os.path.join(PROCESSED_DIR, f"{self.base_processed_filename}_faq_pairs.json") # Caminho para o novo arquivo de FAQ

    def load_pdf_text(self, pdf_path: str) -> str:
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            print(f"Texto extraído de {pdf_path} com sucesso.")
            return text
        except FileNotFoundError:
            print(f"Erro: Arquivo PDF não encontrado em {pdf_path}")
            return ""
        except Exception as e:
            print(f"Erro ao ler o arquivo PDF {pdf_path}: {e}")
            return ""

    def normalize_text(self, text: str) -> str:
        if not text: return ""
        try:
            nfkd_form = unicodedata.normalize('NFKD', text)
            only_ascii = nfkd_form.encode('ASCII', 'ignore')
            text = only_ascii.decode('ASCII')
        except Exception as e:
            print(f"Erro ao normalizar texto (remoção de acentos): {e}")
        return text.lower().strip()

    def tokenize_text(self, text: str) -> List[str]:
        if not text: return []
        normalized_text = self.normalize_text(text)
        processed_text = re.sub(r"[\"#$%&\\'()*+,./:;<=>?@[\\]^_`{|}~]", " ", normalized_text)
        processed_text = re.sub(r"\s+", " ", processed_text).strip()
        tokens = processed_text.split()
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        else:
            tokens = [token for token in tokens if len(token) > 1]
        return tokens

    def semantic_chunking(self, text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
        if not text: return []
        tokens = self.tokenize_text(text)
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
                 if end_pos < len(tokens):
                     last_chunk_tokens = tokens[end_pos:]
                     if last_chunk_tokens:
                         chunks.append(" ".join(last_chunk_tokens))
                 break
        return chunks

    def build_vector_store(self, chunks: List[str]):
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
            if self.vector_store_path and self.chunks_path:
                faiss.write_index(self.vector_store, self.vector_store_path)
                self._save_pickle(chunks, self.chunks_path)
                print(f"Índice FAISS salvo em: {self.vector_store_path}")
                print(f"Chunks salvos em: {self.chunks_path}")
        except Exception as e:
            print(f"Erro ao construir o vector store: {e}")
            self.vector_store = None

    def load_processed_data(self) -> bool:
        if not self.base_processed_filename:
            print("Nome base do arquivo processado não definido. Não é possível carregar dados.")
            return False
        
        loaded_successfully = False
        if os.path.exists(self.vector_store_path) and os.path.exists(self.chunks_path):
            try:
                print(f"Carregando vector store de {self.vector_store_path}...")
                self.vector_store = faiss.read_index(self.vector_store_path)
                print(f"Carregando chunks de {self.chunks_path}...")
                self.chunks = self._load_pickle(self.chunks_path)
                if self.vector_store and self.chunks and self.vector_store.ntotal == len(self.chunks):
                    print("Dados de chunks e vector store carregados com sucesso.")
                    loaded_successfully = True
                else:
                    print("Falha ao carregar dados de chunks/vector store ou dados inconsistentes.")
                    self.vector_store = None
                    self.chunks = []
            except Exception as e:
                print(f"Erro ao carregar dados de chunks/vector store: {e}")
                self.vector_store = None
                self.chunks = []
        
        # Carregar dados de FAQ
        if os.path.exists(self.faq_pairs_path):
            self.faq_data = self._load_json_config(self.faq_pairs_path, default_value={})
            if self.faq_data:
                print(f"Dados de FAQ carregados de {self.faq_pairs_path}.")
            else:
                print(f"Arquivo de FAQ encontrado ({self.faq_pairs_path}), mas vazio ou com erro de leitura.")
        else:
            print(f"Arquivo de FAQ não encontrado em {self.faq_pairs_path}. A funcionalidade de FAQ direta não estará disponível.")
            self.faq_data = {}
            
        return loaded_successfully

    def load_and_process_pdf(self, pdf_path: str, force_reprocess: bool = False):
        self.pdf_path = pdf_path
        pdf_filename = os.path.basename(pdf_path)
        self._set_processed_paths(pdf_filename)

        if not force_reprocess and self.load_processed_data():
            print(f"Usando dados processados existentes para {pdf_filename}.")
            return

        print(f"Processando PDF: {pdf_path}...")
        self.document_text = self.load_pdf_text(pdf_path)
        if not self.document_text:
            print("Falha ao carregar texto do PDF. Abortando processamento.")
            return

        self.chunks = self.semantic_chunking(self.document_text)
        if not self.chunks:
            print("Nenhum chunk gerado a partir do PDF. Abortando.")
            return
        
        self.build_vector_store(self.chunks)
        # Após processar o PDF e construir o vector store, também carregamos/recarregamos os FAQs
        # pois o faq_pairs.json pode ter sido gerado/atualizado pelo script process_faq_script.py
        if os.path.exists(self.faq_pairs_path):
            self.faq_data = self._load_json_config(self.faq_pairs_path, default_value={})
            if self.faq_data:
                print(f"Dados de FAQ (re)carregados de {self.faq_pairs_path} após processamento do PDF.")
        else:
             print(f"Arquivo de FAQ não encontrado em {self.faq_pairs_path} após processamento do PDF.")
             self.faq_data = {}

        print("Processamento do PDF concluído.")

    def get_answer(self, query: str, k: int = 3) -> Tuple[str, str]:
        """Obtém uma resposta para a consulta, priorizando FAQs e depois busca semântica."""
        normalized_query = self.normalize_text(query)

        # 1. Verificar FAQs (correspondência exata normalizada)
        if self.faq_data:
            for faq_q, faq_a in self.faq_data.items():
                if self.normalize_text(faq_q) == normalized_query:
                    print(f"Resposta encontrada no FAQ para: '{query}'")
                    return faq_a, "faq"
        
        # 2. Se não encontrou no FAQ, realizar busca semântica
        if not self.vector_store or not self.chunks:
            # Tenta carregar dados processados se ainda não foram carregados e o pdf_path está definido
            if self.pdf_path and not (self.vector_store and self.chunks):
                print("Vector store ou chunks não prontos. Tentando carregar dados processados...")
                pdf_filename = os.path.basename(self.pdf_path)
                self._set_processed_paths(pdf_filename)
                self.load_processed_data()
            
            if not self.vector_store or not self.chunks: # Verifica novamente após tentativa de carregamento
                message = "Vector store não está pronto ou não há chunks carregados. Por favor, processe um PDF primeiro."
                print(message)
                return message, "error"
                
        if not query:
            return "Consulta vazia.", "error"
        
        try:
            print(f"Buscando semanticamente por: '{query}'")
            query_embedding = self.model.encode([normalized_query]) # Usa a query normalizada
            distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), k)
            
            results = []
            if len(indices[0]) > 0:
                for i in range(len(indices[0])):
                    chunk_index = indices[0][i]
                    if 0 <= chunk_index < len(self.chunks):
                        results.append(self.chunks[chunk_index]) # Apenas o chunk, sem o score por enquanto
                
                # Concatenar os chunks relevantes para formar a resposta contextual
                contextual_answer = "\n\n---\n\n".join(results)
                return contextual_answer, "semantic_search"
            else:
                return "Nenhuma informação relevante encontrada no documento para esta consulta.", "semantic_search_empty"
        except Exception as e:
            print(f"Erro durante a busca semântica: {e}")
            return f"Erro durante a busca: {e}", "error"

    # O método search original pode ser mantido para uso interno ou removido se get_answer o substitui completamente.
    # Por ora, vou comentá-lo para evitar confusão, já que get_answer é mais completo.
    # def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
    #     ...

# Exemplo de uso (para teste)
if __name__ == "__main__":
    if not os.path.exists(os.path.join(CONFIG_DIR, "known_entities.json")):
        with open(os.path.join(CONFIG_DIR, "known_entities.json"), "w") as f:
            json.dump({"entities": ["porto seguro", "azul seguros", "bradesco seguros"]}, f)
    
    if not os.path.exists(os.path.join(CONFIG_DIR, "stopwords_pt.json")):
         with open(os.path.join(CONFIG_DIR, "stopwords_pt.json"), "w") as f:
            json.dump({"stopwords": ["de", "a", "o", "que", "e", "do", "da", "em", "um"]}, f)

    TEST_PDF_DIR = "data"
    TEST_PDF_NAME = "Guia Rápido.pdf"
    TEST_PDF_PATH = os.path.join(TEST_PDF_DIR, TEST_PDF_NAME)

    os.makedirs(TEST_PDF_DIR, exist_ok=True)
    if not os.path.exists(TEST_PDF_PATH):
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(TEST_PDF_PATH)
            c.drawString(100, 750, "Bem-vindo ao Guia Rápido da Porto Seguro.")
            c.drawString(100, 730, "Este documento contém informações sobre seguros e SLA.")
            c.drawString(100, 710, "O SLA para vidros é de 2 dias. O SLA para acessórios é de 3 dias.")
            c.drawString(100, 690, "Para mais detalhes, contate a Azul Seguros.")
            c.save()
            print(f"PDF de teste criado em {TEST_PDF_PATH}")
        except ImportError:
            print("Por favor, instale reportlab para criar o PDF de teste: pip install reportlab")
        except Exception as e:
            print(f"Não foi possível criar o PDF de teste: {e}")

    # Criar um arquivo faq_pairs.json de exemplo para teste
    example_faq_pairs_path = os.path.join(PROCESSED_DIR, f"{os.path.splitext(TEST_PDF_NAME)[0]}_faq_pairs.json")
    if not os.path.exists(example_faq_pairs_path):
        example_faqs = {
            "Qual o SLA para vidros?": "O SLA para vidros é de 2 dias, conforme especificado no documento.",
            "Atendemos vidros para a porto?": "Sim, a Porto Seguro é uma das seguradoras atendidas para serviços de vidros."
        }
        try:
            with open(example_faq_pairs_path, "w", encoding="utf-8") as f:
                json.dump(example_faqs, f, ensure_ascii=False, indent=4)
            print(f"Arquivo FAQ de exemplo criado em: {example_faq_pairs_path}")
        except Exception as e:
            print(f"Erro ao criar arquivo FAQ de exemplo: {e}")

    if os.path.exists(TEST_PDF_PATH):
        print(f"\n--- Testando RAGEngine com {TEST_PDF_PATH} ---")
        engine = RAGEngine()
        # É importante que load_and_process_pdf seja chamado para que os caminhos (incluindo faq_pairs_path) sejam definidos
        # e para que o faq_data seja carregado.
        engine.load_and_process_pdf(TEST_PDF_PATH, force_reprocess=False) 

        print("\n--- Testando busca com FAQ ---")
        # Teste 1: Pergunta que deve ser respondida pelo FAQ
        query_faq = "Qual o SLA para vidros?"
        answer_faq, source_faq = engine.get_answer(query_faq)
        print(f"Pergunta: {query_faq}\nFonte: {source_faq}\nResposta: {answer_faq}\n")

        # Teste 2: Pergunta que NÃO está no FAQ, deve usar busca semântica
        query_semantic = "informações sobre seguros"
        answer_semantic, source_semantic = engine.get_answer(query_semantic)
        print(f"Pergunta: {query_semantic}\nFonte: {source_semantic}\nResposta: {answer_semantic}\n")
        
        # Teste 3: Outra pergunta do FAQ
        query_faq2 = "Atendemos vidros para a porto?"
        answer_faq2, source_faq2 = engine.get_answer(query_faq2)
        print(f"Pergunta: {query_faq2}\nFonte: {source_faq2}\nResposta: {answer_faq2}\n")

    else:
        print(f"Arquivo PDF de teste {TEST_PDF_PATH} não encontrado. Crie-o ou ajuste o caminho.")

