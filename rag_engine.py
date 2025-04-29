# rag_engine.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import openai
from sentence_transformers import SentenceTransformer
import faiss

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Inicializa o motor RAG com um modelo de embeddings.
        
        Args:
            model_name: Nome do modelo SentenceTransformer a ser usado para gerar embeddings
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.index_path = "data/faiss_index.bin"
        self.chunks_path = "data/chunks.pkl"
        
    def preprocess_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Divide o texto em chunks menores com sobreposição.
        
        Args:
            text: Texto completo do documento
            chunk_size: Tamanho aproximado de cada chunk (em caracteres)
            overlap: Tamanho da sobreposição entre chunks
            
        Returns:
            Lista de chunks de texto
        """
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
        
        # Agora dividir as páginas em chunks
        chunks = []
        
        for page in pages:
            # Se a página for menor que o chunk_size, adicione-a como um chunk
            if len(page) < chunk_size:
                chunks.append(page)
                continue
                
            # Dividir a página em parágrafos
            paragraphs = page.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                # Se adicionar o parágrafo ultrapassar o tamanho do chunk
                if len(current_chunk) + len(para) > chunk_size:
                    # Se o chunk atual não estiver vazio, adicione-o à lista
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Comece um novo chunk com a sobreposição
                        words = current_chunk.split()
                        overlap_text = " ".join(words[-min(len(words), overlap):])
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        # Se o parágrafo for maior que o chunk_size
                        if len(para) > chunk_size:
                            # Dividir o parágrafo em sentenças
                            sentences = para.replace(". ", ".\n").split("\n")
                            for i in range(0, len(sentences), chunk_size//20):
                                sentence_chunk = " ".join(sentences[i:i+chunk_size//20])
                                chunks.append(sentence_chunk)
                        else:
                            current_chunk = para
                else:
                    # Adicionar o parágrafo ao chunk atual
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Adicionar o último chunk se não estiver vazio
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
    
    def create_index(self, text: str) -> None:
        """
        Cria um índice FAISS a partir do texto do documento.
        
        Args:
            text: Texto completo do documento
        """
        # Preprocessar o texto em chunks
        self.chunks = self.preprocess_text(text)
        print(f"Documento dividido em {len(self.chunks)} chunks")
        
        # Gerar embeddings para cada chunk
        embeddings = []
        for chunk in self.chunks:
            embedding = self.embedding_model.encode(chunk)
            embeddings.append(embedding)
        
        # Criar índice FAISS
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        faiss.normalize_L2(np.array(embeddings).astype('float32'))
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Salvar o índice e chunks para uso futuro
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Índice criado e salvo em {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Carrega um índice FAISS previamente salvo.
        
        Returns:
            True se o índice foi carregado com sucesso, False caso contrário
        """
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"Índice carregado com {len(self.chunks)} chunks")
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar índice: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Busca os chunks mais relevantes para a consulta.
        
        Args:
            query: Consulta do usuário
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (índice, texto do chunk, score)
        """
        if self.index is None:
            raise ValueError("O índice não foi criado ou carregado")
        
        # Gerar embedding para a consulta
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Buscar chunks similares
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retornar resultados
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Verificar se o índice é válido
                results.append((int(idx), self.chunks[idx], float(distances[0][i])))
        
        return results
    
    def query_with_context(self, client, query: str, model: str, system_prompt: str, top_k: int = 5) -> str:
        """
        Realiza uma consulta usando RAG.
        
        Args:
            client: Cliente OpenAI
            query: Consulta do usuário
            model: Modelo a ser usado
            system_prompt: Prompt do sistema
            top_k: Número de chunks a recuperar
            
        Returns:
            Resposta da IA
        """
        # Buscar chunks relevantes
        relevant_chunks = self.search(query, top_k)
        
        # Preparar o contexto com os chunks mais relevantes
        context = "\n\n---\n\n".join([chunk for _, chunk, _ in relevant_chunks])
        
        # Construir prompt para a IA
        prompt = f"""
Com base EXCLUSIVAMENTE nos seguintes trechos do documento, responda à pergunta: '{query}'

Trechos do documento:
{context}

Responda apenas com base nas informações contidas nesses trechos. Se a informação não estiver presente, diga que não foi possível encontrar a resposta no documento.
"""
        
        # Realizar a consulta à API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def get_chunks_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre os chunks.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.chunks:
            return {"error": "Nenhum chunk disponível"}
        
        lengths = [len(chunk) for chunk in self.chunks]
        return {
            "total_chunks": len(self.chunks),
            "avg_chunk_size": sum(lengths) / len(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
            "total_content_size": sum(lengths)
        }
