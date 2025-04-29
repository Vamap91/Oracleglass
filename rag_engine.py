# rag_engine.py
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import openai

# Definir diretório para modelos
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './models'

class RAGEngine:
    def __init__(self, model_name="distilbert-base-nli-mean-tokens"):
        """
        Inicializa o motor RAG com abordagem de backup para ambientes restritos.
        """
        # Usar implementação simples sem dependências externas
        self.embedding_dim = 384
        self.index = None
        self.chunks = []
        self.index_path = "data/faiss_index.bin"
        self.chunks_path = "data/chunks.pkl"
        
        # Criar pasta data se não existir
        if not os.path.exists("data"):
            os.makedirs("data")
    
    def preprocess_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Divide o texto em chunks menores com sobreposição.
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
        Cria um índice simples de palavras-chave a partir do texto do documento.
        """
        # Preprocessar o texto em chunks
        self.chunks = self.preprocess_text(text)
        print(f"Documento dividido em {len(self.chunks)} chunks")
        
        # Não precisamos criar embeddings com modelos externos
        # Apenas salvar os chunks para uso futuro
        os.makedirs("data", exist_ok=True)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Criar um arquivo de índice vazio apenas para manter compatibilidade
        with open(self.index_path, 'wb') as f:
            pickle.dump({}, f)
        
        print(f"Chunks salvos em {self.chunks_path}")
    
    def load_index(self) -> bool:
        """
        Carrega os chunks previamente salvos.
        """
        try:
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"Chunks carregados: {len(self.chunks)} chunks")
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar chunks: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Busca os chunks mais relevantes para a consulta usando keywords.
        """
        if not self.chunks:
            raise ValueError("Os chunks não foram carregados")
        
        # Extrair palavras-chave da consulta
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        # Calcular relevância de cada chunk
        chunk_scores = []
        for i, chunk in enumerate(self.chunks):
            score = 0
            chunk_lower = chunk.lower()
            
            for keyword in keywords:
                count = chunk_lower.count(keyword)
                if count > 0:
                    score += count
            
            if score > 0:
                chunk_scores.append((i, chunk, score))
        
        # Ordenar por relevância
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Se não encontrou nada relevante, retornar alguns chunks aleatórios
        if not chunk_scores:
            import random
            random_indices = random.sample(range(len(self.chunks)), min(top_k, len(self.chunks)))
            chunk_scores = [(i, self.chunks[i], 0.1) for i in random_indices]
        
        # Limitar ao número de resultados solicitados
        return chunk_scores[:top_k]
    
    def query_with_context(self, client, query: str, model: str, system_prompt: str, top_k: int = 5) -> str:
        """
        Realiza uma consulta usando RAG simplificado.
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
        try:
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
        except Exception as e:
            return f"Erro ao processar consulta: {str(e)}"
    
    def get_chunks_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre os chunks.
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
