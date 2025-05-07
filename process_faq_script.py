import os
import json
from rag_engine import RAGEngine

FAQ_QUESTIONS_FILE = "/home/ubuntu/OracleGlass/data/faq_extracted.txt"
PDF_PATH = "/home/ubuntu/OracleGlass/data/Guia Rápido.pdf"
OUTPUT_FAQ_PAIRS_FILE = "/home/ubuntu/OracleGlass/data/processed/faq_pairs.json"

def process_faqs_from_pdf():
    print(f"Iniciando o processamento do PDF: {PDF_PATH}")
    # Forçar o reprocessamento do PDF para garantir que estamos usando a versão mais recente
    # e que os dados processados (chunks, index) sejam atualizados.
    engine = RAGEngine()
    engine.load_and_process_pdf(PDF_PATH, force_reprocess=True)

    if not engine.vector_store or not engine.chunks:
        print("Falha ao processar o PDF. O vector store ou os chunks não estão disponíveis.")
        return

    print(f"Lendo perguntas do arquivo: {FAQ_QUESTIONS_FILE}")
    try:
        with open(FAQ_QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Arquivo de perguntas não encontrado: {FAQ_QUESTIONS_FILE}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo de perguntas: {e}")
        return

    faq_pairs = {}
    unanswered_questions = []

    print(f"Buscando respostas para {len(questions)} perguntas...")
    for i, question in enumerate(questions):
        print(f"Processando pergunta {i+1}/{len(questions)}: {question}")
        # Usar o método search do RAGEngine para encontrar chunks relevantes
        # Aumentar k para ter mais contexto, se necessário, e depois refinar.
        search_results = engine.search(question, k=3) 

        if search_results:
            # Por enquanto, vamos concatenar os chunks mais relevantes como resposta.
            # Idealmente, um modelo de linguagem resumiria ou extrairia a resposta precisa.
            answer_parts = []
            for chunk, score in search_results:
                answer_parts.append(chunk)
            
            # Concatenar os 3 chunks mais relevantes para formar a resposta.
            # Pode ser necessário um pós-processamento ou um LLM para gerar uma resposta mais concisa.
            combined_answer = "\n\n---\n\n".join(answer_parts)
            faq_pairs[question] = combined_answer
            print(f"  Resposta encontrada (baseada em chunks relevantes).")
        else:
            print(f"  Nenhuma resposta encontrada para: {question}")
            unanswered_questions.append(question)
    
    print(f"Salvando pares de FAQ em: {OUTPUT_FAQ_PAIRS_FILE}")
    try:
        with open(OUTPUT_FAQ_PAIRS_FILE, "w", encoding="utf-8") as f:
            json.dump(faq_pairs, f, ensure_ascii=False, indent=4)
        print("Pares de FAQ salvos com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar os pares de FAQ: {e}")

    if unanswered_questions:
        print("\nAs seguintes perguntas não puderam ser respondidas (nenhum chunk relevante encontrado):")
        for q in unanswered_questions:
            print(f"- {q}")
        print("\nPor favor, verifique se o conteúdo do 'Guia Rápido.pdf' cobre estas perguntas ou se elas precisam ser ajustadas.")
    else:
        print("\nTodas as perguntas foram processadas e tiveram chunks relevantes encontrados.")

if __name__ == "__main__":
    process_faqs_from_pdf()

