# scripts/ingest.py (Version finale v5 avec nettoyage et réessai automatique)
import os
import time
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.supabase import SupabaseVectorStore

print("--- Début du script d'ingestion (v5 - avec réessai automatique) ---")

# --- Étape 1: Chargement des variables d'environnement ---
print("1. Tentative de chargement du fichier .env...")
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not all([supabase_url, supabase_key, openai_api_key]):
    print("\nERREUR CRITIQUE: Une ou plusieurs clés API sont manquantes.")
    exit()

print("   => Clés API chargées avec succès.")

# --- Étape 2: Initialisation des clients ---
print("\n2. Initialisation des clients (Supabase & OpenAI)...")
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    print("   => Clients initialisés avec succès.")
except Exception as e:
    print(f"\nERREUR CRITIQUE lors de l'initialisation des clients: {e}")
    exit()


def ingest_data():
    """
    Lit les documents, les nettoie, les découpe et les stocke dans Supabase.
    """
    print("\n--- Démarrage de la fonction ingest_data ---")
    
    # --- Étape 3: Chargement et NETTOYAGE des documents locaux ---
    docs_path = 'docs/'
    print(f"\n3. Recherche et nettoyage de documents dans le dossier '{docs_path}'...")
    all_texts_with_source = []
    try:
        filenames = os.listdir(docs_path)
    except FileNotFoundError:
        print(f"ERREUR: Le dossier '{docs_path}' n'a pas été trouvé.")
        return
    
    for filename in filenames:
        if filename.endswith(('.txt', '.md')):
            file_path = os.path.join(docs_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                cleaned_text = text.replace('\u0000', '')
                all_texts_with_source.append({'text': cleaned_text, 'source': filename})

    if not all_texts_with_source:
        print("ERREUR: Aucun document valide à traiter.")
        return
        
    print(f"   => {len(all_texts_with_source)} document(s) chargé(s) et nettoyé(s) avec succès.")

    # --- Étape 4: Préparation et découpage des documents ---
    print("\n4. Préparation et découpage des documents...")
    from langchain.docstore.document import Document
    
    documents_to_process = [Document(page_content=item['text'], metadata={'source': item['source']}) for item in all_texts_with_source]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs_splitted = text_splitter.split_documents(documents_to_process)
    total_chunks = len(docs_splitted)
    print(f"   => Documents découpés en {total_chunks} morceaux (chunks).")

    # --- Étape 5: Création des embeddings et stockage PAR LOTS avec RÉESSAI ---
    print("\n5. Création des embeddings et stockage dans Supabase par lots...")
    
    batch_size = 40  # Réduction de la taille pour plus de stabilité
    max_retries = 5 # Nombre maximum de tentatives par lot
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )

    for i in range(0, total_chunks, batch_size):
        batch = docs_splitted[i:i + batch_size]
        
        for attempt in range(max_retries):
            try:
                print(f"   - Traitement du lot {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}, tentative {attempt + 1}/{max_retries}...")
                vector_store.add_documents(batch)
                print("     => Lot traité avec succès.")
                break # Sortir de la boucle de réessai si c'est un succès
            except Exception as e:
                print(f"     - AVERTISSEMENT: Erreur lors du traitement du lot: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1) # Attendre 5s, 10s, 15s...
                    print(f"     - Nouvel essai dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    print("\nERREUR CRITIQUE: Le nombre maximum de tentatives a été atteint pour ce lot.")
                    print("Le script va s'arrêter.")
                    return
            
    print("\n--- Ingestion terminée avec succès ! ---")
    print("--- Fin du script ---")


if __name__ == "__main__":
    ingest_data()
