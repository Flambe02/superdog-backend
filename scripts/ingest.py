# scripts/ingest.py (Version finale avec traitement par lots)
import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.supabase import SupabaseVectorStore

print("--- Début du script d'ingestion (v3 - avec traitement par lots) ---")

# --- Étape 1: Chargement des variables d'environnement ---
print("1. Tentative de chargement du fichier .env...")
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"   - SUPABASE_URL: {'Trouvée' if supabase_url else 'Non trouvée'}")
print(f"   - SUPABASE_KEY: {'Trouvée' if supabase_key else 'Non trouvée'}")
print(f"   - OPENAI_API_KEY: {'Trouvée' if openai_api_key else 'Non trouvée'}")

if not all([supabase_url, supabase_key, openai_api_key]):
    print("\nERREUR CRITIQUE: Une ou plusieurs clés API sont manquantes.")
    print("--- Fin du script ---")
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
    print("--- Fin du script ---")
    exit()


def ingest_data():
    """
    Lit les documents, les découpe et les stocke dans Supabase.
    """
    print("\n--- Démarrage de la fonction ingest_data ---")
    
    # --- Étape 3: Chargement des documents locaux ---
    docs_path = 'docs/'
    print(f"\n3. Recherche de documents dans le dossier '{docs_path}'...")
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
                all_texts_with_source.append({'text': f.read(), 'source': filename})

    if not all_texts_with_source:
        print("ERREUR: Aucun document valide à traiter.")
        return
        
    print(f"   => {len(all_texts_with_source)} document(s) chargé(s) avec succès.")

    # --- Étape 4: Préparation et découpage des documents ---
    print("\n4. Préparation et découpage des documents...")
    from langchain.docstore.document import Document
    
    documents_to_process = [Document(page_content=item['text'], metadata={'source': item['source']}) for item in all_texts_with_source]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs_splitted = text_splitter.split_documents(documents_to_process)
    total_chunks = len(docs_splitted)
    print(f"   => Documents découpés en {total_chunks} morceaux (chunks).")

    # --- NOUVELLE Étape 5: Création des embeddings et stockage PAR LOTS ---
    print("\n5. Création des embeddings et stockage dans Supabase par lots...")
    
    batch_size = 50 # Nous allons envoyer 50 chunks à la fois
    
    # Initialisation du VectorStore SANS documents au départ
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )

    for i in range(0, total_chunks, batch_size):
        # Sélectionne un lot de documents
        batch = docs_splitted[i:i + batch_size]
        
        print(f"   - Traitement du lot {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} (chunks {i+1} à {min(i+batch_size, total_chunks)})...")
        
        try:
            # Ajoute le lot actuel au VectorStore
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"\nERREUR CRITIQUE lors du traitement du lot: {e}")
            print("Le script va s'arrêter. Les lots précédents ont peut-être été enregistrés.")
            print("--- Fin du script ---")
            return
            
    print("\n--- Ingestion terminée avec succès ! ---")
    print("--- Fin du script ---")


if __name__ == "__main__":
    ingest_data()
