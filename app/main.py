# app/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware # Ajout pour la gestion des CORS

# --- Étape 1: Chargement des variables d'environnement ---
print("Chargement des variables d'environnement...")
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"SUPABASE_URL: {'Set' if supabase_url else 'Missing'}")
print(f"SUPABASE_KEY: {'Set' if supabase_key else 'Missing'}")
print(f"OPENAI_API_KEY: {'Set' if openai_api_key else 'Missing'}")

# --- Étape 5: Création de l'application API avec FastAPI ---
print("Création de l'application FastAPI...")
app = FastAPI()

# Configuration des CORS pour autoriser l'application mobile à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines (simple pour le développement)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de données pour la requête entrante (ce que l'app mobile va envoyer)
class Query(BaseModel):
    question: str

# Endpoint de test pour vérifier les variables d'environnement
@app.get("/test")
def test_environment():
    """Endpoint de test pour vérifier la configuration"""
    return {
        "supabase_url": "Set" if supabase_url else "Missing",
        "supabase_key": "Set" if supabase_key else "Missing", 
        "openai_api_key": "Set" if openai_api_key else "Missing",
        "status": "Ready" if all([supabase_url, supabase_key, openai_api_key]) else "Missing Environment Variables"
    }

# Endpoint "racine" pour vérifier que l'API est bien en ligne
@app.get("/")
def read_root():
    return {"message": "SuperDog Backend is running!"}

# Initialisation conditionnelle des services
qa_chain = None

if all([supabase_url, supabase_key, openai_api_key]):
    try:
        # --- Étape 2: Initialisation des clients et services ---
        print("Initialisation des services...")
        # Client Supabase
        supabase_client: Client = create_client(supabase_url, supabase_key)

        # Service d'embedding OpenAI
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

        # Modèle de langage (LLM) OpenAI pour la génération de réponses
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

        # Connexion à notre base de données vectorielle Supabase
        # C'est ici que nous nous connectons à la connaissance de SuperDog
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )

        # --- Étape 3: Définition du Prompt pour guider l'IA ---
        # C'est le "rôle" que l'on donne à SuperDog. C'est très important.
        prompt_template = """
        Tu es SuperDog, un assistant IA amical, positif et bienveillant, expert du monde canin.
        Ton but est d'aider les propriétaires de chiens en leur donnant des conseils clairs, simples et rassurants.
        Utilise uniquement les informations de contexte suivantes pour répondre à la question. Ne réponds pas si la question sort de ce contexte.
        Ton ton doit être encourageant. N'utilise jamais de termes techniques ou compliqués.
        Termine toujours tes réponses importantes par la phrase : "N'oubliez pas, SuperDog est un guide et ne remplace pas l'avis d'un vétérinaire."

        Contexte : {context}

        Question : {question}

        Réponse amicale :"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # --- Étape 4: Création de la chaîne RAG (Retrieval-Augmented Generation) ---
        # C'est le coeur de notre système. LangChain s'occupe de tout :
        # 1. Prendre la question de l'utilisateur.
        # 2. La transformer en vecteur et trouver les documents similaires dans Supabase (le "retrieval").
        # 3. Injecter ces documents dans le prompt.
        # 4. Envoyer le tout au LLM pour qu'il génère la réponse.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={'threshold': 0.35}), # Seuil de similarité pour la pertinence
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True # Pour vérifier si des documents ont été trouvés
        )
        print("Services initialisés avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'initialisation des services: {e}")
        qa_chain = None

# Endpoint principal pour poser des questions
@app.post("/ask")
def ask_superdog(query: Query):
    """
    Reçoit une question de l'application mobile, la traite avec la chaîne RAG
    et retourne la réponse de SuperDog.
    """
    if not all([supabase_url, supabase_key, openai_api_key]):
        raise HTTPException(status_code=500, detail="Backend not properly configured. Missing environment variables.")
    
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="Backend services not initialized.")
    
    try:
        print(f"Question reçue: {query.question}")
        result = qa_chain.invoke({"query": query.question})
        
        # Si la recherche dans Supabase n'a retourné aucun document pertinent (score < 0.35)
        if not result["source_documents"]:
            print("Aucun document pertinent trouvé. Réponse générique.")
            return {"answer": "Hum, cette question est un peu pointue ! Pour la santé et la sécurité de votre compagnon, je vous recommande de consulter directement un vétérinaire. Il saura vous donner la meilleure réponse."}

        print(f"Réponse générée: {result['result']}")
        return {"answer": result["result"]}
    except Exception as e:
        print(f"Erreur lors du traitement de la question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

print("API SuperDog prête à recevoir des requêtes.")
