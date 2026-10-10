"""
🌾 UZHAVAN AI - RAG (Retrieval-Augmented Generation) System
Knowledge retrieval for agriculture, government schemes, and rural services.

Pipeline:
    User Question → Embedding → ChromaDB Vector Search → Top-K Docs → Response

Safety: Government scheme info includes source URLs, last-updated dates,
        and explicit disclaimers when information cannot be verified.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class KnowledgeBase:
    """Manages agricultural knowledge documents for RAG retrieval."""
    
    def __init__(self, agriculture_dir: str = "agriculture", chroma_db_path: str = "rag/chroma_db"):
        # FIX #2: Anchor all paths to PROJECT_ROOT so they resolve correctly
        # regardless of what directory uvicorn/the process is launched from.
        self.agriculture_dir = PROJECT_ROOT / agriculture_dir
        self.chroma_db_path = PROJECT_ROOT / chroma_db_path
        self.documents = []
        self.collection = None
        self._embedding_fn = None

    # FIX #4: Extract embedding function creation into its own method so it
    # can be re-used both when building a new store and when re-loading an
    # existing one (previously _embedding_fn stayed None on reload, causing
    # ChromaDB to use a mismatched function and crash into keyword search).
    def _get_embedding_fn(self):
        if self._embedding_fn is not None:
            return self._embedding_fn
        try:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:
            try:
                from chromadb.utils import embedding_functions
                self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                self._embedding_fn = None
        return self._embedding_fn
    
    def load_knowledge(self):
        """Load all knowledge documents from agriculture/ directory."""
        self.documents = []
        
        # Load crops
        crops_path = self.agriculture_dir / "crops.json"
        if crops_path.exists():
            with open(crops_path, "r", encoding="utf-8") as f:
                crops = json.load(f)
            for tamil_name, info in crops.items():
                doc = f"பயிர்: {tamil_name} ({info['english']})\n"
                doc += f"அறிவியல் பெயர்: {info.get('scientific', 'N/A')}\n"
                doc += f"பருவம்: {', '.join(info.get('season', []))}\n"
                doc += f"பகுதிகள்: {', '.join(info.get('regions', []))}"
                self.documents.append({
                    "text": doc,
                    "source": "agriculture/crops.json",
                    "topic": "crop_info",
                    "crop": tamil_name,
                })
        
        # Load diseases
        diseases_path = self.agriculture_dir / "diseases.json"
        if diseases_path.exists():
            with open(diseases_path, "r", encoding="utf-8") as f:
                diseases = json.load(f)
            for crop, info in diseases.items():
                for disease in info.get("diseases", []):
                    doc = f"பயிர்: {crop}\n"
                    doc += f"நோய்: {disease['tamil']} ({disease['english']})\n"
                    doc += f"அறிகுறிகள்: {disease.get('symptoms', '')}\n"
                    doc += f"சிகிச்சை: {disease.get('treatment', '')}"
                    self.documents.append({
                        "text": doc,
                        "source": "agriculture/diseases.json",
                        "topic": "crop_disease",
                        "crop": crop,
                    })
        
        # Load fertilizers
        fertilizers_path = self.agriculture_dir / "fertilizers.json"
        if fertilizers_path.exists():
            with open(fertilizers_path, "r", encoding="utf-8") as f:
                fertilizers = json.load(f)
            for fert in fertilizers.get("fertilizers", []):
                doc = f"உரம்: {fert['tamil']} ({fert['english']})\n"
                doc += f"வகை: {fert.get('type', '')}\n"
                doc += f"ஊட்டச்சத்து: {fert.get('nutrient', '')}\n"
                doc += f"பயன்பாடு: {fert.get('usage', '')}"
                self.documents.append({
                    "text": doc,
                    "source": "agriculture/fertilizers.json",
                    "topic": "fertilizer",
                })
        
        # Load government schemes (with safety disclaimers)
        schemes_path = self.agriculture_dir / "government_schemes.json"
        if schemes_path.exists():
            with open(schemes_path, "r", encoding="utf-8") as f:
                schemes = json.load(f)
            for scheme in schemes:
                doc = f"அரசு திட்டம்: {scheme['scheme_name_tamil']}\n"
                doc += f"({scheme['scheme_name_english']})\n"
                doc += f"தகுதி: {scheme.get('eligibility', '')}\n"
                doc += f"பயன்கள்: {scheme.get('benefits', '')}\n"
                doc += f"விண்ணப்பம்: {scheme.get('application_procedure', '')}\n"
                doc += f"இணையதளம்: {scheme.get('source_url', '')}\n"
                doc += f"\n⚠️ {scheme.get('disclaimer', 'உத்தியோகபூர்வ இணையதளத்தில் சரிபார்க்கவும்')}"
                self.documents.append({
                    "text": doc,
                    "source": "agriculture/government_schemes.json",
                    "topic": "government_scheme",
                    "source_url": scheme.get("source_url", ""),
                    "verified": scheme.get("verified", False),
                })
        
        print(f"📚 Loaded {len(self.documents)} knowledge documents")
        return self.documents
    
    def build_vector_store(self):
        """Build ChromaDB vector store from documents."""
        try:
            import chromadb
            
            self.chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            client = chromadb.PersistentClient(path=str(self.chroma_db_path))
            embedding_fn = self._get_embedding_fn()
            
            # Delete existing collection if exists
            try:
                client.delete_collection("uzhavan_knowledge")
            except Exception:
                pass
            
            kwargs = {"name": "uzhavan_knowledge", "metadata": {"description": "UZHAVAN AI Agricultural Knowledge Base"}}
            if embedding_fn is not None:
                kwargs["embedding_function"] = embedding_fn
            self.collection = client.create_collection(**kwargs)
            
            # Add documents
            texts = [doc["text"] for doc in self.documents]
            metadatas = [{k: str(v) for k, v in doc.items() if k != "text"} for doc in self.documents]
            ids = [f"doc_{i}" for i in range(len(self.documents))]
            
            # Add in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                self.collection.add(
                    documents=texts[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size],
                )
            
            print(f"✅ Vector store built: {len(texts)} documents indexed")
            return True
            
        except ImportError:
            print("⚠️ ChromaDB not installed. Run: pip install chromadb")
            return False
    
    def retrieve(self, query: str, top_k: int = 5, topic_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Tamil or English query text
            top_k: Number of documents to retrieve
            topic_filter: Optional filter by topic (crop_info, crop_disease, fertilizer, government_scheme)
        
        Returns:
            List of matching documents with scores
        """
        if self.collection is None:
            # Try to load existing collection
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(self.chroma_db_path))
                # FIX #4: Pass the embedding function so ChromaDB uses the
                # same model that was used at index time; previously this was
                # None here, causing a dimension mismatch crash on every cold
                # reload and silently falling through to keyword search.
                embedding_fn = self._get_embedding_fn()
                kwargs = {"name": "uzhavan_knowledge"}
                if embedding_fn is not None:
                    kwargs["embedding_function"] = embedding_fn
                self.collection = client.get_collection(**kwargs)
            except Exception as e:
                print(f"⚠️ Could not load ChromaDB collection: {e} — using keyword search")
                return self._keyword_search(query, top_k)
        
        where_filter = None
        if topic_filter:
            where_filter = {"topic": topic_filter}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
            
            documents = []
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback keyword-based search when vector DB is unavailable."""
        if not self.documents:
            self.load_knowledge()
        
        stop_words = {"what", "are", "the", "types", "of", "in", "is", "a", "an", "for", "on", "to", "and", "or"}
        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if w not in stop_words and len(w) > 1)
        
        # Topic hints
        crop_keywords = {"crop", "crops", "plant", "planted", "grow", "paddy", "rice", "sugarcane", "cotton", "millet", "harvest"}
        fert_keywords = {"fertilizer", "fertilizers", "urea", "dap", "potash", "manure", "vermicompost", "nutrient", "npk"}
        scheme_keywords = {"scheme", "schemes", "government", "pmkisan", "pmfby", "subsidy", "kcc", "electricity", "insurance"}

        is_crop_query = any(w in crop_keywords for w in query_words) or "பயிர்" in query or "நெல்" in query
        is_fert_query = any(w in fert_keywords for w in query_words) or "உரம்" in query
        is_scheme_query = any(w in scheme_keywords for w in query_words) or "திட்டம்" in query or "அரசு" in query

        scored = []
        for doc in self.documents:
            text_lower = doc["text"].lower()
            topic = doc.get("topic", "")
            score = 0.0

            for word in query_words:
                if word in text_lower:
                    score += 2.0

            if is_crop_query and topic == "crop_info":
                score += 5.0
            elif is_fert_query and topic == "fertilizer":
                score += 5.0
            elif is_scheme_query and topic == "government_scheme":
                score += 5.0

            if score > 0:
                scored.append((score, doc))
        
        scored.sort(key=lambda x: -x[0])
        
        if not scored:
            scored = [(1.0, doc) for doc in self.documents if doc.get("topic") == "crop_info"]

        return [
            {"text": doc["text"], "metadata": {k: str(v) for k, v in doc.items() if k != "text"}, "distance": 1.0 / (score + 1)}
            for score, doc in scored[:top_k]
        ]


class ResponseGenerator:
    """
    Generate Tamil responses from retrieved documents.
    
    Supports:
    - Template-based responses (fully offline, no LLM needed)
    - Optional LLM integration for more natural responses
    """
    
    RESPONSE_TEMPLATES = {
        "crop_disease": "🌱 {crop} பயிரில் {disease} நோய் இருக்கலாம்.\n\nஅறிகுறிகள்: {symptoms}\n\nசிகிச்சை: {treatment}\n\n⚠️ இது மாதிரி மதிப்பீடு மட்டுமே. வேளாண் அலுவலரை அணுகவும்.",
        "fertilizer": "🧪 {info}\n\n⚠️ உள்ளூர் மண் பரிசோதனை அடிப்படையில் உரம் போடுவது சிறந்தது.",
        "government_scheme": "🏛️ {info}\n\n⚠️ இந்த தகவல் பொது வழிகாட்டுதல் மட்டுமே. உத்தியோகபூர்வ இணையதளத்தில் சரிபார்க்கவும்: {url}",
        "crop_info": "🌾 {info}",
        "general": "📋 {info}",
        "not_found": "மன்னிக்கவும், இந்த கேள்விக்கு என்னிடம் போதுமான தகவல் இல்லை. வேளாண் உதவி எண்ணை தொடர்பு கொள்ளவும்.",
        "unverified": "⚠️ இந்த தகவலை சரிபார்க்க இயலவில்லை. உத்தியோகபூர்வ ஆதாரங்களில் உறுதி செய்யவும்.",
    }
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        intent: Optional[str] = None,
    ) -> Dict:
        """
        Generate a dual-language (Tamil + English) response from retrieved documents and Ollama LLM.
        
        Returns:
            {"response": str, "response_ta": str, "response_en": str, "sources": list, "confidence": float}
        """
        if not retrieved_docs:
            not_found_ta = "மன்னிக்கவும், இந்த கேள்விக்கு என்னிடம் போதுமான தகவல் இல்லை. வேளாண் உதவி எண்ணை தொடர்பு கொள்ளவும்."
            not_found_en = "Sorry, I do not have enough information for this query. Please contact the agricultural help desk."
            combined = f"1. 🇮🇳 தமிழ்:\n{not_found_ta}\n\n2. 🇬🇧 English:\n{not_found_en}"
            return {
                "response": combined,
                "response_ta": not_found_ta,
                "response_en": not_found_en,
                "sources": [],
                "confidence": 0.0,
            }
        
        # Collect context & sources
        context_texts = []
        sources = []
        for doc in retrieved_docs[:3]:
            context_texts.append(doc.get("text", ""))
            source = doc.get("metadata", {}).get("source", "")
            if source and source not in sources:
                sources.append(source)
        
        context_str = "\n---\n".join(context_texts)
        best_doc = retrieved_docs[0]
        metadata = best_doc.get("metadata", {})
        topic = metadata.get("topic", "general")
        verified = metadata.get("verified", "true").lower() == "true"

        # FIX #1 + #3: Corrected import path and added detailed error logging
        # so failures are visible in the terminal instead of silently falling
        # through to the offline fallback with no diagnostics.
        try:
            # Use a direct import anchored to PROJECT_ROOT (already on sys.path)
            # rather than "backend.services.ollama_service" which is fragile
            # when the process CWD differs from PROJECT_ROOT.
            from backend.services.ollama_service import get_ollama_service
            ollama = get_ollama_service()
            available = ollama.is_available()
            if not available:
                print(f"[RAG] Ollama not reachable at {ollama.host} — using offline fallback")
            else:
                print(f"[RAG] Sending query to Ollama model '{ollama.model}'...")
                ollama_res = ollama.generate_dual_response(query, context_str)
                if ollama_res:
                    res_ta = ollama_res["ta"]
                    res_en = ollama_res["en"]
                    combined_res = f"1. 🇮🇳 தமிழ்:\n{res_ta}\n\n2. 🇬🇧 English:\n{res_en}"
                    print(f"[RAG] ✅ Ollama responded successfully")
                    return {
                        "response": combined_res,
                        "response_ta": res_ta,
                        "response_en": res_en,
                        "sources": sources,
                        "confidence": 0.95,
                        "verified": verified if topic == "government_scheme" else True,
                    }
                else:
                    print("[RAG] Ollama returned empty response — using offline fallback")
        except ImportError as e:
            print(f"[RAG] ❌ Could not import ollama_service: {e}")
        except Exception as e:
            print(f"[RAG] ❌ Ollama call failed ({type(e).__name__}): {e}")

        # Offline Dual-Language Fallback
        print("[RAG] Using offline keyword-based fallback response")
        doc_text = best_doc["text"]
        
        res_ta = doc_text
        if topic == "government_scheme" and not verified:
            res_ta += f"\n\n⚠️ {metadata.get('disclaimer', 'உத்தியோகபூர்வ இணையதளத்தில் சரிபார்க்கவும்')}"
        
        res_en = doc_text \
            .replace("பயிர்:", "Crop:") \
            .replace("அறிவியல் பெயர்:", "Scientific Name:") \
            .replace("பருவம்:", "Season:") \
            .replace("பகுதிகள்:", "Regions:") \
            .replace("உரம்:", "Fertilizer:") \
            .replace("வகை:", "Type:") \
            .replace("ஊட்டச்சத்து:", "Nutrient:") \
            .replace("பயன்பாடு:", "Usage:") \
            .replace("அரசு திட்டம்:", "Government Scheme:") \
            .replace("தகுதி:", "Eligibility:") \
            .replace("பயன்கள்:", "Benefits:") \
            .replace("விண்ணப்பம்:", "Application:") \
            .replace("இணையதளம்:", "Website:") \
            .replace("நோய்:", "Disease:") \
            .replace("அறிகுறிகள்:", "Symptoms:") \
            .replace("சிகிச்சை:", "Treatment:")

        combined_res = f"1. 🇮🇳 தமிழ்:\n{res_ta}\n\n2. 🇬🇧 English:\n{res_en}"

        return {
            "response": combined_res,
            "response_ta": res_ta,
            "response_en": res_en,
            "sources": sources,
            "confidence": 1.0 - best_doc.get("distance", 0.5),
            "verified": verified if topic == "government_scheme" else True,
        }


class RAGPipeline:
    """End-to-end RAG pipeline for UZHAVAN AI."""
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.generator = ResponseGenerator()
        self._initialized = False
    
    def initialize(self):
        """Load knowledge and build vector store."""
        self.kb.load_knowledge()
        self.kb.build_vector_store()
        self._initialized = True
    
    def query(
        self,
        question: str,
        intent: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Full RAG query: retrieve + generate response.
        
        Args:
            question: User's question in Tamil/English
            intent: Detected intent (helps filter relevant docs)
            top_k: Number of documents to retrieve
        """
        if not self._initialized:
            self.initialize()
        
        # Map intent to topic filter
        topic_map = {
            "crop_advice": "crop_info",
            "crop_disease": "crop_disease",
            "fertilizer": "fertilizer",
            "government_scheme": "government_scheme",
            "subsidy": "government_scheme",
            "insurance": "government_scheme",
        }
        topic_filter = topic_map.get(intent)
        
        # Retrieve
        docs = self.kb.retrieve(question, top_k=top_k, topic_filter=topic_filter)
        
        # Generate
        response = self.generator.generate_response(question, docs, intent)
        
        return response


# Singleton instance
_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
