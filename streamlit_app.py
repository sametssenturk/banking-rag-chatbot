
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="Banking RAG Chatbot",
    page_icon="🏦",
    layout="wide"
)

# Gemini API Key - Streamlit secrets veya ortam değişkeninden al
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

# RAG sistemini başlat (sadece ilk çalıştırmada)
@st.cache_resource
def load_rag_system():
    try:
        # ChromaDB path kontrolü
        chroma_db_path = "./chroma_db"
        
        # ChromaDB'yi yükle
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Koleksiyon var mı kontrol et
        try:
            collections = chroma_client.list_collections()
            if not any(col.name == "banking77_collection" for col in collections):
                st.error("❌ ChromaDB koleksiyonu bulunamadı! Lütfen önce Kaggle notebook'u çalıştırarak vektör veritabanını oluşturun.")
                return None
        except Exception as e:
            st.error(f"❌ ChromaDB koleksiyonu kontrol edilemedi: {str(e)}")
            return None
        
        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Vector store
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="banking77_collection",
            embedding_function=embeddings
        )
        
        # LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Prompt
        prompt_template = """
You are a helpful banking assistant. Use the context below to answer the question accurately and professionally.

If the question is related to the provided context, give a detailed and helpful answer.
If the question is not related to banking or the context, politely say that you can only help with banking-related questions.

Context:
{context}

Question: {question}

Answer:
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RAG Chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return rag_chain
    except Exception as e:
        st.error(f"RAG sistemi yüklenirken hata: {str(e)}")
        return None

# RAG sistemini yükle
rag_chain = load_rag_system()

# Başlık ve açıklama
st.title("🏦 Banking RAG Chatbot")
st.markdown("""
Bu chatbot, **Banking77** veri seti ve **RAG (Retrieval-Augmented Generation)** teknolojisi kullanılarak geliştirilmiştir.
Bankacılıkla ilgili sorularınızı **İngilizce** olarak sorabilirsiniz.
""")

# Sidebar - Bilgilendirme
with st.sidebar:
    st.header("ℹ️ Hakkında")
    st.info("""
    **Teknolojiler:**
    - 🤖 Gemini API (LLM)
    - 🧠 Sentence Transformers (Embedding)
    - 💾 ChromaDB (Vector DB)
    - 🔗 LangChain (RAG Framework)
    - 🎨 Streamlit (Web UI)
    """)
    
    st.header("💡 Örnek Sorular")
    example_questions = [
        "I lost my credit card, what should I do?",
        "How can I transfer money to another account?",
        "What are the fees for international transactions?",
        "How do I check my account balance?",
        "Can I change my PIN number?"
    ]
    
    for q in example_questions:
        if st.button(q, key=q, use_container_width=True):
            # Örnek soruyu chat input'a yaz
            st.session_state.example_question = q

# Chat geçmişini sakla
if "messages" not in st.session_state:
    st.session_state.messages = []

# Önceki mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Kaynak Dokümanlar"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}.** {doc}")

# Kullanıcı input'u
prompt = st.chat_input("Bankacılık sorunuzu buraya yazın...")

# Eğer örnek soru tıklandıysa onu kullan
if "example_question" in st.session_state and st.session_state.example_question:
    prompt = st.session_state.example_question
    st.session_state.example_question = None

if prompt:
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Chatbot yanıtı
    with st.chat_message("assistant"):
        if rag_chain is None:
            error_msg = "❌ RAG sistemi yüklenemedi. Lütfen API anahtarınızı kontrol edin."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("🤔 Düşünüyorum..."):
                try:
                    # RAG sisteminden yanıt al
                    result = rag_chain({"query": prompt})
                    response = result['result']
                    sources = [doc.page_content for doc in result.get('source_documents', [])]
                    
                    # Yanıtı göster
                    st.markdown(response)
                    
                    # Kaynak dokümanları göster
                    if sources:
                        with st.expander("📚 Kaynak Dokümanlar"):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"**{i}.** {doc}")
                    
                    # Yanıtı geçmişe ekle
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Üzgünüm, bir hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sohbeti temizle butonu
if st.sidebar.button("🗑️ Sohbeti Temizle", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with ❤️ using LangChain & Gemini API</p>
</div>
""", unsafe_allow_html=True)
