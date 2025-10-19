# 🏦 Banking RAG Chatbot

Kısa açıklama: Banking77 veri seti üzerinde RAG (Retrieval-Augmented Generation) yaklaşımı ile bankacılık sorularına alanına özel, kaynak destekli yanıtlar üreten bir sohbet botu.

## 🎯 Projenin Amacı
- Bankacılık alanındaki sık sorulan soru ve niyetleri (intent) kapsayan bir bilgi tabanını vektör veritabanında tutmak.
- Kullanıcı sorularını semantik olarak arayıp ilgili bağlamı LLM’e (Gemini) vererek doğru ve anlaşılır yanıt üretmek.

## 📚 Veri Seti
- Dataset: PolyAI/BANKING77 — çevrim içi bankacılık sorgularından oluşan ve her bir sorgunun niyeti (intent) ile etiketlendiği bir veri seti.
- Kapsam: 13,083 müşteri hizmetleri sorgusu, 77 ince taneli (fine-grained) niyet; tek alan (bankacılık) üzerinde yoğunlaşır.
- Dil: İngilizce
- Desteklenen görevler: Intent sınıflandırma/deteksiyon
- Yapı:
  - Alanlar: `text` (string), `label` (0–76 arası tamsayı; her biri benzersiz bir intent’i temsil eder)
  - Örnek:
    ```json
    {
      "label": 11,
      "text": "I am still waiting on my card?" // 11 -> "card_arrival"
    }
    ```
- Bölünmeler ve istatistikler:
  - Train: 10,003 örnek
  - Test: 3,080 örnek
  - Tek domain (bankacılık); ort. karakter uzunluğu ~59 (train), ~54 (test)
- Lisans ve atıf:
  - Lisans: Creative Commons Attribution 4.0 (CC BY 4.0)
  - Önerilen atıf: Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders (ACL 2020), https://arxiv.org/abs/2003.04807
- Kaynak: https://huggingface.co/datasets/PolyAI/banking77

Not: Veri İngilizce olduğundan, chatbot’tan en iyi performans İngilizce sorularda alınır.


## 🧪 Kullanılan Yöntemler
- RAG Pipeline: HuggingFace Embeddings (all-mpnet-base-v2) + ChromaDB (retriever) + Gemini (LLM)
- Arama: Semantik vektör benzerliği ile en alakalı parçaların getirilmesi (k=3)
- Üretim: Getirilen bağlam + kullanıcı sorusu ile özelleştirilmiş prompt üzerinden yanıt

## 📈 Elde Edilen Sonuçlar
- Banking77 alanında yüksek kapsama: 77 niyet, 13k+ örnekten üretilmiş vektör dizini
- Tutarlı bağlam kullanımı: Yanıtlarla birlikte kaynak dokümanların gösterimi (Streamlit UI)
- Hızlı ilk yanıt: all-mpnet-base-v2 + Chroma kalıcı indeks sayesinde düşük gecikme

## 🧰 Kullanılan Teknolojiler
- LLM: Google Gemini (gemini-2.5-flash)
- Embedding: sentence-transformers/all-mpnet-base-v2
- Vektör DB: ChromaDB (PersistentClient)
- RAG: LangChain (RetrievalQA + PromptTemplate)
- Arayüz: Streamlit
- Yardımcı: pandas, numpy, transformers, tiktoken

## 🧩 RAG Mimarisi
1) Soru → Embedding
2) ChromaDB’de en alakalı belgeler (k=3) → Bağlam
3) Bağlam + Soru → Gemini LLM → Yanıt

Kalıcı vektör veritabanı: `./chroma_db` (koleksiyon adı: `banking77_collection`).

---

## ▶️ Çalıştırma Kılavuzu

Bu proje iki parçadan oluşur ve rolleri farklıdır:
- `banking_rag_chatbot.ipynb`: Vektör veritabanını (ChromaDB) üretir ve zip’ler.
- `streamlit_app.py`: Üretilmiş veritabanını kullanarak web arayüzünde sohbet deneyimi sunar.

### 1) Vektör Veritabanı Üretimi — `banking_rag_chatbot.ipynb`
- Ortam: Kaggle (önerilir) veya yerel Jupyter
- Adımlar:
  1. Notebook’u aç ve ilk kurulum hücresini çalıştır (sabitlenmiş sürümlerle gerekli paketleri kurar).
  2. Veri yükleme ve embedding üretim hücrelerini sırayla çalıştır.
  3. ChromaDB oluşturma ve doldurma hücrelerini çalıştır (yol: `./chroma_db`, koleksiyon: `banking77_collection`).
  4. En sondaki ZIP hücresini çalıştır; `chroma_db.zip` oluşur. Bu dosyayı indirip depoya ekleyebilirsin.
- Not: LLM testleri için API anahtarı gerekir; sadece veritabanı üretmek için gerekmez.
- Notebook dosyasının kaggle üzerinde çalıştırılmış hali (çıktılarıyla birlikte): https://www.kaggle.com/code/sametsenturk/banking-rag-chatbot

### 2) Web Uygulaması — `streamlit_app.py`
- Gereklilik:
  - Depoda `./chroma_db` klasörü (veya `chroma_db.zip`’ten çıkarılmış hali) bulunmalı.
  - `GOOGLE_API_KEY` ortam değişkeni ya da `.streamlit/secrets.toml`’da tanımlı olmalı.
- Yerel çalıştırma (Windows PowerShell):
```powershell
# Gerekli paketler
pip install -r requirements.txt

# API anahtarı (geçici oturum için)
$env:GOOGLE_API_KEY = "YOUR_API_KEY"

# Uygulamayı başlat
streamlit run streamlit_app.py
```
- Uygulama açıldığında:
  - Soru alanına İngilizce bankacılık sorularınızı yazın veya yandaki örnek soruları tıklayın.
  - Yanıt ve kaynak dokümanlar arayüzde görüntülenir.

---

## 🖥️ Web Arayüzü
- Deploy Link: [https://banking-rag-chatbot.streamlit.app]
![Banking RAG Chatbot Demo](assets/demo.gif)

---

## ✅ Bu Dataset ile Neler Beklemeliyiz? Hangi Sorular Sorulmalı?
- Veri İngilizce bankacılık müşteri sorularından oluşur. Chatbot’tan en iyi performansı almak için:
  - Kart işlemleri: “I lost my credit card”, “How to activate my card?”, “Card not working”
  - Transfer/ödemeler: “How can I transfer money?”, “Why is my transfer pending?”
  - Ücretler/kur: “What are international transaction fees?”, “What is the exchange rate?”
  - Kimlik/limitler: “How to verify my identity?”, “What is my ATM withdrawal limit?”
- Alakasız veya domain dışı sorularda sistem kibarca bankacılık konularıyla sınırlı olduğunu belirtir.

---

## Notlar
- Chroma yapılandırması: Yol `./chroma_db`, koleksiyon `banking77_collection`.
- Embedding modeli: `sentence-transformers/all-mpnet-base-v2` (notebook ve Streamlit’te aynı).
- LLM: `gemini-2.5-flash` (GOOGLE_API_KEY gerektirir).

### Neden `chroma_db/` depoda?
Bu proje küçük ölçekli bir demo olduğundan ve Streamlit Cloud üzerinde hızlı deploy hedeflendiğinden, hazır olarak oluşturulmuş kalıcı ChromaDB klasörü (`./chroma_db`) depoya dahil edilmiştir. Böylece ilk açılışta indeks oluşturma süresi ve ek kurulumlar minimuma iner.

Alternatif (daha üretim-odaklı) seçenekler:
- Build aşamasında indeksi yeniden oluşturmak (örn. bir setup script’i ile)
- Uzak bir vektör veritabanı kullanmak (Pinecone, Qdrant, Weaviate vb.)
- `chroma_db.zip` gibi bir artefact’ı release’e koyup, uygulama başında açmak

Bu repoda amaç hızlı deneme/tekrar üretilebilirlik olduğundan `chroma_db/` versiyon kontrolüne dahil edilmiştir.


