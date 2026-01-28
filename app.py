"""
RAGFlow Semantic Search - Streamlit Application
Приложение для поиска семантически близких чанков через RAGFlow API.
"""

import streamlit as st
from ragflow_client import RAGFlowClient, RAGFlowError, Chunk


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="RAGFlow Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .chunk-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .chunk-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }
    
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .chunk-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .similarity-badge.medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .similarity-badge.low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .chunk-content {
        color: #cbd5e1;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    .chunk-meta {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        font-size: 0.8rem;
        color: #64748b;
    }
    
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* Connection status */
    .status-connected {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-disconnected {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: rgba(30, 41, 59, 0.5);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Search input */
    .stTextInput > div > div > input {
        background: #1e293b !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #64748b;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================
if 'client' not in st.session_state:
    st.session_state.client = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'datasets' not in st.session_state:
    st.session_state.datasets = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'ai_summary' not in st.session_state:
    st.session_state.ai_summary = ""
if 'mind_map' not in st.session_state:
    st.session_state.mind_map = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""


# ============================================================================
# Sidebar - Configuration
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Настройки подключения")
    
    ragflow_url = st.text_input(
        "🌐 RAGFlow URL",
        value="http://localhost:9380",
        placeholder="http://localhost:9380"
    )
    
    api_key = st.text_input("🔑 API Key", type="password")
    
    if st.button("🔌 Подключиться", use_container_width=True):
        if ragflow_url and api_key:
            try:
                client = RAGFlowClient(ragflow_url, api_key)
                datasets = client.list_datasets()
                st.session_state.client = client
                st.session_state.connected = True
                st.session_state.datasets = datasets
                st.success("✅ Подключено!")
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
    
    # Advanced Settings
    if st.session_state.connected:
        st.markdown("---")
        st.markdown("## 🚀 Продвинутые функции")
        
        use_rerank = st.checkbox("🔄 Включить Rerank", value=False)
        rerank_id = st.text_input("🆔 Rerank Model ID", help="ID модели переранжирования из вашего профиля") if use_rerank else None
        
        use_summary = st.checkbox("🤖 ИИ-резюме", value=False)
        assistant_id = st.text_input("🤖 Assistant ID", help="ID ассистента для генерации резюме") if use_summary else None
        
        use_kg_search = st.checkbox("🕸️ Связный поиск (KG)", value=False, help="Использовать Knowledge Graph")
        show_mind_map = st.checkbox("🗺️ Показать Mind Map", value=False)

        st.markdown("---")
        st.markdown("## 📚 Выбор датасетов")
        dataset_options = {d.get('name', d.get('id')): d.get('id') for d in st.session_state.datasets}
        selected_datasets = st.multiselect("Датасеты", options=list(dataset_options.keys()), default=list(dataset_options.keys())[:1] if dataset_options else [])
        st.session_state.selected_dataset_ids = [dataset_options[name] for name in selected_datasets]

    # Search parameters
    st.markdown("---")
    st.markdown("## 🎛️ Параметры поиска")
    top_k = st.slider("📊 Top K", 1, 50, 5)
    similarity_threshold = st.slider("🎯 Порог", 0.0, 1.0, 0.2, 0.05)
    vector_weight = st.slider("⚖️ Вес вектора", 0.0, 1.0, 0.3, 0.1)
    use_highlight = st.checkbox("✨ Подсветка", value=True)
    use_keyword = st.checkbox("🔤 Ключевые слова", value=False)


# ============================================================================
# Main Content
# ============================================================================
st.markdown("<h1 class='main-header'>🔍 RAGFlow Advanced Search</h1>", unsafe_allow_html=True)

# Search input
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Запрос", label_visibility="collapsed")
with col2:
    search_clicked = st.button("🔎 Искать", use_container_width=True, type="primary")

if search_clicked and query:
    if not st.session_state.connected:
        st.error("❌ Подключитесь в боковой панели")
    elif not st.session_state.selected_dataset_ids:
        st.error("❌ Выберите датасет")
    else:
        with st.spinner("🔄 Получение данных..."):
            try:
                # 1. Retrieval
                chunks = st.session_state.client.search(
                    question=query,
                    dataset_ids=st.session_state.selected_dataset_ids,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    vector_similarity_weight=vector_weight,
                    highlight=use_highlight,
                    keyword=use_keyword,
                    use_kg=use_kg_search,
                    rerank_id=rerank_id
                )
                st.session_state.search_results = chunks
                st.session_state.last_query = query
                
                # 2. AI Summary
                if use_summary and assistant_id:
                    summary_resp = st.session_state.client.get_ai_summary(assistant_id, query)
                    st.session_state.ai_summary = summary_resp.get("data", {}).get("answer", "")
                else:
                    st.session_state.ai_summary = ""
                
                # 3. Mind Map
                if show_mind_map:
                    # Get for the first dataset
                    kg_data = st.session_state.client.get_mind_map(st.session_state.selected_dataset_ids[0])
                    st.session_state.mind_map = kg_data.get("mind_map")
                else:
                    st.session_state.mind_map = None
                    
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")

# Display Results
if st.session_state.search_results:
    # 1. AI Summary Section
    if st.session_state.ai_summary:
        st.markdown("### 🤖 ИИ-резюме")
        st.info(st.session_state.ai_summary)
    
    # 2. Mind Map Section
    if st.session_state.mind_map:
        st.markdown("### 🗺️ Mind Map")
        with st.expander("Показать структуру ментальной карты"):
            st.json(st.session_state.mind_map)

    # 3. Stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.search_results)}</div>
            <div class="stat-label">Найдено чанков</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sim = sum(c.similarity for c in st.session_state.search_results) / len(st.session_state.search_results) if st.session_state.search_results else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_sim:.1%}</div>
            <div class="stat-label">Средняя схожесть</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_sim = max(c.similarity for c in st.session_state.search_results) if st.session_state.search_results else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{max_sim:.1%}</div>
            <div class="stat-label">Макс. схожесть</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_docs = len(set(c.document_name for c in st.session_state.search_results))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{unique_docs}</div>
            <div class="stat-label">Документов</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Chunks
    for i, chunk in enumerate(st.session_state.search_results, 1):
        badge_class = "" if chunk.similarity >= 0.7 else "medium" if chunk.similarity >= 0.4 else "low"
        display_content = chunk.highlight if chunk.highlight and use_highlight else chunk.content
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-header">
                <span class="chunk-title">📄 {chunk.document_name}</span>
                <span class="similarity-badge {badge_class}">{chunk.similarity:.1%}</span>
            </div>
            <div class="chunk-content">{display_content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expander for raw data
        with st.expander(f"📋 Подробности чанка #{i}"):
            st.json({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "similarity": chunk.similarity,
                "vector_similarity": chunk.vector_similarity,
                "term_similarity": chunk.term_similarity,
                "content_length": len(chunk.content)
            })
            st.text_area("Полный текст", chunk.content, height=150, key=f"content_{i}")

elif query and search_clicked:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3>Ничего не найдено</h3>
        <p>Попробуйте изменить запрос или снизить порог схожести</p>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.connected:
    st.info("👈 Настройте подключение к RAGFlow в боковой панели")

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💡</div>
        <h3>Готово к поиску</h3>
        <p>Введите запрос и нажмите "Искать" для получения семантически близких чанков</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "RAGFlow Semantic Search • Powered by RAGFlow API"
    "</div>",
    unsafe_allow_html=True
)
