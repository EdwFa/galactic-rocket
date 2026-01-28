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
        placeholder="http://localhost:9380",
        help="URL адрес вашего RAGFlow сервера"
    )
    
    api_key = st.text_input(
        "🔑 API Key",
        type="password",
        placeholder="Введите API ключ",
        help="API ключ можно получить в настройках RAGFlow"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔌 Подключиться", use_container_width=True):
            if ragflow_url and api_key:
                try:
                    client = RAGFlowClient(ragflow_url, api_key)
                    # Try to get datasets to verify connection
                    datasets = client.list_datasets()
                    st.session_state.client = client
                    st.session_state.connected = True
                    st.session_state.datasets = datasets
                    st.success("✅ Подключено!")
                except RAGFlowError as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    st.session_state.connected = False
                except Exception as e:
                    st.error(f"❌ Ошибка подключения: {str(e)}")
                    st.session_state.connected = False
            else:
                st.warning("⚠️ Заполните URL и API Key")
    
    with col2:
        if st.button("🔄 Обновить", use_container_width=True, disabled=not st.session_state.connected):
            if st.session_state.client:
                try:
                    st.session_state.datasets = st.session_state.client.list_datasets()
                    st.success("✅ Обновлено!")
                except RAGFlowError as e:
                    st.error(f"❌ {str(e)}")
    
    # Connection status
    st.markdown("---")
    if st.session_state.connected:
        st.markdown("**Статус:** <span class='status-connected'>● Подключено</span>", unsafe_allow_html=True)
        st.markdown(f"**Датасетов:** {len(st.session_state.datasets)}")
    else:
        st.markdown("**Статус:** <span class='status-disconnected'>● Отключено</span>", unsafe_allow_html=True)
    
    # Dataset selection
    if st.session_state.connected and st.session_state.datasets:
        st.markdown("---")
        st.markdown("## 📚 Выбор датасетов")
        
        dataset_options = {d.get('name', d.get('id')): d.get('id') for d in st.session_state.datasets}
        
        selected_datasets = st.multiselect(
            "Датасеты для поиска",
            options=list(dataset_options.keys()),
            default=list(dataset_options.keys())[:1] if dataset_options else [],
            help="Выберите один или несколько датасетов"
        )
        
        st.session_state.selected_dataset_ids = [dataset_options[name] for name in selected_datasets]
    
    # Search parameters
    st.markdown("---")
    st.markdown("## 🎛️ Параметры поиска")
    
    top_k = st.slider(
        "📊 Количество результатов",
        min_value=1,
        max_value=50,
        value=5,
        help="Максимальное количество возвращаемых чанков"
    )
    
    similarity_threshold = st.slider(
        "🎯 Порог схожести",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Минимальный уровень схожести для включения в результаты"
    )
    
    vector_weight = st.slider(
        "⚖️ Вес векторного сходства",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Баланс между векторным и терминологическим сходством"
    )
    
    use_highlight = st.checkbox("✨ Подсветка терминов", value=True)
    use_keyword = st.checkbox("🔤 Поиск по ключевым словам", value=False)


# ============================================================================
# Main Content
# ============================================================================
st.markdown("<h1 class='main-header'>🔍 RAGFlow Semantic Search</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Поиск семантически близких чанков в вашей базе знаний</p>", unsafe_allow_html=True)

# Search input
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Поисковый запрос",
        placeholder="Введите ваш вопрос или ключевые слова...",
        label_visibility="collapsed"
    )

with col2:
    search_clicked = st.button("🔎 Искать", use_container_width=True, type="primary")

# Perform search
if search_clicked and query:
    if not st.session_state.connected:
        st.error("❌ Сначала подключитесь к RAGFlow")
    elif not hasattr(st.session_state, 'selected_dataset_ids') or not st.session_state.selected_dataset_ids:
        st.error("❌ Выберите хотя бы один датасет")
    else:
        with st.spinner("🔄 Выполняется поиск..."):
            try:
                chunks = st.session_state.client.search(
                    question=query,
                    dataset_ids=st.session_state.selected_dataset_ids,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    vector_similarity_weight=vector_weight,
                    highlight=use_highlight,
                    keyword=use_keyword
                )
                st.session_state.search_results = chunks
                st.session_state.last_query = query
            except RAGFlowError as e:
                st.error(f"❌ Ошибка поиска: {str(e)}")
            except Exception as e:
                st.error(f"❌ Неожиданная ошибка: {str(e)}")

# Display results
if st.session_state.search_results:
    chunks = st.session_state.search_results
    
    # Statistics
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(chunks)}</div>
            <div class="stat-label">Найдено чанков</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sim = sum(c.similarity for c in chunks) / len(chunks) if chunks else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_sim:.1%}</div>
            <div class="stat-label">Средняя схожесть</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_sim = max(c.similarity for c in chunks) if chunks else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{max_sim:.1%}</div>
            <div class="stat-label">Макс. схожесть</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_docs = len(set(c.document_name for c in chunks))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{unique_docs}</div>
            <div class="stat-label">Документов</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"### 📄 Результаты для: *\"{st.session_state.last_query}\"*")
    
    # Chunks display
    for i, chunk in enumerate(chunks, 1):
        # Determine similarity badge class
        if chunk.similarity >= 0.7:
            badge_class = ""
        elif chunk.similarity >= 0.4:
            badge_class = "medium"
        else:
            badge_class = "low"
        
        # Display content (use highlight if available)
        display_content = chunk.highlight if chunk.highlight and use_highlight else chunk.content
        
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-header">
                <span class="chunk-title">📄 {chunk.document_name}</span>
                <span class="similarity-badge {badge_class}">{chunk.similarity:.1%}</span>
            </div>
            <div class="chunk-content">{display_content}</div>
            <div class="chunk-meta">
                <span class="meta-item">🎯 Vector: {chunk.vector_similarity:.1%}</span>
                <span class="meta-item">📝 Term: {chunk.term_similarity:.1%}</span>
                <span class="meta-item">🔖 ID: {chunk.chunk_id[:8]}...</span>
            </div>
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
