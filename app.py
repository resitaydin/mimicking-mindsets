import streamlit as st
from core import RAGEngine
import time
import plotly.graph_objects as go
from datetime import datetime

def initialize_session_state():
    """Initialize or reset session state variables"""
    if "rag_system" not in st.session_state:
        with st.spinner("Sistem yükleniyor..."):
            try:
                st.session_state.rag_system = RAGEngine()
                st.session_state.chat_history = []
                st.session_state.total_chunks = st.session_state.rag_system.collection.count()
                st.session_state.confidence_history = []
            except Exception as e:
                st.error(f"Sistem başlatılırken bir hata oluştu: {str(e)}")
                st.stop()

def format_source_citation(source):
    """Enhanced source citation formatting"""
    title = source.get('title', 'Bilinmiyor')
    file = source.get('file', 'Bilinmiyor')
    similarity = source.get('similarity', 0)
    
    return f"""
    📚 **{title}**
    📄 Kaynak: {file}
    🎯 Benzerlik: {similarity:.2%}
    
    > {source['text']}
    """

def create_confidence_gauge(score):
    """Create a Plotly gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40
            }
        }
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=20, b=10),
        font={'size': 16}
    )
    return fig

def display_confidence_metrics(score, sources):
    """Display simplified confidence metrics with progress bar"""
    score_percentage = f"{score:.0%}"
    
    # Determine color based on score
    bar_color = "red" if score < 0.4 else "orange" if score < 0.7 else "green"
    
    st.markdown(f"""
        <div style='margin-bottom: 1rem;'>
            <p style='margin-bottom: 0.5rem; font-weight: bold;'>Güven Skoru: {score_percentage}</p>
            <div style='width: 100%; background-color: #f0f0f0; border-radius: 10px;'>
                <div style='width: {score_percentage}; height: 20px; background-color: {bar_color}; 
                     border-radius: 10px; transition: width 0.5s ease-in-out;'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Erol Güngör AI",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Erol Güngör'ün eserlerinden derlenen yapay zeka sistemi"
        }
    )
    
    initialize_session_state()
    
    # Enhanced sidebar
    with st.sidebar:
        st.image("erol_gungor.jpg", caption="Prof. Dr. Erol Güngör (1938-1983)")
        
        st.markdown("""
        ### 🎓 Prof. Dr. Erol Güngör
        Türk düşünce hayatının önemli isimlerinden biri olan Prof. Dr. Erol Güngör,
        sosyal psikoloji, kültür ve medeniyet alanlarında önemli eserler vermiştir.
        """)
        
        tabs = st.tabs(["📚 Çalışmalar", "⚙️ Sistem"])
        
        with tabs[0]:
            st.markdown("""
            - 🏛️ Kültür ve Medeniyet
            - 🧠 Sosyal Psikoloji
            - 🕌 İslam Düşüncesi
            - 📜 Türk Modernleşmesi
            """)
        
        with tabs[1]:
            st.markdown(f"""
            - 🤖 Model: Llama-3.3 70B
            - ⚡ Powered by Groq
            - 📊 Toplam Metin: {st.session_state.total_chunks:,} parça
            - 🔄 Son Güncelleme: {datetime.now().strftime('%d.%m.%Y')}
            """)

        if st.button("💬 Sohbeti Temizle"):
            st.session_state.chat_history = []
            st.session_state.confidence_history = []
            st.rerun()
    
    st.title("🎓 Erol Güngör AI")
    st.markdown("""
    Prof. Dr. Erol Güngör'ün eserlerinden derlenen bilgilerle geliştirilmiş yapay zeka sistemi.
    Sorularınızı Türkçe olarak sorabilirsiniz.
    """)
    
    # Enhanced chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "metadata" in message:
                display_confidence_metrics(
                    message['metadata']['confidence_score'],
                    message['metadata']['sources']
                )
                
                with st.expander("📚 Kaynakları Göster"):
                    if message["metadata"]["sources"]:
                        for source in message["metadata"]["sources"]:
                            st.markdown(format_source_citation(source))
                    
                    if message["metadata"].get("query_time"):
                        st.markdown(f"⚡ Yanıt Süresi: {message['metadata']['query_time']:.2f} saniye")
    
    # Enhanced user input handling
    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                try:
                    start_time = time.time()
                    response = st.session_state.rag_system.get_response(prompt)
                    query_time = time.time() - start_time
                    
                    st.markdown(response["response"])
                    
                    # Update confidence history
                    st.session_state.confidence_history.append(response["confidence_score"])
                    
                    # Add enhanced metadata to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["response"],
                        "metadata": {
                            "confidence_score": response["confidence_score"],
                            "sources": response["sources"],
                            "query_time": query_time
                        }
                    })
                    
                    display_confidence_metrics(response['confidence_score'], response['sources'])
                    
                    with st.expander("📚 Kaynakları Göster"):
                        if response["sources"]:
                            for source in response["sources"]:
                                st.markdown(format_source_citation(source))
                        
                        st.markdown(f"⚡ Yanıt Süresi: {query_time:.2f} saniye")
                        
                except Exception as e:
                    st.error(f"Yanıt oluşturulurken bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()