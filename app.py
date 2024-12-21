import streamlit as st
from rag_system import ErolGungorRAG
import time

def initialize_session_state():
    """Initialize or reset session state variables"""
    if "rag_system" not in st.session_state:
        with st.spinner("Sistem yükleniyor..."):
            try:
                st.session_state.rag_system = ErolGungorRAG()
                st.session_state.chat_history = []
                st.session_state.total_chunks = st.session_state.rag_system.collection.count()
            except Exception as e:
                st.error(f"Sistem başlatılırken bir hata oluştu: {str(e)}")
                st.stop()

def format_source_citation(source):
    """Format source citation in a more readable way"""
    title = source.get('title', 'Bilinmiyor')
    author = source.get('author', 'Bilinmiyor')
    similarity = source.get('similarity', 0)
    
    return f"""
    📚 **{title}**
    Benzerlik: {similarity:.2%}
    > {source['text']}
    """

def display_confidence_indicator(score):
    """Display a simplified confidence indicator"""
    color = "red" if score < 0.4 else "yellow" if score < 0.7 else "green"
    st.progress(score, text=f"Güven Skoru: {score:.0%}")
    
    if score < 0.4:
        st.warning("⚠️ Bu yanıt düşük güvenilirliğe sahip olabilir.")
    elif score > 0.8:
        st.success("✅ Bu yanıt yüksek güvenilirliğe sahip.")

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
    
    # Sidebar with enhanced information
    with st.sidebar:
        st.image("erol_gungor.jpg", caption="Prof. Dr. Erol Güngör (1938-1983)")
        
        st.markdown("""
        ### Hakkında
        Prof. Dr. Erol Güngör, Türk düşünce hayatının önemli isimlerinden biridir.
        Sosyal psikoloji, kültür ve medeniyet konularında önemli eserler vermiştir.
        """)
        
        with st.expander("Çalışma Alanları", expanded=False):
            st.markdown("""
            - 📚 Kültür ve Medeniyet
            - 🧠 Sosyal Psikoloji
            - 🕌 İslam Düşüncesi
            - 🏛️ Türk Modernleşmesi
            """)
        
        # Add enhanced system info
        st.markdown("---")
        st.markdown("### Sistem Bilgisi")
        st.markdown(f"""
        - 🤖 Model: Llama-3.3 70B
        - ⚡ Powered by Groq
        - 📊 Toplam Metin Parçası: {st.session_state.total_chunks:,}
        """)
        
        # Add clear chat button
        if st.button("💬 Sohbeti Temizle"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.title("🎓 Erol Güngör AI")
    st.markdown("""
    Prof. Dr. Erol Güngör'ün eserlerinden derlenen bilgilerle geliştirilmiş yapay zeka sistemi.
    Sorularınızı Türkçe olarak sorabilirsiniz.
    """)
    
    # Chat interface with enhanced visualization
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "metadata" in message:
                display_confidence_indicator(message['metadata']['confidence_score'])
                    
                with st.expander("Kaynakları Göster"):
                    if message["metadata"]["sources"]:
                        st.markdown("**Kullanılan Kaynaklar:**")
                        for source in message["metadata"]["sources"]:
                            st.markdown(format_source_citation(source))
    
    # User input with enhanced error handling
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
                    
                    # Add to chat history with enhanced metadata
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["response"],
                        "metadata": {
                            "confidence_score": response["confidence_score"],
                            "sources": response["sources"],
                            "query_time": query_time
                        }
                    })
                    
                    display_confidence_indicator(response['confidence_score'])
                        
                    with st.expander("Kaynakları Göster"):
                        if response.get("metadata"):
                            st.markdown(f"⚡ Yanıt Süresi: {query_time:.2f} saniye")
                        
                        if response["sources"]:
                            st.markdown("**Kullanılan Kaynaklar:**")
                            for source in response["sources"]:
                                st.markdown(format_source_citation(source))
                        
                except Exception as e:
                    st.error(f"Yanıt oluşturulurken bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()