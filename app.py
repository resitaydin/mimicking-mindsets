import streamlit as st
from rag_system import ErolGungorRAG

def initialize_session_state():
    if "rag_system" not in st.session_state:
        with st.spinner("Sistem yükleniyor..."):
            st.session_state.rag_system = ErolGungorRAG()
            st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="Erol Güngör AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Sidebar with information
    with st.sidebar:
        st.image("erol_gungor.jpg", caption="Prof. Dr. Erol Güngör (1938-1983)")
        st.markdown("""
        ### Hakkında
        Prof. Dr. Erol Güngör, Türk düşünce hayatının önemli isimlerinden biridir.
        Sosyal psikoloji, kültür ve medeniyet konularında önemli eserler vermiştir.
        
        Çalışma alanları:
        - Kültür ve Medeniyet
        - Sosyal Psikoloji
        - İslam Düşüncesi
        - Türk Modernleşmesi
        """)
        
        # Add system info
        st.markdown("---")
        st.markdown("### Sistem Bilgisi")
        st.markdown("- Model: Llama-3.3 70B")
        st.markdown("- Powered by Groq")
    
    st.title("🎓 Erol Güngör AI")
    st.markdown("""
    Prof. Dr. Erol Güngör'ün eserlerinden derlenen bilgilerle geliştirilmiş yapay zeka sistemi.
    """)
    
    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("Detaylar"):
                    st.markdown(f"**Güven Skoru:** {message['metadata']['confidence_score']:.0%}")
                    if message["metadata"]["sources"]:
                        st.markdown("**Kaynaklar:**")
                        for source in message["metadata"]["sources"]:
                            st.markdown(f"📚 **{source['file']}**\n> {source['text']}")
    
    # User input
    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                response = st.session_state.rag_system.get_response(prompt)
                
                st.markdown(response["response"])
                
                # Add to chat history with metadata
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["response"],
                    "metadata": {
                        "confidence_score": response["confidence_score"],
                        "sources": response["sources"]
                    }
                })
                
                with st.expander("Detaylar"):
                    st.markdown(f"**Güven Skoru:** {response['confidence_score']:.0%}")
                    st.progress(response['confidence_score'])
                    
                    if response["sources"]:
                        st.markdown("**Kaynaklar:**")
                        for source in response["sources"]:
                            st.markdown(f"📚 **{source['file']}**\n> {source['text']}")

if __name__ == "__main__":
    main()