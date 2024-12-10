import streamlit as st
from rag_system import ErolGungorRAG

def initialize_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = ErolGungorRAG()

def main():
    st.set_page_config(page_title="Erol Güngör AI", layout="wide")
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
    
    st.title("🎓 Erol Güngör AI")
    st.markdown("""
    Prof. Dr. Erol Güngör'ün eserlerinden derlenen bilgilerle geliştirilmiş olan yapay zeka sistemi.
    """)
    
    # Input area
    user_input = st.text_input("Sorunuzu yazın:")
    
    if user_input:
        # Get response
        response = st.session_state.rag_system.get_response(user_input)
        
        # Display response with confidence score
        st.markdown("### Yanıt:")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Prof. Dr. Erol Güngör:** {response['response']}")
        
        with col2:
            confidence_percentage = int(response['confidence_score'] * 100)
            st.metric(
                label="Güven Skoru",
                value=f"%{confidence_percentage}",
                delta=None,
                help="Sistemin yanıtın doğruluğuna olan güven seviyesi"
            )
            
            # Visual confidence indicator
            st.progress(response['confidence_score'])
        
        # Display sources in a cleaner format
        with st.expander("Kaynaklar"):
            for source in response["sources"]:
                st.markdown(f"""
                📚 **{source['file']}**  
                > {source['text']}
                ---
                """)

if __name__ == "__main__":
    main() 