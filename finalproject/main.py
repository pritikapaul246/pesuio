import streamlit as st    #framework
import os
from utils.assistant import CustomAIAssistant  

def main():
    
    st.title("CareerCompass")

    if "assistant" not in st.session_state:
        st.session_state.assistant = None
        st.session_state.messages = []

    if st.session_state.assistant is None:
        data_path = "./data"  
        index_path = "./index" 
        try:
            
            st.session_state.assistant = CustomAIAssistant(
                data_path=data_path,
                index_path=index_path
            )
            status_msg = "Loaded existing index." if os.path.exists(index_path) else "Created new index."
            st.success(status_msg)
        except Exception as e:
            st.error(f"Error initializing assistant: {str(e)}")
            return

   #history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your job-related and resume-building questions here"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        #query
        with st.chat_message("assistant"):
            try:
                
                result = st.session_state.assistant.query(prompt)
                
                if isinstance(result, dict):
                    answer = result.get("answer", "No answer found")

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
           
                    st.error("Unexpected response format")
            except Exception as e:
               
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
if __name__ == "__main__":
    main()

