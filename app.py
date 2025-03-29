import streamlit as st
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Funções Auxiliares ---
def create_collection(client, collection_name):
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # Ajuste o tamanho conforme o modelo
        )
        st.success(f"Coleção '{collection_name}' criada com sucesso!")
    except Exception as e:
        st.error(f"Erro ao criar a coleção: {e}")

def delete_collection(client, collection_name):
    try:
        client.delete_collection(collection_name=collection_name)
        st.success(f"Coleção '{collection_name}' excluída com sucesso!")
    except Exception as e:
        st.error(f"Erro ao excluir a coleção: {e}")

def upload_pdf_and_index(client, collection_name, pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        # TODO: Integrar um modelo de embedding aqui (ex: OpenAI, Sentence Transformers)
        # Exemplo com dados dummy:
        vectors = [[0.1] * 1536 for _ in range(len(chunks))]  # Substitua por embeddings reais

        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=i,  # Gere IDs únicos
                    vector=vectors[i],
                    payload={"content": chunk},
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)

        st.success(f"PDF '{pdf_file.name}' indexado na coleção '{collection_name}'!")
    except Exception as e:
        st.error(f"Erro ao processar e indexar o PDF: {e}")
        st.exception(e) # Mostra o stack trace completo

def test_qdrant_connection(url, port):
    try:
        client = QdrantClient(url=url, port=port)
        client.get_collection(collection_name="__test_connection__")  # Tenta acessar uma coleção inexistente para testar a conexão
        return True, None  # Conexão bem-sucedida
    except Exception as e:
        return False, str(e) # Conexão falhou

# --- Interface Streamlit ---
def main():
    st.title("Gerenciador de Banco de Vetores Qdrant")

    # Sidebar para configurações do Qdrant
    st.sidebar.header("Configurações do Qdrant")
    qdrant_url = st.sidebar.text_input("URL do Qdrant", value="http://localhost")
    qdrant_port = st.sidebar.number_input("Porta do Qdrant", value=6333, step=1)

    if st.sidebar.button("Testar Conexão"):
        success, error = test_qdrant_connection(qdrant_url, qdrant_port)
        if success:
            st.sidebar.success("Conexão com o Qdrant bem-sucedida!")
        else:
            st.sidebar.error(f"Erro ao conectar ao Qdrant: {error}")
            # Não retorna aqui, deixa o usuário tentar novamente

    try:
        client = QdrantClient(url=qdrant_url, port=qdrant_port)
        # Conexão já foi testada, não precisa repetir aqui (a menos que queira reconectar periodicamente)
        # st.sidebar.success("Conectado ao Qdrant!") # Remova ou condicione se quiser reconectar automaticamente
    except Exception as e:
        st.sidebar.error(f"Erro ao conectar ao Qdrant: {e}")
        return  # Impede que o resto do app execute se não conseguir conectar

    # Abas
    tab1, tab2 = st.tabs(["Gerenciamento de Coleções", "Indexação de PDFs"])

    with tab1:
        st.header("Gerenciamento de Coleções")
        collection_name = st.text_input("Nome da Coleção")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Criar Coleção"):
                create_collection(client, collection_name)
        with col2:
            if st.button("Excluir Coleção"):
                delete_collection(client, collection_name)

    with tab2:
        st.header("Indexação de PDFs")
        collection_name_pdf = st.text_input("Nome da Coleção para Indexar o PDF")
        pdf_file = st.file_uploader("Carregar PDF", type="pdf")

        if pdf_file is not None:
            if st.button("Indexar PDF"):
                upload_pdf_and_index(client, collection_name_pdf, pdf_file)


if __name__ == "__main__":
    main()
