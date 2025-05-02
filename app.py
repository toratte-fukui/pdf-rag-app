import os
import tempfile

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader

openai_api_key = st.secrets["openai"]["api_key"]

# Streamlit UI
st.title("💬 PDF検索チャットボット (gpt-4)")

uploaded_file = st.file_uploader(
    "① PDFファイルをアップロードしてください (例：[国税庁 確定申告のしくみ](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/01.htm))",
    type="pdf",
)
query = st.text_input("② 質問を入力してください (例：去年からの変更点を要約して)")

if uploaded_file and query:
    # 一時保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filepath = tmp_file.name

    # ① PDF読み込み
    with st.spinner("PDFを読み込んでいます..."):
        reader = PdfReader(tmp_filepath)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                splitter = RecursiveCharacterTextSplitter(
                    separators=[
                        "\n\n",
                        "\n",
                        " ",
                        ".",
                        ",",
                        "\u200b",  # Zero-width space
                        "\uff0c",  # Fullwidth comma
                        "\u3001",  # Ideographic comma
                        "\uff0e",  # Fullwidth full stop
                        "\u3002",  # Ideographic full stop
                        "",
                    ],
                    chunk_size=500,
                    chunk_overlap=250,
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"page_number": i + 1, "source": f"page_{i + 1}"},
                        )
                    )

        # ② ベクタDB作成（毎回新規）
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        if "vectordb" in st.session_state:
            del st.session_state["vectordb"]
        st.session_state["vectordb"] = FAISS.from_documents(docs, embeddings)

        # ③ 質問処理
        retriever: VectorStoreRetriever = st.session_state["vectordb"].as_retriever(
            search_kwargs={"k": 3}
        )
        relevant_docs = retriever.get_relevant_documents(query)

    if not relevant_docs:
        st.warning(
            "回答: このPDFには該当する情報が見つかりませんでした。わかりません。"
        )
    else:
        with st.spinner("考え中..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4"),
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
            )

            result = qa_chain({"query": query})

        answer = result["result"]
        source_docs = result.get("source_documents", [])

        if not answer.strip():
            st.warning(
                "回答: このPDFには該当する情報が見つかりませんでした。わかりません。"
            )
        else:
            # 回答表示
            st.success("回答:")
            st.write(answer)

        # 参照元と原文抜粋も表示（ページ単位で重複排除）
        st.info("参照元と引用:")

        # すでに表示したページを記録
        displayed_sources = set()

        for doc in source_docs:
            source = doc.metadata.get("source", "不明なページ")
            if source not in displayed_sources:
                st.write(f"📄 **{source}**")
                with st.expander("原文を見る"):
                    st.write(doc.page_content)
                displayed_sources.add(source)

    # 一時ファイル削除
    os.remove(tmp_filepath)
