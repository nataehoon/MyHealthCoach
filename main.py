from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, text
import os

# 랭체인 및 Ollama 관련 라이브러리
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="MyHealthCoach")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

CONNECTION_STRING = "postgresql+psycopg2://postgres:au3094ne@localhost:5432/myhealthcoach"
COLLECTION_NAME = "rehab_knowledge"

vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

class KnowledgeInput(BaseModel):
    content: str
    source: str

class ChatRequest(BaseModel):
    user_id: str
    question: str
    user_history: Optional[str] = None

@app.get("/")
def read_root():
    return {"status":"running", "message":"MyHealthCoach 서버가 준비되었습니다."}

UPLOAD_DIR = "./static/pdfs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # 파일 저장
        file_content = await file.read()
        file_size = len(file_content)
        file_path = os.path.join(UPLOAD_DIR, file.filename) 
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # 파일 정보 기록
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as conn:
            query = text("INSERT INTO uploaded_files(filename, file_path, file_size) VALUES(:filename, :path, :size)")
            conn.execute(query, {
                "filename": file.filename,
                "path": file_path,
                "size": file_size
            })
            conn.commit()

        print("DB 저장 성공")

        # PDF 로드 및 텍스트 추출
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print("텍스트 추출")

        # 텍스트 쪼개기(chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        print("텍스트 쪼개기")

        # 출처 정보를 메타데이터에 추가
        for split in splits:
            split.metadata["source"] = file.filename
        print("메타 데이터 저장")
        
        vector_store.add_documents(sanitize_docs(splits)) 
        print("벡터 DB 저장")
        
        return {"status": "success", "chunks_saved": len(splits), "message": f"'{file.filename}' 파일 분석 및 벡터 DB 저장 완료!"}
    except Exception as e:
        # 실패할 경우 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask")
async def ask_question(request: ChatRequest):
    docs = vector_store.similarity_search(request.question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    user_info = f"사용자 특이사항: {request.user_history}\n" if request.user_history else ""
    prompt = f"""
                당신은 숙현된 건강운동관리사 입니다.
                제공된 [참고 문헌]과 [사용자 정보]를 바탕으로 안전하고 전문적인 운동 조언을 제공하세요.
    
                {user_info}
                [참고 문헌]
                {context}
                
                [질문]
                {request.question}
                
                답변 시 주의사항: 
                - 근거가 되는 논문이나 서적이 있다면 언급하세요.
                - 확실하지 않은 의학적 조언은 피하고 전문가 상담을 권고하세요.
            """
    
    response = llm.invoke(prompt)
    
    return {
        "answer": response.content,
        "sources": [d.metadata["source"] for d in docs]
    }

def sanitize_docs(docs):
    for doc in docs:
        doc.page_content = doc.page_content.encode('utf-8', 'ignore').replace(b'\x00', b'').decode('utf-8')
        doc.metadata = {k: (v.encode('utf-8', 'ignore').replace(b'\x00', b'').decode('utf-8') if isinstance(v, str) else v) 
                        for k, v in doc.metadata.items()}
    return docs

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")