from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 랭체인 및 Ollama 관련 라이브러리
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

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

@app.post("/add-knowledge")
async def add_knowledge(data: KnowledgeInput):
    try:
        doc = Document(page_content=data.content, metadata={"source":data.source})
        vector_store.add_documents([doc])
        return {"message": f"'{data.source}' 내용이 성공적으로 저장되었습니다."}
    except Exception as e:
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