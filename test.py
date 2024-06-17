from sqlalchemy import create_engine, Column, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import IntegrityError
import json
import ast
import torch
from pyvi.ViTokenizer import tokenize

Base = declarative_base()

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(String, primary_key=True, index=True)
    text = Column(String, index=True)
    vector = Column(LargeBinary)

class TextRequest(BaseModel):
    text: str

class Store(BaseModel):
    storeId: int
    embedding: str

class EmbeddingStore(BaseModel):
    stores: list[Store]
    input: str

# Khởi tạo ứng dụng FastAPI
app = FastAPI()
model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')


# Kết nối tới PostgreSQL
DATABASE_URL = "postgresql://postgres:123456@localhost/handmade_api"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Hàm tạo session cho mỗi request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hàm API để tạo hoặc cập nhật vector nhúng
@app.post("/embed/")
def create_embedding(request: TextRequest, db: Session = Depends(get_db)):
    try:
        # Mã hóa chuỗi thành vector
        encoded_text = request.text
        vector = model.encode(encoded_text, convert_to_tensor=True)

        # Convert the tensor to a NumPy array before storing
        vector_array = vector.cpu().numpy()

        vector_json = json.dumps(vector_array.tolist())

        return vector_json
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    
# Hàm API để so sánh
@app.post("/find-relational-stores/")
def findRelationStore(request: EmbeddingStore, db: Session = Depends(get_db)):
    try:
        str_embeddings = []
        storeIds = []
        for store in request.stores:
            embedding_str = store.embedding
            str_embeddings.append(embedding_str)

            storeIds.append(store.storeId)

        def parse_string_vector(string_vector):
            return ast.literal_eval(string_vector)

        parsed_vectors = [parse_string_vector(vec) for vec in str_embeddings]

        embeddings = torch.tensor(parsed_vectors)

        # Câu query
        query_sentence = tokenize(request.input)  # Thay thế bằng câu query của bạn
        query_embedding = model.encode(query_sentence, convert_to_tensor=True)  # Mã hóa câu query

        # Tính toán similarity score giữa câu query và các câu trong sentences
        similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings, dim=1)

        # Sắp xếp các câu theo similarity score
        sorted_stores = sorted(zip(storeIds, similarity_scores), key=lambda x: x[1], reverse=True)

        # Lấy ra top 5 cửa hàng
        top_5_stores = sorted_stores[:5]

        # In ra các câu và similarity score tương ứng
        ids = []
        for store_id, similarity_score in top_5_stores:
            ids.append(store_id)
        return ids
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))