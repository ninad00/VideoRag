import modal
import os
import whisper
import google.generativeai as genai
from groq import Groq
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
import subprocess
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoProcessor
import json
import boto3
from io import BytesIO
from PIL import Image
from pydantic import BaseModel

class ProcessRequest(BaseModel):
    video_id: str
    duration: int

class ChatRequest(BaseModel):
    video_id:str
    query: str
    top_k: int
    history: list[dict]

app = modal.App(name="videorag")
volume = modal.Volume.from_name("videorag-vol", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "openai-whisper==20250625",
        "numpy",
        "fastapi",
        "opencv-python-headless",
        "Pillow",
        "numpy",
        "boto3",
        "scikit-learn",
        "torch",
        "transformers",
        "sentencepiece",
        "python-dotenv",
        "google-generativeai",
        "groq",
        "sentence-transformers",
        "supabase"
    )
)

    

#################################
#HELPER FUNCTIONS
#################################

def upload_frame_to_s3(frame_np,bucket: str,video_id: str,frame_index: int,s3_client=None,quality: int = 70):
    img = Image.fromarray(frame_np).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    key = f"{video_id}/frames/{frame_index}.jpg"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer,
        ContentType="image/jpeg",
    )

    return key
def extract_keyframes(duration: int, video_id: str,siglip_model=None,siglip_processor=None, device=None, s3_client=None,):
    
    video_path=f"https://{os.environ['AWS_OUT_BUCKET']}.s3.{os.environ['AWS_REGION']}.amazonaws.com/{video_id}/master.m3u8"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if siglip_model is None or siglip_processor is None:
        model_name = "google/siglip-so400m-patch14-384"
        siglip_model = AutoModel.from_pretrained(model_name).to(device)
        siglip_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    clusters = 40 if duration <= 300 else 80 if duration <= 800 else 100
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    data = json.loads(subprocess.check_output(probe_cmd))
    w, h = data["streams"][0]["width"], data["streams"][0]["height"]

    # ---- FFmpeg frame extraction (NO timestamps) ----
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "scale=iw/2:ih/2",
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    frame_size = (w // 2) * (h // 2) * 3
    frames_list = []

    while True:
        raw = process.stdout.read(frame_size)
        if len(raw) != frame_size:
            break
        frame = np.frombuffer(raw, np.uint8).reshape((h // 2, w // 2, 3))
        frames_list.append(frame)

    process.stdout.close()
    process.wait()

    if not frames_list:
        return [], np.empty((0,))

    all_frames = np.array(frames_list)
    num_frames = len(all_frames)

    # ---- SIGLIP embeddings ----
    embeddings = []
    batch_size = 8

    for i in range(0, num_frames, batch_size):
        imgs = [Image.fromarray(f) for f in all_frames[i:i + batch_size]]
        inputs = siglip_processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = siglip_model.get_image_features(**inputs)
        output = siglip_model.get_image_features(**inputs)
        if hasattr(output, "pooler_output"):
            emb = output.pooler_output
        else:
            emb = output

        embeddings.append(emb.detach().cpu().numpy())


    embeddings = np.vstack(embeddings)

    # ---- KMeans selection ----
    k = min(clusters, num_frames)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    selected_indices = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            continue
        dists = np.linalg.norm(embeddings[idxs] - centroids[i], axis=1)
        selected_indices.append(idxs[np.argmin(dists)])

    selected_indices = sorted(set(selected_indices))

    keyframe_emb_list = []
    keyframes_meta = []

    for src_idx in selected_indices:
        frame = all_frames[src_idx]

        s3_key = upload_frame_to_s3(
            frame,
            os.environ["AWS_OUT_BUCKET"],
            video_id,
            frame_index=src_idx,  
            s3_client=s3_client, 
            quality=70
        )

        keyframe_emb_list.append(embeddings[src_idx])
        keyframes_meta.append({
            "frame_index": int(src_idx),   
            "s3_key": s3_key
        })

    keyframe_embeddings = np.array(keyframe_emb_list)

    return keyframes_meta, keyframe_embeddings
def chunk_full_transcript(text: str, max_chars=300):
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(" ".join(current)) >= max_chars:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks
def store_text_embeddings(full_transcript: str,video_id:str,sentence_model=None,supabase=None,):
    
    chunks = chunk_full_transcript(full_transcript)
    embeddings = [
        sentence_model.encode(chunk).tolist()
        for chunk in chunks
    ]

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "session_id": video_id,
            "chunk_index": i,
            "content": chunk,
            "embedding": emb,
        })

    supabase.table("text_embeddings").upsert(rows).execute()
def store_image_embeddings(keyframes_meta: list[dict],image_embeddings: list[list[float]],video_id:str,supabase=None):
    rows = []
    for meta, emb in zip(keyframes_meta, image_embeddings):
        rows.append({
            "session_id": video_id,
            "frame_index": meta["frame_index"],
            "embedding": emb.tolist(),
        })

    supabase.table("image_embeddings").upsert(rows).execute()
def retrieve_text_by_query(query: str,video_id: str, top_k: int = 5,sentence_model=None,supabase=None
):
    query_embedding = sentence_model.encode(query).tolist()
    response = supabase.rpc(
        "match_text_embeddings",
        {
            "query_embedding": query_embedding,
            "match_session_id": video_id,
            "match_count": top_k,
        },
    ).execute()

    return [row["content"] for row in response.data]

def retrieve_images_by_query(vlm_query: str,video_id: str, top_k: int = 8,siglip_model=None,siglip_processor=None, device=None,supabase=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = siglip_processor(
        text=[vlm_query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).to(device)

    with torch.no_grad():
        output = siglip_model.get_text_features(**inputs)
        if hasattr(output, "pooler_output"):
            emb = output.pooler_output[0]
        else:
            emb = output[0]

        query_embedding = emb.detach().cpu().numpy().tolist()


    response = supabase.rpc(
        "match_image_embeddings",
        {
            "query_embedding": query_embedding,
            "match_session_id": video_id,
            "match_count": top_k,
        },
    ).execute()
    return [row["frame_index"] for row in response.data]
def generate_text_answer(query: str,video_id: str, history: list[dict],groq_client=None, top_k: int = 5, sentence_model=None, supabase=None) -> str:

    chat_history = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in history[-2:]
    )

    retrieved_docs = retrieve_text_by_query(query,video_id=video_id, sentence_model= sentence_model, supabase=supabase, top_k=top_k)
    
    context = "\n\n".join(retrieved_docs)


    prompt = f"""
You are a helpful and precise assistant.

Answer the question in DETAIL using  the information provided in the Context.



If the user asks for a summary, provide a clear and detailed summary
based strictly on the Context.

---

Answer format:

Answer:
<concise, direct answer>

Sources:
<merge all relevant sentences from the Context into one coherent paragraph>

---

Conversation History:
{chat_history}

Context:
{context}

Question:
{query}

"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content
def make_query_for_vlm(user_query: str, rag_text: str, groq_client=None) -> str:
    prompt = f"""
You are an assistant that rewrites queries so they are optimized for retrieving the most relevant images from a vision-embedding model.

Inputs:
User Question: {user_query}
RAG Answer: {rag_text}

Output:
A rewritten visual-focused image-retrieval query in less than 64 tokens

"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return response.choices[0].message.content
def download_image_from_s3(object_key: str, s3_client=None) -> Image.Image:
    try:
        response = s3_client.get_object(
            Bucket=os.environ["AWS_OUT_BUCKET"],
            Key=object_key
        )
        return Image.open(BytesIO(response["Body"].read())).convert("RGB")
    except s3_client.exceptions.NoSuchKey:
        print(f"[WARN] Missing S3 key: {object_key}")
        return None

def transcribe( video_id: str, whisper_model=None) -> str:
    m3u8_url=f"https://{os.environ['AWS_OUT_BUCKET']}.s3.{os.environ['AWS_REGION']}.amazonaws.com/{video_id}/audio/index.m3u8"
    
    cmd = [
            "ffmpeg",
            "-i", m3u8_url,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
    ]
    process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
    )
    raw_audio = process.stdout.read()
    audio = (
            np.frombuffer(raw_audio, np.int16)
            .astype(np.float32) / 32768.0
        )
    text = whisper_model.transcribe(audio,fp16=True,verbose=False)['text']
    return text
def rag_with_gemini(video_id:str,query_actual: str,query_vlm: str,rag_text: str,top_k: int = 8,siglip_model=None,siglip_processor=None, device=None,gemini_model=None,s3_client=None,supabase=None):
    frame_indices = retrieve_images_by_query(query_vlm,video_id=video_id, supabase=supabase, top_k=top_k,siglip_model=siglip_model,siglip_processor=siglip_processor, device=device)
    images=[]
    for frame_index in frame_indices[:6]:
        # image_url = f"https://{os.environ['AWS_OUT_BUCKET']}.s3.{os.environ['AWS_REGION']}.amazonaws.com/{video_id}/frames/{int(frame_index)}.jpg"
        images.append(download_image_from_s3(f"{video_id}/frames/{int(frame_index)}.jpg", s3_client=s3_client))

    STRICT_CONTEXT = (
        "You are a knowledgeable and professional assistant specialized in answering questions about videos.\n"
        "Use the provided context (text and images) to answer the user's question.\n"
        "if the context has absolutely no information relevant to the question, respond with 'I don't know.But if something related is found,you can elaborate'\n"
        "Provide a detailed, clear, and well-structured explanation.\n"
        "Maintain a professional, instructional tone similar to that of an expert tutor."
        "### Output Format Rules (MANDATORY)\n\n"

    "Your response MUST follow this structure:\n\n"

    "## Answer\n"
    "- Provide a clear, concise, and direct answer to the user's question.\n"
    "- Maintain a professional, instructional tone.\n\n"

    "## Evidence from Context\n"
    "- List specific observations from the provided text and/or images.\n"
    "- Use bullet points.\n"
    )

    message_parts = [
        f"{STRICT_CONTEXT}\nQuestion: {query_actual}\nContext:\n{rag_text+query_vlm}"
    ]

    for item in images:
        message_parts.append(item)

    response = gemini_model.generate_content(message_parts)
    return response.text

##################################
#ENDPOINTS
##################################
@app.cls(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    concurrency_limit=1,
    secrets=[modal.Secret.from_name("videorag-secrets")]
)
class VideoProcessor:
    @modal.enter()
    def load_models(self):
        from supabase import create_client
        load_dotenv()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.siglip_model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384"
        ).to(self.device)

        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384",
            use_fast=True
        )

        self.whisper_model = None
        self.sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_REGION"],
        )
        self.s3 = session.client("s3")

        self.supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"]
        )

    @modal.method()
    def process(self, video_id: str, duration: int):
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base", device=self.device)

        metadata, embeddings = extract_keyframes(
            duration=duration,
            video_id=video_id,
            siglip_model=self.siglip_model,
            siglip_processor=self.siglip_processor,
            device=self.device,
            s3_client=self.s3
        )

        text = transcribe(video_id, whisper_model=self.whisper_model)

        store_text_embeddings(
            text,
            video_id=video_id,
            sentence_model=self.sentence_model,
            supabase=self.supabase
        )

        store_image_embeddings(
            metadata,
            video_id=video_id,
            image_embeddings=embeddings,
            supabase=self.supabase
        )

        return {"status": "processed", "video_id": video_id}

@app.cls(
    image=image,
    gpu="L4",
    timeout=60 * 10,
    concurrency_limit=2,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("videorag-secrets")]
)
class VideoChat:
    @modal.enter()
    def load_models(self):
        from supabase import create_client
        load_dotenv()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        self.siglip_model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384"
        ).to(self.device)

        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384",
            use_fast=True
        )

        self.sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.gemini_model = GenerativeModel("gemini-2.5-flash")
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_REGION"],
        )
        self.s3 = session.client("s3")

        self.supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"]
        )

    @modal.method()
    def chat(self, video_id: str, query: str, history: list[dict]|None = None, top_k: int = 8):
        if history is None:
            history = []
        text_ans = generate_text_answer(
            query=query,
            video_id=video_id,
            history=history,
            groq_client=self.groq_client,
            sentence_model=self.sentence_model,
            supabase=self.supabase,
            top_k=top_k
        )

        vlm_query = make_query_for_vlm(query, text_ans, self.groq_client)

        return rag_with_gemini(
            video_id=video_id,
            query_actual=query,
            query_vlm=vlm_query,
            rag_text=text_ans,
            top_k=top_k,
            siglip_model=self.siglip_model,
            siglip_processor=self.siglip_processor,
            device=self.device,
            gemini_model=self.gemini_model,
            s3_client=self.s3,
            supabase=self.supabase
        )

video_processor = VideoProcessor()
video_chat = VideoChat()

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("videorag-secrets")]
)
@modal.fastapi_endpoint(method="POST")
def process_endpoint(req: ProcessRequest):
    return video_processor.process.remote(
        video_id=req.video_id,
        duration=req.duration
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("videorag-secrets")]
)
@modal.fastapi_endpoint(method="POST")
def chat_endpoint(req: ChatRequest):
    return video_chat.chat.remote(
        video_id=req.video_id,
        query=req.query,
        history=req.history,
        top_k=req.top_k
    )
