# VideoRAG - Intelligent Video Analysis & Chat Platform

A comprehensive video processing and intelligent chatbot system that goes beyond traditional subtitle-based video analysis. VideoRAG combines video transcoding with advanced RAG (Retrieval-Augmented Generation) to enable intelligent conversations about video content using both visual and textual information.

## 🎯 Project Overview

Traditional video chatbots for platforms like YouTube only analyze subtitles, missing crucial visual context. VideoRAG solves this by:
- Processing video frames to extract visual information
- Transcribing audio with high accuracy
- Creating searchable embeddings for both text and images
- Enabling multimodal conversations using 3 specialized LLMs

## 🏗️ Architecture

VideoRAG consists of two main components:

### Part 1: Video Transcoding & Streaming System
- **Docker Worker** for HLS video transcoding
- **AWS S3** storage for processed videos
- **SQS polling** for job management
- **MySQL database** via Prisma ORM
- **Video.js player** for adaptive streaming

### Part 2: VideoRAG Pipeline (FastAPI on Modal)
A sophisticated AI pipeline deployed on Modal that processes videos through:

1. **FFmpeg** - Video preprocessing and frame extraction
2. **K-means Clustering** - Intelligent keyframe selection
3. **Whisper** - Audio transcription
4. **SIGLIP Embeddings** - Visual and text embeddings
5. **Supabase Vector Store** - Embedding storage with similarity search
6. **3 LLM Chain**:
   - **LLM 1** (Llama 3.1): Text-based RAG for context retrieval
   - **LLM 2** (Llama 3.1): Visual query generation
   - **LLM 3** (Gemini 2.5 Flash): Multimodal reasoning with images

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Part 1: Video Processing                 │
└─────────────────────────────────────────────────────────────────┘

User Upload → S3 Raw Bucket → SQS Queue → Docker Worker
                                              ↓
                                         FFmpeg Transcode
                                              ↓
                                    HLS (1080p/720p/480p)
                                              ↓
                                    S3 Output Bucket ← MySQL DB
                                              ↓
                                         Video.js Player

┌─────────────────────────────────────────────────────────────────┐
│                    Part 2: VideoRAG Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

HLS Video → Modal Process Endpoint
                ↓
          FFmpeg Extract Frames
                ↓
          SIGLIP Embeddings
                ↓
          K-means Clustering → Selected Keyframes → S3
                ↓
          Whisper Transcription
                ↓
          Text Chunking & Embeddings
                ↓
          Supabase Vector Store
                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Chat Interface                           │
└─────────────────────────────────────────────────────────────────┘

User Query → LLM 1 (Text RAG) → Context Retrieval
                ↓
          LLM 2 (Visual Query) → Generate Image Search Query
                ↓
          Retrieve Top-K Images & Text
                ↓
          LLM 3 (Multimodal) → Final Answer with Evidence
```

## 🛠️ Technology Stack

### Backend (Part 1 - Video Streaming)
- **Node.js** with Express
- **Prisma ORM** with MySQL
- **AWS SDK** (S3, SQS)
- **FFmpeg** for video processing
- **Docker** containerization

### VideoRAG Pipeline (Part 2)
- **Modal** for serverless GPU deployment
- **Python 3.12**
- **Whisper** (OpenAI) - Audio transcription
- **SIGLIP** (Google) - Vision-language embeddings
- **Sentence Transformers** - Text embeddings
- **scikit-learn** - K-means clustering
- **Supabase** - Vector database
- **LLMs**:
  - Llama 3.1 8B (via Groq) - Text RAG & visual query
  - Gemini 2.5 Flash - Multimodal reasoning
- **FastAPI** for HTTP endpoints

### Frontend
- **React 19** with Vite
- **React Router** for navigation
- **Video.js** for HLS playback
- **Tailwind CSS** for styling
- **Axios** for API calls

## 📋 Prerequisites

- Node.js 20+
- Python 3.12
- Docker & Docker Compose
- AWS Account (S3, SQS)
- Modal Account
- Supabase Account
- API Keys:
  - Google Gemini API
  - Groq API

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/ninad00/VideoRag.git
cd VideoRag
```

### 2. Database Setup
```bash
cd backend/docker_db
docker-compose up -d
```

### 3. Backend Setup

#### Environment Variables
Create `backend/.env`:
```env
# Database
DATABASE_URL="mysql://root:ninad2005#@localhost:3306/videosite"

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
AWS_RAW_BUCKET=your_raw_bucket_name
AWS_OUT_BUCKET=your_output_bucket_name

# Modal
MODAL_PROCESS_URL=https://your-modal-deployment--videorag-process-endpoint.modal.run
```

#### Install & Run
```bash
cd backend
npm install
npm run prisma:generate
npm run prisma:migrate
npm run dev
```

### 4. Docker Worker Setup

Create `backend/docker/.env`:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
AWS_RAW_BUCKET=your_raw_bucket_name
AWS_OUT_BUCKET=your_output_bucket_name
MODAL_PROCESS_URL=https://your-modal-deployment--videorag-process-endpoint.modal.run
```

Build and run:
```bash
cd backend/docker
docker build -t video-worker .
docker run --env-file .env video-worker
```

### 5. VideoRAG Pipeline (Modal)

#### Setup Modal
```bash
pip install modal
modal setup
```

#### Create Modal Secrets
```bash
modal secret create videorag-secrets \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  AWS_REGION=your_region \
  AWS_OUT_BUCKET=your_bucket \
  SUPABASE_URL=your_supabase_url \
  SUPABASE_SERVICE_KEY=your_supabase_key \
  GOOGLE_API_KEY=your_gemini_key \
  GROQ_API_KEY=your_groq_key
```

#### Deploy to Modal
```bash
cd backend/videorag
modal deploy mod.py
```

### 6. Supabase Setup

Create two tables with vector extensions:

```sql
-- Enable vector extension
create extension if not exists vector;

-- Text embeddings table
create table text_embeddings (
  id bigserial primary key,
  session_id text not null,
  chunk_index int not null,
  content text not null,
  embedding vector(384)
);

-- Image embeddings table
create table image_embeddings (
  id bigserial primary key,
  session_id text not null,
  frame_index int not null,
  embedding vector(1152)
);

-- Create RPC functions for similarity search
create or replace function match_text_embeddings(
  query_embedding vector(384),
  match_session_id text,
  match_count int
)
returns table (
  content text,
  similarity float
)
language sql
as $$
  select content, 1 - (embedding <=> query_embedding) as similarity
  from text_embeddings
  where session_id = match_session_id
  order by embedding <=> query_embedding
  limit match_count;
$$;

create or replace function match_image_embeddings(
  query_embedding vector(1152),
  match_session_id text,
  match_count int
)
returns table (
  frame_index int,
  similarity float
)
language sql
as $$
  select frame_index, 1 - (embedding <=> query_embedding) as similarity
  from image_embeddings
  where session_id = match_session_id
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

### 7. Frontend Setup

Create `frontend/.env`:
```env
VITE_API_URL=http://localhost:3000
```

Install & run:
```bash
cd frontend
npm install
npm run dev
```

## 🎮 Usage

### 1. Upload a Video
- Navigate to `/upload`
- Select an MP4 video file
- Upload begins → sends to S3 raw bucket
- SQS triggers Docker worker
- Worker transcodes to HLS (480p, 720p, 1080p)
- Uploads to S3 output bucket
- Triggers Modal VideoRAG processing

### 2. VideoRAG Processing (Automatic)
When transcoding completes:
- Extracts frames from HLS video
- Generates SIGLIP embeddings for all frames
- Uses K-means to select keyframes (40-100 depending on duration)
- Stores keyframes in S3
- Transcribes audio using Whisper
- Chunks text and creates embeddings
- Stores all embeddings in Supabase

### 3. Watch & Chat
- Browse videos on homepage
- Click to watch with HLS adaptive streaming
- Open AI chat panel
- Ask questions about the video:
  - "What is shown in this video?"
  - "Explain the key concepts"
  - "What happens around the 5-minute mark?"

### 4. How Chat Works
1. **User sends query**
2. **LLM 1 (Text RAG)**: Retrieves relevant transcript segments from Supabase
3. **LLM 2 (Visual Query)**: Converts user query + text context into visual search query
4. **Retrieve top-K frames**: Uses SIGLIP embeddings to find relevant frames
5. **LLM 3 (Multimodal)**: Analyzes frames + text + query to generate comprehensive answer

## 🎨 Features

### Video Processing
- ✅ Adaptive HLS streaming (480p, 720p, 1080p)
- ✅ Audio-only track support
- ✅ Automatic thumbnail generation
- ✅ Subtitle extraction (if available)
- ✅ Quality selection in player

### VideoRAG Intelligence
- ✅ Visual understanding via keyframe analysis
- ✅ Accurate transcription with Whisper
- ✅ Efficient similarity search with vector embeddings
- ✅ Multi-LLM reasoning chain
- ✅ Context-aware responses
- ✅ Evidence-based answers with sources

### User Interface
- ✅ Netflix-inspired design
- ✅ Real-time chat interface
- ✅ Responsive layout
- ✅ Quality selector
- ✅ Video gallery

## 🔑 Key Differentiators

Unlike traditional YouTube chatbots that only use subtitles:

1. **Visual Understanding**: Analyzes actual video frames to understand visual content
2. **Smart Frame Selection**: K-means clustering selects the most representative frames
3. **Multimodal Reasoning**: Combines text and visual information for comprehensive answers
4. **Three-LLM Pipeline**: Specialized models for different tasks (text retrieval, visual query, multimodal reasoning)
5. **Efficient Storage**: Uses vector embeddings for fast similarity search in Supabase

## 📊 Model Details

| Component | Model | Purpose | Provider |
|-----------|-------|---------|----------|
| Audio Transcription | Whisper Base | Convert audio to text | OpenAI |
| Visual Embeddings | SIGLIP-SO400M-384 | Encode frames & text queries | Google |
| Text Embeddings | all-MiniLM-L6-v2 | Encode text chunks | Sentence Transformers |
| Text RAG | Llama 3.1 8B Instant | Retrieve relevant context | Groq |
| Visual Query | Llama 3.1 8B Instant | Generate image search query | Groq |
| Multimodal Answer | Gemini 2.5 Flash | Final reasoning with images | Google |

## 🔐 Security Notes

- Store all API keys in environment variables
- Use Supabase service keys (not anon keys) for backend
- Configure S3 bucket policies appropriately
- Use presigned URLs for uploads
- Keep Modal secrets secure

## 📁 Project Structure

```
VideoRag/
├── backend/
│   ├── controllers/        # Route handlers
│   ├── docker/            # Docker worker for transcoding
│   │   ├── dockerfile
│   │   ├── worker.js      # SQS polling worker
│   │   └── transcode.js   # FFmpeg HLS transcoding
│   ├── docker_db/         # MySQL Docker setup
│   ├── functions/         # Helper functions
│   ├── prisma/            # Database schema
│   ├── routes/            # API routes
│   ├── videorag/          # Modal deployment
│   │   └── mod.py         # VideoRAG pipeline
│   ├── server.js          # Express server
│   └── package.json
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Router setup
│   │   ├── Homepage.jsx   # Video gallery
│   │   ├── Video.jsx      # Video player + chat
│   │   └── upload.jsx     # Upload interface
│   └── package.json
└── README.md
```

## 🚧 Development

### Running Locally
```bash
# Terminal 1: MySQL
cd backend/docker_db && docker-compose up

# Terminal 2: Backend
cd backend && npm run dev

# Terminal 3: Docker Worker
cd backend/docker && docker run --env-file .env video-worker

# Terminal 4: Frontend
cd frontend && npm run dev
```

### Modal Development
```bash
# Test locally
modal serve backend/videorag/mod.py

# Deploy
modal deploy backend/videorag/mod.py
```

## 🐛 Troubleshooting

**Video not processing?**
- Check SQS queue has messages
- Verify Docker worker is running
- Check AWS credentials

**Chat not responding?**
- Verify Modal endpoints are deployed
- Check API keys in Modal secrets
- Ensure Supabase functions are created

**HLS not playing?**
- Check S3 bucket CORS settings
- Verify S3 object URLs are accessible
- Check browser console for errors

## 🎓 Use Cases

- **Educational Content**: Analyze lectures and tutorials with visual context
- **Documentation**: Create interactive video documentation
- **Training Materials**: Build intelligent assistants for training videos
- **Research**: Analyze video data with AI assistance

## 📝 License

This project is available for educational and research purposes.

## 👥 Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- Test your changes locally
- Update documentation as needed

## 🙏 Acknowledgments

- Modal for serverless GPU infrastructure
- OpenAI Whisper for transcription
- Google SIGLIP for vision-language embeddings
- Groq for fast LLM inference
- Supabase for vector database

---

**Built with ❤️ for better video understanding**
