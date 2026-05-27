# 🎥 Distributed Video Processing & Streaming Pipeline

##System Architechture
<img width="1386" height="818" alt="image" src="https://github.com/user-attachments/assets/61955730-f627-42d3-bb17-6f38028d8bc0" />


## Deployed site:https://video-streaming-ruby.vercel.app/


A highly scalable, asynchronous video processing and streaming platform designed to handle large file uploads, transcode videos into adaptive bitrates (HLS/DASH), and generate AI-powered subtitles without blocking the main application server. 

This project demonstrates production-grade system design principles including decoupled message queues, client-to-storage direct uploads, and scalable worker nodes.

## 🚀 Key Features

* **Direct-to-S3 Multipart Uploads:** Bypasses the Node.js backend entirely using pre-signed URLs, preventing server memory exhaustion during massive 1GB+ video uploads.
* **Asynchronous Transcoding Pipeline:** Uses AWS SQS to decouple the upload lifecycle from heavy CPU-bound FFmpeg video chunking processes.
* **Adaptive Bitrate Streaming:** Transcodes original video files into multiple resolutions (240p, 720p, 1080p, 4K) using HLS protocols for optimal playback on any network condition.
* **AI-Powered Subtitles:** Integrates OpenAI Whisper on a secondary backend to automatically generate and serve subtitle tracks.
* **Secure Authentication:** Implements Google OAuth2 for secure user sessions and access management.
* **Optimized Relational State Tracking:** Utilizes MySQL for highly concurrent state management (Pending ➔ Uploaded ➔ Chunking ➔ Complete), storing lightweight job IDs and metadata rather than bloated document objects to maintain fast read/write speeds.

## 🏗️ System Architecture

### 1. Upload & Registration Phase
1. The **Client** authenticates via Google OAuth2.
2. The user initiates an upload. The **Express Backend** creates a `PENDING` record in the **MySQL database**.
3. The backend generates and returns an array of **AWS S3 Pre-signed URLs** (one for each chunk).
4. The client uploads chunks concurrently, directly to S3.

### 2. Event-Driven Processing Phase
1. Upon upload completion, S3 stitches the file together. A notification is sent to an **AWS SQS Queue**.
2. **Dockerized FFmpeg Workers** continuously poll the SQS queue.
3. A worker picks up the job, downloads the raw video, and begins chunking it into 2-10 second clips across multiple resolutions.
4. Transcoded chunks and the `.m3u8` manifest are uploaded back to a public-facing S3 bucket.

### 3. ML Subtitles & Delivery Phase
1. The **Whisper AI** backend extracts the audio track and generates transcripts.
2. The MySQL Database is updated to flag the video status as `READY`.
3. The client fetches the metadata and streams the video using **Video.js**, seamlessly adapting resolution based on bandwidth.

## 🛠️ Tech Stack

**Frontend**
* Framework: React (Deployed on Vercel)
* Player: Video.js (HLS/DASH support)
* Authentication: Google OAuth2

**Backend (API & State Management)**
* Server: Node.js with Express (Deployed on Railway)
* Database: MySQL (Deployed on Railway)
* Auth: Google OAuth2

**Processing Workers (Transcoding & ML)**
* Containerization: Docker
* Video Processing: FFmpeg (HLS chunking)
* AI Subtitles: OpenAI Whisper

**Infrastructure & Cloud**
* Object Storage: AWS S3
* Message Broker: AWS SQS
* Transcoding: FFMPEG


## 💡 Engineering Highlights

* **Resilience against Poison Pills:** Configured SQS with proper Visibility Timeouts (to account for long transcoding times) and Dead Letter Queues (DLQ) to isolate corrupted video files without crashing the worker fleet.
* **S3 Cost Optimization:** Implemented S3 Lifecycle rules (`AbortIncompleteMultipartUpload`) to automatically clean up orphaned chunks from abandoned client uploads.
* **Stateless Workers:** FFmpeg and Whisper Docker containers are entirely stateless, allowing them to be horizontally scaled simply by tracking the SQS queue depth.

## 💻 Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/video-streaming-pipeline.git](https://github.com/yourusername/video-streaming-pipeline.git)
   cd video-streaming-pipeline
