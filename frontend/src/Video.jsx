import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import videojs from "video.js";
import "video.js/dist/video-js.css";
import "videojs-contrib-quality-levels";

export default function PlayerPage() {
  const { id } = useParams();
  const videoRef = useRef(null);
  const playerRef = useRef(null);
  const chatEndRef = useRef(null);
  const [hlsUrl, setHlsUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [qualities, setQualities] = useState([]);
  const [selectedQuality, setSelectedQuality] = useState("auto");
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! I'm your AI assistant. How can I help you today?",
      timestamp: "10:30 AM"
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [history, setHistory] = useState([]); // backend history
  const [isThinking, setIsThinking] = useState(false); // loading indicator

  const CHAT_ENDPOINT =
    "https://ninad-k2005--videorag-chat-endpoint.modal.run";



  useEffect(() => {
    const fetchVideo = async () => {
      try {
        const res = await axios.get(`http://localhost:3000/video/${id}`);
        setHlsUrl(res.data.hls_url);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchVideo();
  }, [id]);

  useEffect(() => {
    if (!videoRef.current || !hlsUrl) return;

    playerRef.current = videojs(videoRef.current, {
      controls: true,
      fluid: true,
      preload: "auto",
      html5: {
        vhs: {
          overrideNative: true,
          enableLowInitialPlaylist: true
        }
      }
    });

    playerRef.current.src({
      src: hlsUrl,
      type: "application/x-mpegURL"
    });
    playerRef.current.on("play", async () => {
      try {
        // await axios.post(CHAT_ENDPOINT, {
        //   video_id: id,
        //   query: "__warmup__",
        //   history: [],
        //   top_k: 5
        // });
        console.log("Chat container warmed");
      } catch {
        // safe to ignore
      }
    });


    // Setup quality levels
    playerRef.current.ready(() => {
      const qualityLevels = playerRef.current.qualityLevels();
      qualityLevels.on('addqualitylevel', () => {
        const levels = [];
        for (let i = 0; i < qualityLevels.length; i++) {
          const level = qualityLevels[i];
          levels.push({
            index: i,
            height: level.height,
            width: level.width,
            bitrate: level.bitrate,
            label: `${level.height}p`
          });
        }
        // Sort by height (lowest to highest)
        levels.sort((a, b) => a.height - b.height);
        setQualities(levels);

        // Enable auto quality by default (all levels enabled)
        for (let i = 0; i < qualityLevels.length; i++) {
          qualityLevels[i].enabled = true;
        }
      });
    });

    return () => {
      playerRef.current?.dispose();
      playerRef.current = null;
    };
  }, [hlsUrl]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleQualityChange = (qualityIndex) => {
    if (!playerRef.current) return;

    const qualityLevels = playerRef.current.qualityLevels();

    if (qualityIndex === "auto") {
      // Enable all qualities for automatic switching
      for (let i = 0; i < qualityLevels.length; i++) {
        qualityLevels[i].enabled = true;
      }
      setSelectedQuality("auto");
    } else {
      // Disable all qualities except selected one
      for (let i = 0; i < qualityLevels.length; i++) {
        qualityLevels[i].enabled = i === qualityIndex;
      }
      setSelectedQuality(qualityIndex);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isThinking) return;

    const userMsg = {
      role: "user",
      content: inputMessage
    };

    // UI update (keep your UI exactly)
    setMessages(prev => [
      ...prev,
      {
        ...userMsg,
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit"
        })
      }
    ]);

    setInputMessage("");
    setIsThinking(true);

    // keep ONLY last 3 Q&A pairs (6 messages)
    const newHistory = [...history, userMsg].slice(-6);
    setHistory(newHistory);

    try {
      const res = await axios.post(CHAT_ENDPOINT, {
        video_id: id,
        query: userMsg.content,
        history: newHistory,
        top_k: 5
      });

      const assistantText = res.data;

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: assistantText,
          timestamp: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
          })
        }
      ]);

      setHistory(prev =>
        [...prev, { role: "assistant", content: assistantText }].slice(-6)
      );
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
          timestamp: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
          })
        }
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-6">
            <div className="absolute inset-0 border-4 border-red-600/20 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-transparent border-t-red-600 rounded-full animate-spin"></div>
          </div>
          <p className="text-gray-400 text-lg font-light tracking-wide">Loading your content...</p>
        </div>
      </div>
    );
  }

  if (!hlsUrl) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="text-center">
          <svg className="w-24 h-24 mx-auto mb-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="text-gray-300 text-xl font-light mb-2">Content Unavailable</p>
          <p className="text-gray-500 text-sm">This video is currently not available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white" style={{ fontFamily: "'Netflix Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
        
        body {
          overflow-x: hidden;
        }

        .video-js {
          font-family: inherit;
        }

        .video-js .vjs-big-play-button {
          background-color: rgba(229, 9, 20, 0.9);
          border: none;
          border-radius: 50%;
          width: 80px;
          height: 80px;
          line-height: 80px;
          font-size: 48px;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          transition: all 0.3s ease;
        }

        .video-js .vjs-big-play-button:hover {
          background-color: rgba(229, 9, 20, 1);
          transform: translate(-50%, -50%) scale(1.1);
        }

        .video-js .vjs-control-bar {
          background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
          height: 60px;
        }

        .video-js .vjs-slider {
          background-color: rgba(255,255,255,0.2);
        }

        .video-js .vjs-play-progress {
          background-color: #e50914;
        }

        .video-js .vjs-load-progress {
          background: rgba(255,255,255,0.3);
        }

        .quality-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .quality-badge:hover {
          transform: translateY(-2px);
        }

        .chat-message {
          animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .chat-bubble {
          word-wrap: break-word;
          max-width: 100%;
        }

        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }

        .scrollbar-thin::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 10px;
        }

        .scrollbar-thin::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 10px;
        }

        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .gradient-border {
          position: relative;
        }

        .gradient-border::before {
          content: '';
          position: absolute;
          inset: 0;
          border-radius: 0.75rem;
          padding: 1px;
          background: linear-gradient(135deg, rgba(229, 9, 20, 0.5), rgba(229, 9, 20, 0.1));
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          pointer-events: none;
        }
      `}</style>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-b from-black/80 to-transparent backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-3xl font-bold tracking-tight" style={{ fontFamily: "'Bebas Neue', sans-serif", letterSpacing: '0.05em' }}>
            <span className="text-red-600">VIDEO</span> LEARNING
          </h1>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setChatOpen(!chatOpen)}
              className="relative p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-all duration-300 group"
            >
              <svg className="w-6 h-6 text-gray-300 group-hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              {messages.length > 1 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-600 rounded-full text-xs flex items-center justify-center font-semibold">
                  {messages.length - 1}
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-20 pb-8">
        <div className="container mx-auto px-6">
          <div className="flex gap-6">
            {/* Video Section */}
            <div className={`transition-all duration-500 ${chatOpen ? 'w-2/3' : 'w-full'}`}>
              <div className="bg-black rounded-xl overflow-hidden shadow-2xl">
                <div data-vjs-player>
                  <video
                    ref={videoRef}
                    className="video-js vjs-big-play-centered"
                    playsInline
                  />
                </div>
              </div>

              {/* Quality Controls */}
              {qualities.length > 0 && (
                <div className="mt-6">
                  <div className="flex items-center gap-3 mb-3">
                    <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                    </svg>
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Video Quality</h3>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    <button
                      onClick={() => handleQualityChange("auto")}
                      className={`quality-badge px-5 py-2.5 rounded-lg font-medium text-sm transition-all duration-300 ${selectedQuality === "auto"
                        ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                        : "bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white"
                        }`}
                    >
                      Auto
                    </button>
                    {qualities.map((quality) => (
                      <button
                        key={quality.index}
                        onClick={() => handleQualityChange(quality.index)}
                        className={`quality-badge px-5 py-2.5 rounded-lg font-medium text-sm transition-all duration-300 ${selectedQuality === quality.index
                          ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                          : "bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white"
                          }`}
                      >
                        {quality.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Video Info */}
              <div className="mt-8">
                <h2 className="text-2xl font-semibold mb-3">Watch and interact with your content</h2>
                <p className="text-gray-400 text-sm leading-relaxed">
                  Experience seamless video playback with adaptive quality streaming. Use the AI assistant to get help, ask questions, or learn more about the content.
                </p>
              </div>
            </div>

            {/* Chat Panel */}
            <div className={`transition-all duration-500 ${chatOpen ? 'w-1/3 opacity-100' : 'w-0 opacity-0 overflow-hidden'}`}>
              {chatOpen && (
                <div className="bg-gradient-to-br from-[#141414] to-[#0a0a0a] rounded-xl shadow-2xl h-[calc(100vh-8rem)] flex flex-col gradient-border">
                  {/* Chat Header */}
                  <div className="p-5 border-b border-white/10 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-red-600 to-red-700 flex items-center justify-center text-lg font-bold shadow-lg">
                        AI
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">AI Assistant</h3>
                        <div className="flex items-center gap-1.5">
                          <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                          <span className="text-xs text-gray-400">Online</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setChatOpen(false)}
                      className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                    >
                      <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>

                  {/* Chat Messages */}
                  <div className="flex-1 overflow-y-auto p-5 space-y-4 scrollbar-thin">
                    {messages.map((msg, idx) => (
                      <div key={idx} className="chat-message">
                        {msg.role === "user" ? (
                          /* USER MESSAGE — keep bubble */
                          <div className="flex justify-end">
                            <div className="max-w-[70%]">
                              <div className="chat-bubble rounded-2xl px-4 py-3 bg-red-600 text-white rounded-tr-sm">
                                <p className="text-sm whitespace-pre-line">{msg.content}</p>
                              </div>
                              <p className="text-xs text-gray-500 mt-1.5 text-right">
                                {msg.timestamp}
                              </p>
                            </div>
                          </div>
                        ) : (
                          /* ASSISTANT MESSAGE — ChatGPT style */
                          <div className="w-full px-2 py-2">
                            <div className="text-gray-100 text-sm leading-relaxed whitespace-pre-line">
                              {msg.content}
                            </div>
                            <p className="text-xs text-gray-500 mt-2">
                              {msg.timestamp}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                    {isThinking && (
                      <div className="chat-message flex justify-start">
                        <div className="chat-bubble rounded-2xl px-4 py-3 bg-white/5 text-gray-400 italic">
                          AI is thinking…
                        </div>
                      </div>
                    )}

                    <div ref={chatEndRef} />
                  </div>

                  {/* Chat Input */}
                  <div className="p-4 border-t border-white/10">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                        placeholder="Type a message..."
                        className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-red-600/50 focus:bg-white/10 transition-all placeholder-gray-500"
                      />
                      <button
                        onClick={handleSendMessage}
                        className="bg-red-600 hover:bg-red-700 text-white rounded-lg px-4 py-3 transition-all duration-300 hover:shadow-lg hover:shadow-red-600/30 disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={!inputMessage.trim()}
                      >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}