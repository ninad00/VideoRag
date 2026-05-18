// import { useEffect, useRef, useState } from "react";
// import { useParams } from "react-router-dom";
// import axios from "axios";
// import videojs from "video.js";
// import "video.js/dist/video-js.css";
// import "videojs-contrib-quality-levels";

// export default function PlayerPage() {
//   const { id } = useParams();
//   const videoRef = useRef(null);
//   const playerRef = useRef(null);

//   const [hlsUrl, setHlsUrl] = useState("");
//   const [loading, setLoading] = useState(true);
//   const [qualities, setQualities] = useState([]);
//   const [selectedQuality, setSelectedQuality] = useState("auto");

//   useEffect(() => {
//     const fetchVideo = async () => {
//       try {
//         const res = await axios.get(`http://localhost:3000/video/${id}`);
//         setHlsUrl(res.data.hls_url);
//       } catch (err) {
//         console.error(err);
//       } finally {
//         setLoading(false);
//       }
//     };
//     fetchVideo();
//   }, [id]);

//   useEffect(() => {
//     if (!videoRef.current || !hlsUrl) return;

//     playerRef.current = videojs(videoRef.current, {
//       controls: true,
//       fluid: true,
//       preload: "auto",
//       html5: {
//         vhs: {
//           overrideNative: true,
//           enableLowInitialPlaylist: true
//         }
//       }
//     });

//     playerRef.current.src({
//       src: hlsUrl,
//       type: "application/x-mpegURL"
//     });

//     // ✅ ADD SUBTITLES TRACK (HLS patched already → this works)
//     playerRef.current.ready(() => {
//       playerRef.current.addRemoteTextTrack(
//         {
//           kind: "subtitles",
//           src: `${hlsUrl.replace("master.m3u8", "subtitles/en.vtt")}`,
//           srclang: "en",
//           label: "English",
//           default: true
//         },
//         false
//       );

//       const qualityLevels = playerRef.current.qualityLevels();
//       qualityLevels.on("addqualitylevel", () => {
//         const levels = [];
//         for (let i = 0; i < qualityLevels.length; i++) {
//           const level = qualityLevels[i];
//           levels.push({
//             index: i,
//             height: level.height,
//             width: level.width,
//             bitrate: level.bitrate,
//             label: `${level.height}p`
//           });
//         }

//         levels.sort((a, b) => a.height - b.height);
//         setQualities(levels);

//         for (let i = 0; i < qualityLevels.length; i++) {
//           qualityLevels[i].enabled = true;
//         }
//       });
//     });

//     return () => {
//       playerRef.current?.dispose();
//       playerRef.current = null;
//     };
//   }, [hlsUrl]);

//   const handleQualityChange = (qualityIndex) => {
//     if (!playerRef.current) return;

//     const qualityLevels = playerRef.current.qualityLevels();

//     if (qualityIndex === "auto") {
//       for (let i = 0; i < qualityLevels.length; i++) {
//         qualityLevels[i].enabled = true;
//       }
//       setSelectedQuality("auto");
//     } else {
//       for (let i = 0; i < qualityLevels.length; i++) {
//         qualityLevels[i].enabled = i === qualityIndex;
//       }
//       setSelectedQuality(qualityIndex);
//     }
//   };

//   if (loading) {
//     return (
//       <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
//         <p className="text-gray-400 text-lg">Loading your content...</p>
//       </div>
//     );
//   }

//   if (!hlsUrl) {
//     return (
//       <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
//         <p className="text-gray-300 text-xl">Content Unavailable</p>
//       </div>
//     );
//   }

//   return (
//     <div className="min-h-screen bg-[#0a0a0a] text-white">
//       <main className="pt-20 pb-8">
//         <div className="container mx-auto px-6">
//           <div className="flex gap-6">

//             {/* Video Section */}
//             <div className="w-full">
//               <div className="bg-black rounded-xl overflow-hidden shadow-2xl">
//                 <div data-vjs-player>
//                   <video
//                     ref={videoRef}
//                     className="video-js vjs-big-play-centered"
//                     playsInline
//                   />
//                 </div>
//               </div>

//               {/* Quality Controls */}
//               {qualities.length > 0 && (
//                 <div className="mt-6">
//                   <div className="flex flex-wrap gap-3">
//                     <button
//                       onClick={() => handleQualityChange("auto")}
//                       className={`px-5 py-2.5 rounded-lg ${selectedQuality === "auto"
//                         ? "bg-red-600 text-white"
//                         : "bg-white/5 text-gray-400"
//                         }`}
//                     >
//                       Auto
//                     </button>

//                     {qualities.map((quality) => (
//                       <button
//                         key={quality.index}
//                         onClick={() => handleQualityChange(quality.index)}
//                         className={`px-5 py-2.5 rounded-lg ${selectedQuality === quality.index
//                           ? "bg-red-600 text-white"
//                           : "bg-white/5 text-gray-400"
//                           }`}
//                       >
//                         {quality.label}
//                       </button>
//                     ))}
//                   </div>
//                 </div>
//               )}

//               {/* Info */}
//               <div className="mt-8">
//                 <h2 className="text-2xl font-semibold mb-3">
//                   Video with Subtitles
//                 </h2>
//                 <p className="text-gray-400 text-sm">
//                   Adaptive streaming with subtitle support enabled.
//                 </p>
//               </div>
//             </div>
//           </div>
//         </div>
//       </main>
//     </div>
//   );
// }

import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import videojs from "video.js";
import "video.js/dist/video-js.css";
import "videojs-contrib-quality-levels";

import {
  Play,
  MonitorPlay,
  Settings2,
  ArrowLeft,
  BadgeCheck,
  Captions,
  Radio,
} from "lucide-react";

export default function PlayerPage() {

  const { id } = useParams();
  const navigate = useNavigate();

  const videoRef = useRef(null);
  const playerRef = useRef(null);

  const [video, setVideo] = useState(null);
  const [hlsUrl, setHlsUrl] = useState("");
  const [loading, setLoading] = useState(true);

  const [qualities, setQualities] = useState([]);
  const [selectedQuality, setSelectedQuality] = useState("auto");

  // FETCH VIDEO
  useEffect(() => {

    const fetchVideo = async () => {

      try {

        const res = await axios.get(
          `https://videositebackend-production.up.railway.app/video/${id}`
        );

        setVideo(res.data);
        setHlsUrl(res.data.hls_url);

      } catch (err) {

        console.error(err);

      } finally {

        setLoading(false);

      }
    };

    fetchVideo();

  }, [id]);

  // INIT PLAYER
  useEffect(() => {

    if (!videoRef.current || !hlsUrl) return;

    playerRef.current = videojs(
      videoRef.current,
      {
        controls: true,
        fluid: true,
        preload: "auto",

        html5: {
          vhs: {
            overrideNative: true,
            enableLowInitialPlaylist: true,
          },
        },
      }
    );

    playerRef.current.src({
      src: hlsUrl,
      type: "application/x-mpegURL",
    });

    playerRef.current.ready(() => {

      // SUBTITLES
      playerRef.current.addRemoteTextTrack(
        {
          kind: "subtitles",
          src: `${hlsUrl.replace(
            "master.m3u8",
            "subtitles/en.vtt"
          )}`,
          srclang: "en",
          label: "English",
          default: true,
        },
        false
      );

      // QUALITY LEVELS
      const qualityLevels =
        playerRef.current.qualityLevels();

      qualityLevels.on("addqualitylevel", () => {

        const levels = [];

        for (let i = 0; i < qualityLevels.length; i++) {

          const level = qualityLevels[i];

          levels.push({
            index: i,
            height: level.height,
            width: level.width,
            bitrate: level.bitrate,
            label: `${level.height}p`,
          });
        }

        levels.sort(
          (a, b) => a.height - b.height
        );

        setQualities(levels);

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

  // QUALITY CHANGE
  const handleQualityChange = (qualityIndex) => {

    if (!playerRef.current) return;

    const qualityLevels =
      playerRef.current.qualityLevels();

    if (qualityIndex === "auto") {

      for (let i = 0; i < qualityLevels.length; i++) {
        qualityLevels[i].enabled = true;
      }

      setSelectedQuality("auto");

    } else {

      for (let i = 0; i < qualityLevels.length; i++) {
        qualityLevels[i].enabled =
          i === qualityIndex;
      }

      setSelectedQuality(qualityIndex);
    }
  };

  // LOADING
  if (loading) {

    return (
      <div className="min-h-screen bg-[#0b0f19] flex items-center justify-center">

        <div className="text-center">

          <div className="w-16 h-16 border-4 border-white/10 border-t-white rounded-full animate-spin mx-auto mb-5" />

          <p className="text-gray-400 text-lg">
            Loading stream...
          </p>

        </div>

      </div>
    );
  }

  // NO VIDEO
  if (!hlsUrl) {

    return (
      <div className="min-h-screen bg-[#0b0f19] flex items-center justify-center">

        <div className="text-center">

          <p className="text-3xl font-bold mb-3">
            Video unavailable
          </p>

          <p className="text-gray-500">
            Unable to load stream.
          </p>

        </div>

      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0b0f19] text-white">

      {/* NAVBAR */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0b0f19]/80 border-b border-white/10">

        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">

          {/* Left */}
          <div className="flex items-center gap-4">

            <button
              onClick={() => navigate("/")}
              className="p-2 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition"
            >
              <ArrowLeft size={20} />
            </button>

            <div className="flex items-center gap-3">

              <div className="p-2 rounded-xl bg-white/10 border border-white/10">
                <MonitorPlay size={22} />
              </div>

              <div>

                <h1 className="font-bold text-lg">
                  StreamFlow
                </h1>

                <p className="text-[11px] text-gray-500">
                  Adaptive Streaming
                </p>

              </div>
            </div>
          </div>

          {/* LIVE */}
          {/* <div className="flex items-center gap-2 bg-white/5 border border-white/10 px-4 py-2 rounded-full">

            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />

            <span className="text-sm text-gray-300">
              LIVE STREAM
            </span>

          </div> */}
        </div>
      </header>

      {/* CONTENT */}
      <main className="max-w-7xl mx-auto px-6 py-8">

        <div className="grid lg:grid-cols-[1fr_350px] gap-8">

          {/* VIDEO SIDE */}
          <div>

            {/* PLAYER */}
            <div className="overflow-hidden rounded-3xl border border-white/10 bg-black shadow-2xl">

              <div data-vjs-player>
                <video
                  ref={videoRef}
                  className="video-js vjs-big-play-centered"
                  playsInline
                />
              </div>

            </div>

            {/* Description */}
            <div className="mt-7 max-w-4xl">

              <h2 className="text-lg font-semibold text-white mb-3">
                Description
              </h2>

              <p className="text-gray-400 leading-relaxed whitespace-pre-wrap">
                {video?.description || "No description available."}
              </p>

            </div>

          </div>

          {/* SIDEBAR */}
          <aside className="space-y-6">

            {/* QUALITY */}
            <div className="bg-white/[0.04] border border-white/10 rounded-3xl p-6 backdrop-blur-xl">

              <div className="flex items-center gap-3 mb-6">

                <div className="p-2 rounded-xl bg-white/10">
                  <Settings2 size={18} />
                </div>

                <h3 className="text-xl font-semibold">
                  Playback Quality
                </h3>

              </div>

              <div className="grid grid-cols-2 gap-3">

                <button
                  onClick={() =>
                    handleQualityChange("auto")
                  }
                  className={`rounded-2xl px-4 py-3 text-sm font-medium transition ${selectedQuality === "auto"
                    ? "bg-white text-black"
                    : "bg-white/5 border border-white/10 hover:bg-white/10 text-gray-300"
                    }`}
                >
                  Auto
                </button>

                {qualities.map((quality) => (

                  <button
                    key={quality.index}
                    onClick={() =>
                      handleQualityChange(
                        quality.index
                      )
                    }
                    className={`rounded-2xl px-4 py-3 text-sm font-medium transition ${selectedQuality ===
                      quality.index
                      ? "bg-white text-black"
                      : "bg-white/5 border border-white/10 hover:bg-white/10 text-gray-300"
                      }`}
                  >
                    {quality.label}
                  </button>

                ))}

              </div>

            </div>

            {/* STREAM INFO */}
            <div className="bg-white/[0.04] border border-white/10 rounded-3xl p-6 backdrop-blur-xl">

              <h3 className="text-xl font-semibold mb-6">
                Stream Info
              </h3>

              <div className="space-y-4">

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">
                    Format
                  </span>

                  <span className="font-medium">
                    HLS (.m3u8)
                  </span>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">
                    Subtitles
                  </span>

                  <span className="font-medium">
                    Enabled
                  </span>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">
                    Quality Levels
                  </span>

                  <span className="font-medium">
                    {qualities.length || "—"}
                  </span>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">
                    Current
                  </span>

                  <span className="font-medium">
                    {selectedQuality === "auto"
                      ? "Auto"
                      : qualities.find(
                        (q) =>
                          q.index ===
                          selectedQuality
                      )?.label}
                  </span>
                </div>

              </div>

            </div>

            {/* VIDEO ID */}
            <div className="bg-white/[0.04] border border-white/10 rounded-3xl p-6 backdrop-blur-xl">

              <h3 className="text-xl font-semibold mb-4">
                Creator
              </h3>

              <div className="bg-black/30 border border-white/10 rounded-2xl px-4 py-3 text-sm text-gray-400 break-all">
                {video?.uploaderName}
              </div>


            </div>

          </aside>

        </div>

      </main>
    </div>
  );
}