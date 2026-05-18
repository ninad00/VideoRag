// import { useEffect, useState } from "react";
// import { useNavigate } from "react-router-dom";
// import axios from "axios";

// export default function HomePage() {
//   const [videos, setVideos] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const navigate = useNavigate();

//   useEffect(() => {
//     const fetchVideos = async () => {
//       try {
//         const res = await axios.get("http://localhost:3000/video");
//         setVideos(res.data);
//       } catch (err) {
//         console.error(err);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchVideos();
//   }, []);

//   if (loading) {
//     return (
//       <div className="flex items-center justify-center h-screen text-gray-400">
//         Loading videos...
//       </div>
//     );
//   }

//   return (
//     <div className="min-h-screen bg-gray-900 p-6">
//       <h1 className="text-2xl font-semibold text-white mb-6">
//         Videos
//       </h1>

//       <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
//         {videos.map(video => (
//           <div
//             key={video.id}
//             onClick={() => navigate(`/watch/${video.id}`)}
//             className="cursor-pointer group"
//           >
//             <img
//               src={video.thumbnail_url}
//               alt={video.title}
//               className="rounded-lg w-full h-40 object-cover group-hover:opacity-80 transition"
//               loading="lazy"
//             />
//             <h3 className="text-sm text-gray-200 mt-2 truncate">
//               {video.title}
//               <br />
//               ID:
//               <br />
//               {video.id}
//             </h3>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// }
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import { Menu } from "lucide-react";
import { useAuth } from "./Context/authContext.jsx";
import axios from "axios";
import {
  Search,
  Play,
  MonitorPlay,
  Clock3,
} from "lucide-react";

export default function HomePage() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const { user, logout } = useAuth();

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    fetchVideos();
  }, []);

  useEffect(() => {
    const delay = setTimeout(() => {
      fetchVideos(search);
    }, 400);

    return () => clearTimeout(delay);
  }, [search]);

  const fetchVideos = async (query = "") => {
    try {
      setLoading(true);

      const res = await axios.get(
        `https://videositebackend-production.up.railway.app/video/`,
        {
          params: {
            search: query,
          },
        }
      );

      setVideos(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-white">

      {/* Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0b0f19]/80 border-b border-white/10">

        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">

          {/* Logo */}
          <div className="flex items-center gap-3">

            <div className="p-2 rounded-xl bg-white/10 border border-white/10">
              <MonitorPlay size={22} />
            </div>

            <div>
              <h1 className="font-bold text-lg tracking-tight">
                StreamFlow
              </h1>

              <p className="text-[11px] text-gray-500">
                Video Platform
              </p>
            </div>
          </div>

          {/* Search */}
          <div className="w-full max-w-xl mx-10 hidden md:block">

            <div className="relative">

              <Search
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500"
              />

              <input
                type="text"
                placeholder="Search videos..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-2xl pl-11 pr-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-white/10 focus:border-white/20 transition"
              />

            </div>
          </div>

          {/* Upload */}
          <div className="flex items-center gap-3">

            <button
              onClick={() => navigate("/upload")}
              className="bg-white text-black hover:bg-gray-200 transition px-5 py-2 rounded-xl text-sm font-semibold"
            >
              Upload
            </button>

            <button
              onClick={() => setSidebarOpen(true)}
              className="bg-white/10 border border-white/10 p-3 rounded-xl hover:bg-white/20 transition"
            >
              <Menu size={18} />
            </button>

          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-7xl mx-auto px-6 pt-10 pb-6">

        <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-white/[0.07] to-white/[0.02] p-10">

          {/* Glow */}
          <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 blur-3xl rounded-full" />

          <div className="relative z-10">

            <div className="inline-flex items-center gap-2 bg-white/10 border border-white/10 rounded-full px-4 py-2 mb-6 text-sm text-gray-300">
              <Play size={14} />
              Adaptive Streaming Enabled
            </div>

            <h2 className="text-5xl font-bold leading-tight max-w-2xl">
              Stream your videos
              <span className="block text-gray-400">
                with modern playback.
              </span>
            </h2>

            <p className="text-gray-400 mt-5 max-w-xl leading-relaxed">
              Upload and watch high-quality videos with subtitles,
              adaptive bitrate streaming and multi-device playback.
            </p>

          </div>
        </div>
      </section>

      {/* Videos */}
      <main className="max-w-7xl mx-auto px-6 pb-14">

        {/* Heading */}
        <div className="flex items-center justify-between mb-8">

          <div>
            <h3 className="text-2xl font-bold">
              Trending Videos
            </h3>

            <p className="text-gray-500 mt-1">
              {videos.length} videos available
            </p>
          </div>

        </div>

        {/* Loading */}
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-7">

            {[...Array(8)].map((_, i) => (
              <div
                key={i}
                className="animate-pulse"
              >
                <div className="bg-white/5 rounded-2xl aspect-video mb-3" />
                <div className="h-4 bg-white/5 rounded w-3/4 mb-2" />
                <div className="h-3 bg-white/5 rounded w-1/2" />
              </div>
            ))}

          </div>
        ) : videos.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center">

            <div className="p-5 rounded-3xl bg-white/5 border border-white/10 mb-5">
              <Search size={40} className="text-gray-500" />
            </div>

            <h3 className="text-2xl font-semibold mb-2">
              No videos found
            </h3>

            <p className="text-gray-500">
              Try searching with a different keyword.
            </p>
          </div>
        ) : (

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-7">

            {videos.map((video) => (

              <div
                key={video.id}
                onClick={() => navigate(`/watch/${video.id}`)}
                className="group cursor-pointer"
              >

                {/* Thumbnail */}
                <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5">

                  <img
                    src={video.thumbnail_url}
                    alt={video.title}
                    loading="lazy"
                    className="w-full aspect-video object-cover group-hover:scale-105 transition duration-500"
                  />

                  {/* Overlay */}
                  <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition" />

                  {/* Play Button */}
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition">

                    <div className="p-4 rounded-full bg-white/20 backdrop-blur-md border border-white/20">
                      <Play fill="white" size={28} />
                    </div>

                  </div>

                  {/* Duration */}
                  {/* <div className="absolute bottom-3 right-3 bg-black/70 backdrop-blur-md px-2 py-1 rounded-lg text-xs flex items-center gap-1">
                    <Clock3 size={12} />
                    12:45
                  </div> */}

                </div>

                {/* Content */}
                <div className="mt-4">

                  <h4 className="font-semibold text-[15px] line-clamp-2 group-hover:text-gray-300 transition">
                    {video.title}
                  </h4>

                  <div className="flex items-center justify-between mt-2 text-sm text-gray-500">

                    {/* <span>
                      Stream Video
                    </span> */}
                    <span className="truncate max-w-[100px]">
                      {video.id}
                    </span>
                    <span className="truncate max-w-[100px]">
                      {video.uploaderName || "Anonymous"}
                    </span>

                  </div>
                </div>

              </div>

            ))}

          </div>

        )}
      </main>
      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        user={user}
        logout={logout}
      />
    </div>
  );
}