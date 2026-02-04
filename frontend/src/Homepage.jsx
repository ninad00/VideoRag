import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function HomePage() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const res = await axios.get("http://localhost:3000/video");
        setVideos(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchVideos();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-400">
        Loading videos...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <h1 className="text-2xl font-semibold text-white mb-6">
        Videos
      </h1>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
        {videos.map(video => (
          <div
            key={video.id}
            onClick={() => navigate(`/watch/${video.id}`)}
            className="cursor-pointer group"
          >
            <img
              src={video.thumbnail_url}
              alt={video.title}
              className="rounded-lg w-full h-40 object-cover group-hover:opacity-80 transition"
              loading="lazy"
            />
            <h3 className="text-sm text-gray-200 mt-2 truncate">
              {video.title}
              <br />
              ID:
              <br />
              {video.id}
            </h3>
          </div>
        ))}
      </div>
    </div>
  );
}
