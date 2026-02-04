import { useState } from "react";
import axios from "axios";

export default function UploadPage() {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState("");

    const onFileChange = (e) => {
        const f = e.target.files[0];
        if (!f || f.type !== "video/mp4") {
            alert("Only MP4 files allowed");
            return;
        }
        setFile(f);
    };

    const upload = async () => {
        if (!file) return;

        setUploading(true);
        setProgress(0);
        setMessage("");

        try {
            const { data } = await axios.post("http://localhost:3000/upload/posturl", {
                filename: file.name,
                contentType: file.type
            });

            await axios.put(data.url, file, {
                headers: { "Content-Type": file.type },
            });

            setMessage("Upload successful! Processing video...");
            setFile(null);
        } catch (err) {
            console.error(err);
            setMessage("Upload failed");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
            <div className="bg-gray-800 p-6 rounded-lg w-full max-w-md">
                <h2 className="text-white text-xl mb-4">Upload Video</h2>

                <input
                    type="file"
                    accept="video/mp4"
                    onChange={onFileChange}
                    className="text-gray-300 mb-4"
                />

                {file && (
                    <p className="text-gray-400 text-sm mb-2">
                        {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                )}

                {uploading && (
                    <div className="w-full bg-gray-700 rounded h-2 mb-3">
                        <div
                            className="bg-green-500 h-2 rounded"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                )}

                <button
                    onClick={upload}
                    disabled={!file || uploading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded disabled:opacity-50"
                >
                    {uploading ? "Uploading..." : "Upload"}
                </button>

                {message && (
                    <p className="text-gray-400 text-sm mt-3">{message}</p>
                )}
            </div>
        </div>
    );
}
