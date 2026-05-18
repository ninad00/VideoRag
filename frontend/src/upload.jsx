// import { useState } from "react";
// import axios from "axios";

// export default function UploadPage() {
//     const [file, setFile] = useState(null);
//     const [uploading, setUploading] = useState(false);
//     const [progress, setProgress] = useState(0);
//     const [message, setMessage] = useState("");
//     const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB

//     const onFileChange = (e) => {
//         const f = e.target.files[0];
//         if (!f || f.type !== "video/mp4") {
//             alert("Only MP4 files allowed");
//             return;
//         }
//         setFile(f);
//     };

//     const upload = async () => {
//         if (!file) return;

//         setUploading(true);
//         setProgress(0);
//         setMessage("");

//         try {
//             // const { data } = await axios.post("http://localhost:3000/upload/posturl", {
//             //     filename: file.name,
//             //     contentType: file.type
//             // });

//             // await axios.put(data.url, file, {
//             //     headers: { "Content-Type": file.type },
//             // });


//             const startRes = await axios.post(
//                 "http://localhost:3000/upload/multipart/start",
//                 {
//                     filename: file.name,
//                     contentType: file.type
//                 }
//             );

//             const {
//                 uploadId,
//                 key,
//                 fileId
//             } = startRes.data;


//             const totalParts = Math.ceil(
//                 file.size / CHUNK_SIZE
//             );


//             const signedUrls = await Promise.all(
//                 Array.from(
//                     { length: totalParts },
//                     async (_, index) => {

//                         const partNumber = index + 1;

//                         const res = await axios.post(
//                             "http://localhost:3000/upload/multipart/sign",
//                             {
//                                 key,
//                                 uploadId,
//                                 partNumber
//                             }
//                         );

//                         return {
//                             partNumber,
//                             url: res.data.url
//                         };
//                     }
//                 )
//             );


//             const uploadedParts = await Promise.all(
//                 signedUrls.map(async part => {

//                     const start =
//                         (part.partNumber - 1) *
//                         CHUNK_SIZE;

//                     const end = Math.min(
//                         start + CHUNK_SIZE,
//                         file.size
//                     );

//                     const blob = file.slice(start, end);

//                     const uploadRes = await axios.put(
//                         part.url,
//                         blob,
//                         {
//                             headers: {
//                                 "Content-Type": file.type
//                             }
//                         }
//                     );

//                     return {
//                         ETag:
//                             uploadRes.headers.etag.replaceAll(
//                                 '"',
//                                 ""
//                             ),

//                         PartNumber:
//                             part.partNumber
//                     };
//                 })
//             );



//             await axios.post(
//                 "http://localhost:3000/upload/multipart/complete",
//                 {
//                     key,
//                     uploadId,
//                     fileId,
//                     parts: uploadedParts
//                 }
//             );

//             setMessage("Upload successful! Processing video...");
//             setFile(null);
//         } catch (err) {
//             console.error(err);
//             setMessage("Upload failed");
//         } finally {
//             setUploading(false);
//         }
//     };

//     return (
//         <div className="min-h-screen bg-gray-900 flex items-center justify-center">
//             <div className="bg-gray-800 p-6 rounded-lg w-full max-w-md">
//                 <h2 className="text-white text-xl mb-4">Upload Video</h2>

//                 <input
//                     type="file"
//                     accept="video/mp4"
//                     onChange={onFileChange}
//                     className="text-gray-300 mb-4"
//                 />

//                 {file && (
//                     <p className="text-gray-400 text-sm mb-2">
//                         {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
//                     </p>
//                 )}

//                 {uploading && (
//                     <div className="w-full bg-gray-700 rounded h-2 mb-3">
//                         <div
//                             className="bg-green-500 h-2 rounded"
//                             style={{ width: `${progress}%` }}
//                         />
//                     </div>
//                 )}

//                 <button
//                     onClick={upload}
//                     disabled={!file || uploading}
//                     className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded disabled:opacity-50"
//                 >
//                     {uploading ? "Uploading..." : "Upload"}
//                 </button>

//                 {message && (
//                     <p className="text-gray-400 text-sm mt-3">{message}</p>
//                 )}
//             </div>
//         </div>
//     );
// }
import { useState } from "react";
import axios from "axios";
import {
    UploadCloud,
    Film,
    CheckCircle2,
    Loader2,
    XCircle,
} from "lucide-react";
import { useAuth } from "./Context/authContext";


export default function UploadPage() {

    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState("");
    const [description, setDescription] = useState("");
    const { user } = useAuth();

    const CHUNK_SIZE = 10 * 1024 * 1024;

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

            // START MULTIPART
            const startRes = await axios.post(
                "https://videositebackend-production.up.railway.app/upload/multipart/start",
                {
                    filename: file.name,
                    uploaderName: user?.username || "Anonymous",
                    description: description,
                    contentType: file.type,
                }
            );

            const {
                uploadId,
                key,
                fileId,
            } = startRes.data;

            const totalParts = Math.ceil(
                file.size / CHUNK_SIZE
            );

            // GET SIGNED URLS
            const signedUrls = await Promise.all(

                Array.from(
                    { length: totalParts },

                    async (_, index) => {

                        const partNumber = index + 1;

                        const res = await axios.post(
                            "https://videositebackend-production.up.railway.app/upload/multipart/sign",
                            {
                                key,
                                uploadId,
                                partNumber,
                            }
                        );

                        return {
                            partNumber,
                            url: res.data.url,
                        };
                    }
                )
            );

            // UPLOAD CHUNKS
            const uploadedParts = [];

            for (let i = 0; i < signedUrls.length; i++) {

                const part = signedUrls[i];

                const start =
                    (part.partNumber - 1) *
                    CHUNK_SIZE;

                const end = Math.min(
                    start + CHUNK_SIZE,
                    file.size
                );

                const blob = file.slice(start, end);

                const uploadRes = await axios.put(
                    part.url,
                    blob,
                    {
                        headers: {
                            "Content-Type": file.type,
                        },
                    }
                );

                uploadedParts.push({
                    ETag: uploadRes.headers.etag.replaceAll(
                        '"',
                        ""
                    ),
                    PartNumber: part.partNumber,
                });

                // Progress
                setProgress(
                    Math.round(
                        ((i + 1) / totalParts) * 100
                    )
                );
            }

            // COMPLETE UPLOAD
            await axios.post(
                "https://videositebackend-production.up.railway.app/upload/multipart/complete",
                {
                    key,
                    uploadId,
                    fileId,
                    parts: uploadedParts,
                }
            );

            setMessage("Upload successful! Processing video...");
            setFile(null);
            setDescription("");
        } catch (err) {

            console.error(err);
            setMessage("Upload failed");

        } finally {

            setUploading(false);

        }
    };

    return (
        <div className="min-h-screen bg-[#0b0f19] text-white flex items-center justify-center px-6 py-10 overflow-hidden">

            {/* Background Glow */}
            <div className="absolute top-10 left-10 w-72 h-72 bg-blue-500/10 blur-3xl rounded-full" />
            <div className="absolute bottom-10 right-10 w-80 h-80 bg-cyan-400/10 blur-3xl rounded-full" />

            <div className="relative z-10 w-full max-w-5xl grid lg:grid-cols-2 gap-10 items-center">

                {/* Left Section */}
                <div className="hidden lg:block">

                    <div className="inline-flex items-center gap-3 mb-8">

                        <div className="p-4 rounded-2xl bg-white/10 border border-white/10">
                            <Film size={28} />
                        </div>

                        <div>
                            <h1 className="text-3xl font-bold">
                                StreamFlow
                            </h1>

                            <p className="text-gray-400 text-sm">
                                Upload & Stream Platform
                            </p>
                        </div>

                    </div>

                    <h2 className="text-5xl font-bold leading-tight mb-6">
                        Upload your videos
                        <span className="block text-gray-400">
                            instantly & securely.
                        </span>
                    </h2>

                    <p className="text-gray-400 text-lg leading-relaxed mb-10 max-w-lg">
                        Stream high-quality content with adaptive bitrate,
                        subtitles, cloud processing and fast multipart uploads.
                    </p>

                    <div className="grid grid-cols-3 gap-4">

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
                            <h3 className="text-2xl font-bold mb-1">
                                4K
                            </h3>

                            <p className="text-gray-400 text-sm">
                                Ultra HD
                            </p>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
                            <h3 className="text-2xl font-bold mb-1">
                                HLS
                            </h3>

                            <p className="text-gray-400 text-sm">
                                Adaptive Streaming
                            </p>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
                            <h3 className="text-2xl font-bold mb-1">
                                Cloud
                            </h3>

                            <p className="text-gray-400 text-sm">
                                Multipart Uploads
                            </p>
                        </div>

                    </div>
                </div>

                {/* Upload Card */}
                <div className="bg-white/[0.04] border border-white/10 backdrop-blur-2xl rounded-3xl p-8 shadow-2xl">

                    {/* Header */}
                    <div className="text-center mb-8">

                        <div className="inline-flex p-4 rounded-2xl bg-white/10 border border-white/10 mb-5">
                            <UploadCloud size={32} />
                        </div>

                        <h2 className="text-4xl font-bold mb-2">
                            Upload Video
                        </h2>

                        <p className="text-gray-400">
                            MP4 • Multipart upload • Cloud processing
                        </p>

                    </div>

                    {/* Upload Area */}
                    <label className="group relative flex flex-col items-center justify-center border-2 border-dashed border-white/10 hover:border-white/20 transition rounded-3xl p-10 cursor-pointer bg-white/[0.03]">

                        <input
                            type="file"
                            accept="video/mp4"
                            onChange={onFileChange}
                            className="hidden"
                        />

                        <div className="p-5 rounded-full bg-white/10 mb-5 group-hover:scale-105 transition">
                            <UploadCloud size={38} />
                        </div>

                        <h3 className="text-xl font-semibold mb-2">
                            Drag & drop your video
                        </h3>

                        <p className="text-gray-500 text-sm text-center">
                            Or click to browse from your device
                        </p>

                    </label>

                    {/* Description */}
                    <div className="mt-6">

                        <label className="block text-sm text-gray-300 mb-3">
                            Video Description
                        </label>

                        <textarea
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Write something about your video..."
                            rows={5}
                            className="w-full bg-black/30 border border-white/10 rounded-2xl px-4 py-4 text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-white/10 focus:border-white/20 transition"
                        />

                    </div>

                    {/* File Info */}
                    {file && (

                        <div className="mt-6 bg-white/5 border border-white/10 rounded-2xl p-4 flex items-center justify-between">

                            <div className="flex items-center gap-4">

                                <div className="p-3 rounded-xl bg-white/10">
                                    <Film size={22} />
                                </div>

                                <div>
                                    <p className="font-medium truncate max-w-[220px]">
                                        {file.name}
                                    </p>

                                    <p className="text-sm text-gray-500">
                                        {(file.size / 1024 / 1024).toFixed(2)} MB
                                    </p>
                                </div>

                            </div>

                            <CheckCircle2 className="text-green-400" />

                        </div>

                    )}

                    {/* Progress */}
                    {uploading && (

                        <div className="mt-6">

                            <div className="flex items-center justify-between mb-2">

                                <p className="text-sm text-gray-400">
                                    Uploading...
                                </p>

                                <p className="text-sm font-medium">
                                    {progress}%
                                </p>

                            </div>

                            <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden">

                                <div
                                    className="h-full bg-white transition-all duration-300"
                                    style={{
                                        width: `${progress}%`,
                                    }}
                                />

                            </div>

                        </div>

                    )}

                    {/* Upload Button */}
                    <button
                        onClick={upload}
                        disabled={!file || uploading}
                        className="w-full mt-8 bg-white text-black hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 py-4 rounded-2xl font-semibold text-lg flex items-center justify-center gap-3"
                    >

                        {uploading ? (
                            <>
                                <Loader2
                                    size={22}
                                    className="animate-spin"
                                />
                                Uploading...
                            </>
                        ) : (
                            <>
                                <UploadCloud size={22} />
                                Upload Video
                            </>
                        )}

                    </button>

                    {/* Status */}
                    {message && (

                        <div
                            className={`mt-5 rounded-2xl p-4 flex items-center gap-3 border ${message.includes("successful")
                                ? "bg-green-500/10 border-green-500/20 text-green-300"
                                : "bg-red-500/10 border-red-500/20 text-red-300"
                                }`}
                        >

                            {message.includes("successful") ? (
                                <CheckCircle2 size={20} />
                            ) : (
                                <XCircle size={20} />
                            )}

                            <span className="text-sm font-medium">
                                {message}
                            </span>

                        </div>

                    )}

                </div>
            </div>
        </div>
    );
}
