// import { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import axios from 'axios';

// const Signup = () => {
//     const [email, setEmail] = useState("");
//     const [username, setUsername] = useState("");
//     const [password, setPassword] = useState("");
//     const navigate = useNavigate();

//     const handleSubmit = async (e) => {
//         e.preventDefault();
//         try {
//             await axios.post("http://localhost:3000/auth/register", { username, password, email });
//             navigate("/login");
//         } catch (error) {
//             console.error("Error occurred while signing up:", error);
//         }
//     }

//     return (
//         <div className="min-h-screen flex items-center justify-center bg-gray-900">
//             <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-md w-full max-w-sm">
//                 <h2 className="text-2xl font-bold text-white mb-4">Sign Up</h2>
//                 <div className="mb-4">
//                     <label className="block text-gray-300 text-sm font-bold mb-2" htmlFor="username">
//                         Username
//                     </label>
//                     <input
//                         id="username"
//                         type="text"
//                         value={username}
//                         onChange={(e) => setUsername(e.target.value)}
//                         className="bg-gray-700 text-gray-300 placeholder:text-gray-500 border border-gray-500 rounded py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
//                         placeholder="Enter your username"
//                     />
//                 </div>
//                 <div className="mb-4">
//                     <label className="block text-gray-300 text-sm font-bold mb-2" htmlFor="email">
//                         Email
//                     </label>
//                     <input
//                         id="email"
//                         type="email"
//                         value={email}
//                         onChange={(e) => setEmail(e.target.value)}
//                         className="bg-gray-700 text-gray-300 placeholder:text-gray-500 border border-gray-500 rounded py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
//                         placeholder="Enter your email"
//                     />
//                 </div>
//                 <div className="mb-4">
//                     <label className="block text-gray-300 text-sm font-bold mb-2" htmlFor="password">
//                         Password
//                     </label>
//                     <input
//                         id="password"
//                         type="password"
//                         value={password}
//                         onChange={(e) => setPassword(e.target.value)}
//                         className="bg-gray-700 text-gray-300 placeholder:text-gray-500 border border-gray-500 rounded py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
//                         placeholder="Enter your password"
//                     />
//                 </div>
//                 <button
//                     type="submit"
//                     className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
//                 >
//                     Sign Up
//                 </button>
//             </form>
//         </div>
//     );
// };

// export default Signup;
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Play, MonitorPlay } from "lucide-react";

const Signup = () => {
    const [email, setEmail] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");

    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            await axios.post(
                "https://videositebackend-production.up.railway.app/auth/register",
                {
                    username,
                    password,
                    email,
                }
            );

            navigate("/login");
        } catch (error) {
            console.error("Error occurred while signing up:", error);
        }
    };

    return (
        <div className="min-h-screen bg-[#0b0f19] text-white flex overflow-hidden">

            {/* Left Side */}
            <div className="hidden lg:flex w-1/2 relative items-center justify-center border-r border-white/10">

                {/* Glow */}
                <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500/10 blur-3xl rounded-full" />
                <div className="absolute bottom-10 right-10 w-80 h-80 bg-cyan-400/10 blur-3xl rounded-full" />

                <div className="relative z-10 max-w-xl px-12">

                    {/* Brand */}
                    <div className="flex items-center gap-4 mb-10">

                        <div className="p-4 rounded-2xl bg-white/10 border border-white/10 backdrop-blur-md">
                            <MonitorPlay size={32} />
                        </div>

                        <div>
                            <h1 className="text-3xl font-bold tracking-tight">
                                StreamFlow
                            </h1>

                            <p className="text-gray-400 text-sm">
                                Modern video streaming platform
                            </p>
                        </div>
                    </div>

                    {/* Heading */}
                    <h2 className="text-5xl font-bold leading-tight mb-6">
                        Start streaming
                        <span className="block text-gray-400">
                            without limits.
                        </span>
                    </h2>

                    <p className="text-gray-400 text-lg leading-relaxed mb-10">
                        Create your account to upload, manage and stream
                        high-quality videos with adaptive playback support.
                    </p>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-4">

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                            <h3 className="text-2xl font-bold mb-1">4K</h3>
                            <p className="text-gray-400 text-sm">
                                Ultra HD
                            </p>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                            <h3 className="text-2xl font-bold mb-1">HLS</h3>
                            <p className="text-gray-400 text-sm">
                                Adaptive bitrate
                            </p>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                            <h3 className="text-2xl font-bold mb-1">Cloud</h3>
                            <p className="text-gray-400 text-sm">
                                Storage ready
                            </p>
                        </div>

                    </div>
                </div>
            </div>

            {/* Right Side */}
            <div className="flex-1 flex items-center justify-center px-6 relative">

                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.04),transparent_30%)]" />

                <form
                    onSubmit={handleSubmit}
                    className="relative z-10 w-full max-w-md bg-white/[0.04] border border-white/10 backdrop-blur-2xl rounded-3xl p-8 shadow-2xl"
                >

                    {/* Logo */}
                    <div className="flex justify-center mb-8">

                        <div className="p-4 rounded-2xl bg-white/10 border border-white/10">
                            <Play size={28} fill="white" />
                        </div>

                    </div>

                    {/* Heading */}
                    <div className="text-center mb-8">

                        <h2 className="text-4xl font-bold mb-2">
                            Create Account
                        </h2>

                        <p className="text-gray-400">
                            Join the streaming platform
                        </p>

                    </div>

                    {/* Username */}
                    <div className="mb-5">

                        <label className="block text-sm text-gray-300 mb-2">
                            Username
                        </label>

                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Choose a username"
                            className="w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-white/20 focus:border-white/20 transition"
                        />

                    </div>

                    {/* Email */}
                    <div className="mb-5">

                        <label className="block text-sm text-gray-300 mb-2">
                            Email
                        </label>

                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Enter your email"
                            className="w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-white/20 focus:border-white/20 transition"
                        />

                    </div>

                    {/* Password */}
                    <div className="mb-6">

                        <label className="block text-sm text-gray-300 mb-2">
                            Password
                        </label>

                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Create a password"
                            className="w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-white/20 focus:border-white/20 transition"
                        />

                    </div>

                    {/* Submit */}
                    <button
                        type="submit"
                        className="w-full bg-white text-black hover:bg-gray-200 transition-all duration-300 py-3 rounded-xl font-semibold text-lg"
                    >
                        Create Account
                    </button>

                    {/* Footer */}
                    <p className="text-center text-gray-400 mt-8 text-sm">

                        Already have an account?{" "}

                        <span
                            onClick={() => navigate("/login")}
                            className="text-white hover:text-gray-300 cursor-pointer font-semibold"
                        >
                            Sign In
                        </span>

                    </p>

                </form>
            </div>
        </div>
    );
};

export default Signup;