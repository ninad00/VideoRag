import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./Homepage.jsx";
import PlayerPage from "./Video.jsx";
import Login from "./Login.jsx";
import UploadPage from "./upload.jsx";
import ProtectedRoute from "./routes/ProtectedRoute.jsx";
import { AuthProvider } from "./Context/authContext.jsx";
import Signup from "./SIgnup.jsx";
import Dashboard from "./Sidebar.jsx";

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/" element={<ProtectedRoute><HomePage /></ProtectedRoute>} />
          <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/watch/:id" element={<ProtectedRoute><PlayerPage /></ProtectedRoute>} />
          {/* <Route path="/upload" element={<ProtectedRoute><UploadPage /></ProtectedRoute>} /> */}
        </Routes>
      </Router>
    </AuthProvider>
  );
}
