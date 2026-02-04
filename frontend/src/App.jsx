import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./Homepage.jsx";
import PlayerPage from "./Video.jsx";
import UploadPage from "./upload.jsx";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/watch/:id" element={<PlayerPage />} />
        <Route path="/upload" element={<UploadPage />} />
      </Routes>
    </Router>
  );
}
