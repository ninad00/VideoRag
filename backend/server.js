import express from "express";
import cors from "cors";
import upload from "./routes/upload.js";
import video from "./routes/video.js";
import deletes from "./routes/delete.js";

const PORT = 3000;

const app = express();
app.use(express.json());

app.use(cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use('/upload', upload);
app.use('/delete', deletes);
app.use('/video', video);

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});