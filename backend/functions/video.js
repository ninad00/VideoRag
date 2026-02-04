import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "*");
    next();
});

app.use("/output", express.static(path.join(__dirname, "output")));

app.get("/", (req, res) => {
    res.send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HLS Player</title>
<link href="https://vjs.zencdn.net/8.10.0/video-js.css" rel="stylesheet">
<style>
body{margin:0;padding:20px;background:#121212;color:#fff;font-family:Arial}
.container{max-width:1200px;margin:auto}
.video-js{width:100%;aspect-ratio:16/9}
.controls{margin-top:20px;display:flex;gap:10px}
.btn{padding:10px 18px;background:#444;border:none;border-radius:6px;color:#fff;cursor:pointer}
.btn.active{background:#4CAF50}
</style>
</head>
<body>
<div class="container">
<h1>HLS Player</h1>
<video id="player" class="video-js vjs-default-skin vjs-big-play-centered" controls preload="auto"></video>
<div class="controls">
<button class="btn active" onclick="setQuality(0,this)">360p</button>
<button class="btn" onclick="setQuality(1,this)">720p</button>
<button class="btn" onclick="setQuality(2,this)">1080p</button>
</div>
</div>

<script src="https://vjs.zencdn.net/8.10.0/video.min.js"></script>
<script>
const player=videojs("player",{fluid:true,responsive:true,html5:{vhs:{overrideNative:true}}});
player.src({src:"/output/master.m3u8",type:"application/x-mpegURL"});
let levels;

window.setQuality=function(i,btn){
if(!levels)return;
for(let j=0;j<levels.length;j++)levels[j].enabled=false;
levels[i].enabled=true;
document.querySelectorAll(".btn").forEach(b=>b.classList.remove("active"));
btn.classList.add("active");
};

player.ready(()=>{
levels=player.qualityLevels();
for(let i=0;i<levels.length;i++)levels[i].enabled=false;
levels[0].enabled=true;
});
</script>
</body>
</html>`);
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
