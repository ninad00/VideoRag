import { spawn } from "child_process";
import fs from "fs";
import path from "path";

function runFFmpeg(args) {
    return new Promise((resolve, reject) => {
        const ff = spawn("ffmpeg", args);
        ff.stderr.on("data", d => process.stdout.write(d));
        ff.on("close", c => c === 0 ? resolve() : reject(c));
    });
}

function hasAudio(inputPath) {
    return new Promise(resolve => {
        const ff = spawn("ffprobe", [
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            inputPath
        ]);

        let found = false;
        ff.stdout.on("data", d => {
            if (d.toString().trim()) found = true;
        });

        ff.on("close", () => resolve(found));
    });
}

function hasSubtitles(inputPath) {
    return new Promise(resolve => {
        const ff = spawn("ffmpeg", ["-i", inputPath]);
        let found = false;
        ff.stderr.on("data", d => {
            if (d.toString().includes("Subtitle")) found = true;
        });
        ff.on("close", () => resolve(found));
    });
}

export async function processVideo(inputPath, outputDir) {
    fs.mkdirSync(outputDir, { recursive: true });

    ["v0", "v1", "v2"].forEach(d =>
        fs.mkdirSync(path.join(outputDir, d), { recursive: true })
    );

    const qualities = [
        { scale: "1920:1080", bitrate: "5000k", bw: 5128000, res: "1920x1080", idx: 0 },
        { scale: "1280:720", bitrate: "2800k", bw: 2928000, res: "1280x720", idx: 1 },
        { scale: "854:480", bitrate: "1400k", bw: 1528000, res: "854x480", idx: 2 }
    ];

    for (const q of qualities) {
        await runFFmpeg([
            "-i", inputPath,
            "-vf", `scale=${q.scale},format=yuv420p`,
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-profile:v", "main",
            "-crf", "20",
            "-g", "48",
            "-sc_threshold", "0",
            "-b:v", q.bitrate,
            "-an",
            "-f", "hls",
            "-hls_time", "4",
            "-hls_playlist_type", "vod",
            "-hls_flags", "independent_segments",
            "-hls_segment_filename", `${outputDir}/v${q.idx}/seg_%03d.ts`,
            `${outputDir}/v${q.idx}/index.m3u8`
        ]);
    }

    const audioExists = await hasAudio(inputPath);

    if (audioExists) {
        fs.mkdirSync(path.join(outputDir, "audio"), { recursive: true });

        await runFFmpeg([
            "-i", inputPath,
            "-map", "0:a:0",
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "hls",
            "-hls_time", "4",
            "-hls_playlist_type", "vod",
            "-hls_flags", "independent_segments",
            "-hls_segment_filename", `${outputDir}/audio/seg_%03d.ts`,
            `${outputDir}/audio/index.m3u8`
        ]);
    }

    const master = [
        "#EXTM3U",
        "#EXT-X-VERSION:3"
    ];

    if (audioExists) {
        master.push(
            "",
            "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"audio\",NAME=\"English\",DEFAULT=YES,AUTOSELECT=YES,URI=\"audio/index.m3u8\""
        );
    }

    for (const q of qualities) {
        master.push(
            "",
            `#EXT-X-STREAM-INF:BANDWIDTH=${q.bw},RESOLUTION=${q.res},CODECS="avc1.4d401f${audioExists ? ",mp4a.40.2" : ""}"${audioExists ? ',AUDIO="audio"' : ""}`,
            `v${q.idx}/index.m3u8`
        );
    }

    fs.writeFileSync(path.join(outputDir, "master.m3u8"), master.join("\n"));

    fs.mkdirSync(path.join(outputDir, "thumbnails"), { recursive: true });
    await runFFmpeg([
        "-i", inputPath,
        "-frames:v", "1",
        `${outputDir}/thumbnails/thumb.jpg`
    ]);


    if (await hasSubtitles(inputPath)) {
        fs.mkdirSync(path.join(outputDir, "subtitles"), { recursive: true });
        await runFFmpeg([
            "-i", inputPath,
            "-map", "0:s:0",
            `${outputDir}/subtitles/en.vtt`
        ]);
    }

    return { success: true, outputDir, audio: audioExists };
}


