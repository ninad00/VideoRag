// import {
//   SQSClient, ReceiveMessageCommand, GetQueueUrlCommand,
//   DeleteMessageCommand
// } from "@aws-sdk/client-sqs";
// import { S3Client, GetObjectCommand, PutObjectCommand, DeleteObjectCommand } from "@aws-sdk/client-s3";
// import fs from "fs";
// import path from "path";
// import { processVideo } from "./transcode.js";
// import "dotenv/config";

// const RAW_BUCKET = process.env.AWS_RAW_BUCKET;
// const OUT_BUCKET = process.env.AWS_OUT_BUCKET;
// const REGION = process.env.AWS_REGION;

// // AWS clients

// console.log("AWS_REGION =", process.env.AWS_REGION);
// // console.log("QUEUE_URL =", process.env.AWS_SQS_QUEUE_URL);

// const sqs = new SQSClient({ region: process.env.AWS_REGION });

// const { QueueUrl } = await sqs.send(
//   new GetQueueUrlCommand({
//     QueueName: "kyu" // EXACT queue name
//   })
// );

// console.log("Resolved QueueUrl:", QueueUrl);
// const QUEUE_URL = QueueUrl;

// const s3 = new S3Client({
//   region: REGION,
//   credentials: {
//     accessKeyId: process.env.AWS_ACCESS_KEY_ID,
//     secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
//   }
// });


// async function downloadFromS3(bucket, key, dest) {
//   const res = await s3.send(new GetObjectCommand({
//     Bucket: bucket,
//     Key: key
//   }));

//   await new Promise((resolve, reject) => {
//     const file = fs.createWriteStream(dest);
//     res.Body.pipe(file);
//     res.Body.on("error", reject);
//     file.on("finish", resolve);
//   });
// }

// async function uploadDir(dir, prefix) {
//   const files = fs.readdirSync(dir, { recursive: true });

//   for (const f of files) {
//     const full = path.join(dir, f);
//     if (fs.statSync(full).isFile()) {
//       await s3.send(new PutObjectCommand({
//         Bucket: OUT_BUCKET,
//         Key: `${prefix}/${f}`,
//         Body: fs.createReadStream(full)
//       }));
//     }
//   }
// }

// async function loop() {
//   console.log("Worker started, polling SQS…");

//   while (true) {
//     const { Messages } = await sqs.send(new ReceiveMessageCommand({
//       QueueUrl: QUEUE_URL,
//       MaxNumberOfMessages: 1,
//       WaitTimeSeconds: 20
//     }));

//     if (!Messages) continue;

//     const msg = Messages[0];

//     // ✅ CORRECT S3 EVENT PARSING
//     const event = JSON.parse(msg.Body);
//     const record = event.Records[0];

//     const bucket = record.s3.bucket.name;
//     const key = decodeURIComponent(
//       record.s3.object.key.replace(/\+/g, " ")
//     );

//     console.log("Processing:", bucket, key);

//     const input = "/tmp/input.mp4";
//     const output = "/tmp/output";

//     await downloadFromS3(bucket, key, input);

//     await processVideo(input, output);

//     const fileName = path.basename(key);
//     console.log("File name:", fileName);        // e.g. input.mp4
//     const baseName = path.parse(fileName).name; // e.g. input
//     const uploadPrefix = `${baseName}`;
//     console.log("Uploading to S3 with prefix:", uploadPrefix);
//     await uploadDir(output, uploadPrefix);

//     // delete raw file after success
//     await s3.send(new DeleteObjectCommand({
//       Bucket: bucket,
//       Key: key
//     }));
//     console.log("Deleted raw file from S3:", key);

//     await sqs.send(new DeleteMessageCommand({
//       QueueUrl: QUEUE_URL,
//       ReceiptHandle: msg.ReceiptHandle
//     }));
//     console.log("Deleted message from SQS:", msg.MessageId);

//     fs.rmSync("/tmp", { recursive: true, force: true });
//   }
// }

// loop().catch(console.error);
import {
  SQSClient,
  ReceiveMessageCommand,
  GetQueueUrlCommand,
  DeleteMessageCommand
} from "@aws-sdk/client-sqs";

import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
  DeleteObjectCommand
} from "@aws-sdk/client-s3";

import fs from "fs";
import path from "path";
import { exec } from "child_process";
import { processVideo } from "./transcode.js";
import "dotenv/config";

/* =========================
   ENV
========================= */

const RAW_BUCKET = process.env.AWS_RAW_BUCKET;
const OUT_BUCKET = process.env.AWS_OUT_BUCKET;
const REGION = process.env.AWS_REGION;
const MODAL_PROCESS_URL = process.env.MODAL_PROCESS_URL;

console.log("AWS_REGION =", REGION);

/* =========================
   AWS CLIENTS
========================= */

const sqs = new SQSClient({ region: REGION });

const { QueueUrl } = await sqs.send(
  new GetQueueUrlCommand({ QueueName: "kyu" })
);

console.log("Resolved QueueUrl:", QueueUrl);

const s3 = new S3Client({
  region: REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
  }
});

/* =========================
   HELPERS
========================= */

function getDuration(filePath) {
  return new Promise((resolve, reject) => {
    exec(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${filePath}"`,
      (err, stdout) => {
        if (err) return reject(err);
        resolve(Math.round(parseFloat(stdout)));
      }
    );
  });
}

async function downloadFromS3(bucket, key, dest) {
  const res = await s3.send(
    new GetObjectCommand({ Bucket: bucket, Key: key })
  );

  await new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    res.Body.pipe(file);
    res.Body.on("error", reject);
    file.on("finish", resolve);
  });
}

async function uploadDir(dir, prefix) {
  const files = fs.readdirSync(dir, { recursive: true });

  for (const f of files) {
    const full = path.join(dir, f);
    if (fs.statSync(full).isFile()) {
      await s3.send(
        new PutObjectCommand({
          Bucket: OUT_BUCKET,
          Key: `${prefix}/${f}`,
          Body: fs.createReadStream(full)
        })
      );
    }
  }
}

async function notifyModalProcess(videoId, duration) {
  const res = await fetch(MODAL_PROCESS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: videoId,
      duration
    })
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Modal process failed: ${res.status} ${text}`);
  }

  const json = await res.json();
  console.log("Modal process triggered:", json);
}

/* =========================
   MAIN LOOP
========================= */

async function loop() {
  console.log("Worker started, polling SQS…");

  while (true) {
    const { Messages } = await sqs.send(
      new ReceiveMessageCommand({
        QueueUrl,
        MaxNumberOfMessages: 1,
        WaitTimeSeconds: 20
      })
    );

    if (!Messages) continue;

    const msg = Messages[0];

    try {
      const event = JSON.parse(msg.Body);
      const record = event.Records[0];

      const bucket = record.s3.bucket.name;
      const key = decodeURIComponent(
        record.s3.object.key.replace(/\+/g, " ")
      );

      console.log("Processing:", bucket, key);

      const input = "/tmp/input.mp4";
      const output = "/tmp/output";

      await downloadFromS3(bucket, key, input);
      await processVideo(input, output);

      const fileName = path.basename(key);
      const baseName = path.parse(fileName).name;

      await uploadDir(output, baseName);

      const duration = await getDuration(input);
      console.log("Video duration (s):", duration);

      try {
        await notifyModalProcess(baseName, duration);
      } catch (err) {
        console.error("Modal RAG processing failed:", err.message);
      }

      await s3.send(
        new DeleteObjectCommand({ Bucket: bucket, Key: key })
      );

      await sqs.send(
        new DeleteMessageCommand({
          QueueUrl,
          ReceiptHandle: msg.ReceiptHandle
        })
      );

      console.log("Finished:", baseName);
    } catch (err) {
      console.error("Worker error:", err);
    } finally {
      try {
        fs.rmSync("/tmp/input.mp4", { force: true });
        fs.rmSync("/tmp/output", { recursive: true, force: true });
      } catch { }
    }
  }
}

loop().catch(console.error);
