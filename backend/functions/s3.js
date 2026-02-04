import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import dotenv from 'dotenv';

dotenv.config();

const s3Client = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
});

export async function putObject(filename, ContentType) {
    const command = new PutObjectCommand({
        Bucket: process.env.AWS_RAW_BUCKET,
        Key: `videos/${filename}.mp4`,
        ContentType: ContentType
    });
    const url = await getSignedUrl(s3Client, command);
    return url;
}

async function getObject(key) {
    const command = new GetObjectCommand({
        Bucket: process.env.AWS_RAW_BUCKET,
        Key: key
    });
    const url = await getSignedUrl(s3Client, command);
    return url;
}
// async function init() {
//     console.log("S3 Module Initialized");
//     console.log("url for object is ", await getObject("/user/demo"));
// }

// init();

