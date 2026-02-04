import { putObject } from '../functions/s3.js';
import  prisma  from '../prisma.js';
import dotenv from 'dotenv';
dotenv.config();


export async function putPresignedUrl(req, res) {
    try {
        const { filename, contentType } = req.body;
        if (!filename || !contentType) {
            return res.status(400).json({ error: 'Filename and contentType are required' });
        }
        if (contentType !== 'video/mp4') {
            return res.status(400).json({ error: 'Only video/mp4 content type is supported' });
        }
        const video = await prisma.video.create({
            data: {
                title: filename,
                rawBucket: process.env.AWS_RAW_BUCKET,
                outBucket: process.env.AWS_OUT_BUCKET,
                status: 'UPLOADED',
                createdAt: new Date(),
            },
        });
        const fileId = video.id;

        const url = await putObject(fileId, contentType);
        await prisma.video.update({
            where: { id: fileId },
            data: {
                status: 'READY',
                thumbnail: `https://${process.env.AWS_OUT_BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${fileId}/thumbnails/thumb.jpg`,
                hls: `https://${process.env.AWS_OUT_BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${fileId}/master.m3u8`
            },
        });

        return res.status(200).json({ url });
    } catch (error) {

        return res.status(500).json({ error: `${error.message}` });
    }
}

export async function getPresignedUrl(req, res) {
    try {
        const { filename, contentType } = req.body;
        if (!filename || !contentType) {
            return res.status(400).json({ error: 'Filename and contentType are required' });
        }
        if (contentType !== 'video/mp4') {
            return res.status(400).json({ error: 'Only video/mp4 content type is supported' });
        }
        const url = await putObject(filename, contentType);

        return res.status(200).json({ url });
    } catch (error) {

        return res.status(500).json({ error: `${error.message}` });
    }
}