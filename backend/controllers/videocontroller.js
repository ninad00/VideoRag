import prisma from '../prisma.js';

export async function getVideos(req, res) {
    try {
        const videos = await prisma.video.findMany({
            orderBy: { createdAt: 'desc' },
            select: {
                id: true,
                title: true,
                thumbnail: true,
                description: true,
                duration: true
            }
        });

        const data = videos.map(video => ({
            id: video.id,
            title: video.title,
            thumbnail_url: video.thumbnail,
            description: video.description,
            duration: video.duration
        }));

        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
}

export async function getVideoById(req, res) {
    try {
        const video = await prisma.video.findUnique({
            where: { id: req.params.id }
        });

        if (!video) {
            return res.status(404).json({ error: "Video not found" });
        }

        res.json({
            id: video.id,
            title: video.title,
            hls_url: video.hls,
            description: video.description,
            duration: video.duration
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
}

export async function deleteVideo(req, res) {
    try {
        const video = await prisma.video.delete({
            where: { id: req.params.id }
        })
        res.json({ message: "Video deleted successfully" })
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
}