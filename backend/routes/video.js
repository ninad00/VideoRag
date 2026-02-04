import express from "express";

import { getVideos } from '../controllers/videocontroller.js';
import { getVideoById } from '../controllers/videocontroller.js';

const router = express.Router();

router.get('/', getVideos);
router.get('/:id', getVideoById);

export default router;