import express from 'express';
import { deleteVideo } from '../controllers/videocontroller.js';

const router = express.Router();

router.delete('/:id', deleteVideo);

export default router;