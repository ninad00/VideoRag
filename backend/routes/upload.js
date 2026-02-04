import express from "express";
import { putPresignedUrl } from '../controllers/s3controller.js';
const router = express.Router();

router.post('/posturl', putPresignedUrl);
export default router;