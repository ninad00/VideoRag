
import bcrypt from "bcrypt";
import prisma from "../prisma.js";
import { generateToken } from "../middlewares/jwt.js";
import { google } from 'googleapis';

import dotenv from 'dotenv';
dotenv.config();

const oauth2Client = new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    process.env.GOOGLE_REDIRECT_URI
);

async function hashPassword(password) {
    const saltRounds = 10;
    return await bcrypt.hash(password, saltRounds);
}

async function verifyPassword(password, hash) {
    return await bcrypt.compare(password, hash);
}

export const register = async (req, res) => {
    const { username, password, email } = req.body;
    if (!username || !password || !email) {
        return res.status(400).json({ error: 'Username, email, and password are required' });
    }
    try {
        const existingUser = await prisma.user.findUnique({ where: { username } });
        const existingEmail = await prisma.user.findUnique({ where: { email } });
        if (existingEmail) {
            return res.status(400).json({ error: 'Email already registered' });
        }
        if (existingUser) {
            return res.status(400).json({ error: 'Username already exists' });
        }
        const user = await prisma.user.create({
            data: {
                username: username,
                passwordHash: await hashPassword(password),
                email: email,
            }
        });
        res.json({ message: 'User registered successfully' });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal server error' });
    }
};



export const login = async (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) {
        return res.status(400).json({ error: 'Username and password are required' });
    }
    try {
        const user = await prisma.user.findUnique({ where: { username } });
        if (!user || !(await verifyPassword(password, user.passwordHash))) {
            return res.status(401).json({ error: 'Invalid username or password' });
        }
        const token = generateToken(user.username, user.id);

        res.cookie("token", token, {
            httpOnly: true,
            secure: true,
            sameSite: "none",
            maxAge: 7 * 24 * 60 * 60 * 1000
        });
        res.json({ message: "Login successful" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal server error' });
    }
};
export const googleAuth = async (req, res) => {
    const authUrl = oauth2Client.generateAuthUrl({
        access_type: 'offline',
        scope: [
            'https://www.googleapis.com/auth/userinfo.profile',
            'https://www.googleapis.com/auth/userinfo.email'
        ],
        prompt: 'consent'
    });
    res.redirect(authUrl);
};

export const googleAuthCallback = async (req, res) => {
    const { code } = req.query;

    if (!code) {
        return res.status(400).json({ error: 'Authorization code is required' });
    }

    try {
        console.log("Google callback hit");
        const { tokens } = await oauth2Client.getToken(code);
        oauth2Client.setCredentials(tokens);

        const oauth2 = google.oauth2({ version: 'v2', auth: oauth2Client });
        const { data: googleUser } = await oauth2.userinfo.get();
        const { id: googleId, email, picture } = googleUser;


        const user = await prisma.user.findUnique({ where: { email } });

        if (!user) {
            // Not registered — send them to sign up first
            return res.redirect(
                `${process.env.FRONTEND_URL}/login?error=no_account`
            );
        }

        // Email found — link googleId if this is their first Google login
        if (!user.googlesub) {
            await prisma.user.update({
                where: { email },
                data: { googlesub: googleId, avatarUrl: picture }
            });
        }

        const token = generateToken(user.username, user.id);
        res.cookie("token", token, {
            httpOnly: true,
            secure: true,
            sameSite: "none",
            maxAge: 7 * 24 * 60 * 60 * 1000
        });
        res.redirect(process.env.FRONTEND_URL + '/');// Redirect to homepage after successful login

    } catch (err) {
        console.error('Google OAuth error:', err);
        res.status(500).json({ error: 'Google authentication failed' });
    }
};

export const getMe = async (req, res) => {
    try {
        const user = await prisma.user.findUnique({ where: { id: req.user.id }, select: { id: true, username: true, email: true } });
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.json(user);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal server error' });
    }
};

export const logout = async (req, res) => {
    try {
        res.clearCookie("token", {
            httpOnly: true,
            secure: true,
            sameSite: "none",
        });

        res.status(200).json({
            message: "Logged out successfully",
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal server error' });
    }
};