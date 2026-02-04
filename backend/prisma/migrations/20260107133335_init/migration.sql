-- CreateTable
CREATE TABLE `Video` (
    `id` VARCHAR(191) NOT NULL,
    `title` VARCHAR(191) NULL,
    `description` VARCHAR(191) NULL,
    `thumbnail` VARCHAR(191) NULL,
    `hls` VARCHAR(191) NULL,
    `rawBucket` VARCHAR(191) NOT NULL,
    `outBucket` VARCHAR(191) NOT NULL,
    `status` ENUM('UPLOADED', 'READY') NOT NULL DEFAULT 'UPLOADED',
    `duration` DOUBLE NULL,
    `createdAt` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
