FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# ... ด้านบนปล่อยไว้เหมือนเดิมครับ ...

EXPOSE 3000

# ใช้คำสั่งนี้เพื่อตัดปัญหาระบบคัดกรองคำสั่งของ npm ออกไป
CMD ["npx", "next", "dev", "-H", "0.0.0.0", "-p", "3000"]