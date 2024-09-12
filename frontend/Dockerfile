# Frontend Dockerfile

# Use the official Node.js image
FROM node:18

# Create app directory inside container
WORKDIR /app

# Clear any npm authentication from previous environments
RUN npm config set registry https://registry.npmjs.org/ \
    && npm config delete //registry.npmjs.org/:_authToken

# Install app dependencies
COPY package*.json ./
RUN npm install

# Copy the app source code
COPY . .

# Expose the app on port 3000
EXPOSE 3000

# Start the React app
CMD ["npm", "start"]

