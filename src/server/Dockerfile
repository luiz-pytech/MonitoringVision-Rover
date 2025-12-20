FROM node:18-alpine

WORKDIR /usr/src/app

# Copia arquivos de dependências
COPY package.json yarn.lock ./

# Instala dependências com yarn
RUN yarn install --production

# Copia código restante
COPY . .

EXPOSE 8080

CMD ["node", "index.js"]
