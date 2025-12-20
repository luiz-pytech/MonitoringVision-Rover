// server/index.js
const WebSocket = require('ws');

const PORT = 8080;
const wss = new WebSocket.Server({ port: PORT });
console.log(`✅ WS relay running at ws://0.0.0.0:${PORT}`);

let robots = new Map(); // deviceId -> ws

wss.on('connection', (ws, req) => {
  console.log('Client connected:', req.socket.remoteAddress);

  ws.on('message', raw => {
    const msg = raw.toString();
    console.log('<<', msg);

    // simple protocol: registration and commands
    // registration: "register:robot:robot1"
    // controller sends: "cmd:robot1:FORWARD" etc.
    if (msg.startsWith('register:')) {
      const parts = msg.split(':'); // ["register","robot","deviceId"]
      if (parts[1] === 'robot' && parts[2]) {
        robots.set(parts[2], ws);
        ws.deviceId = parts[2];
        console.log(`🤖 Robot registered: ${parts[2]}`);
      }
      return;
    }

    if (msg.startsWith('cmd:')) {
      const parts = msg.split(':'); // ["cmd","deviceId","ACTION"]
      const target = parts[1];
      const action = parts.slice(2).join(':');
      const robotSocket = robots.get(target);
      if (robotSocket && robotSocket.readyState === WebSocket.OPEN) {
        robotSocket.send(action);
        console.log(`-> forwarded to ${target}: ${action}`);
      } else {
        console.log(`! robot ${target} not connected`);
      }
      return;
    }

    // If message from robot (just log)
    console.log('unhandled message:', msg);
  });

  ws.on('close', () => {
    if (ws.deviceId) {
      robots.delete(ws.deviceId);
      console.log(`Robot disconnected: ${ws.deviceId}`);
    } else {
      console.log('Client disconnected');
    }
  });
});
